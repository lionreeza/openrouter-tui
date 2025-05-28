package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/spf13/viper"

	"github.com/gdamore/tcell/v2"
	"github.com/rivo/tview"
)

// Message represents a chat message
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// CompletionRequest is the request payload for OpenRouter
type CompletionRequest struct {
	Model     string    `json:"model"`
	Messages  []Message `json:"messages"`
	Stream    bool      `json:"stream"`
	MaxTokens int       `json:"max_tokens,omitempty"`
}

// CompletionResponse represents each chunk of streaming response
type CompletionResponse struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
	} `json:"choices"`
}

// Config holds the application configuration
type Config struct {
	OpenRouter struct {
		APIKey    string `mapstructure:"api_key"`
		Model     string `mapstructure:"model"`
		Timeout   int    `mapstructure:"timeout"`
		MaxTokens int    `mapstructure:"max_tokens"`
	} `mapstructure:"openrouter"`
}

// ChatUI manages the terminal UI
type ChatUI struct {
	app            *tview.Application
	chatHistory    *tview.TextView
	inputField     *tview.InputField
	statusBar      *tview.TextView
	loadingSpinner *tview.TextView
	flex           *tview.Flex
	client         *http.Client
	cfg            *Config
	messages       []Message
	mu             sync.Mutex
	loadingActive  bool
	assistantText  *strings.Builder // Buffer for assistant's current response
}

func loadConfig() (*Config, error) {
	v := viper.New()
	v.SetConfigName("config")
	v.SetConfigType("yaml")
	v.AddConfigPath(".")
	v.AddConfigPath("$HOME/.openrouter")

	if err := v.ReadInConfig(); err != nil {
		return nil, fmt.Errorf("failed to read config: %w", err)
	}

	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	return &cfg, nil
}

func NewChatUI(cfg *Config) *ChatUI {
	return &ChatUI{
		app:      tview.NewApplication(),
		cfg:      cfg,
		messages: []Message{},
		client: &http.Client{
			Timeout: time.Duration(cfg.OpenRouter.Timeout) * time.Second,
		},
	}
}

func (ui *ChatUI) SetupUI() {
	// Configure the chat history view
	ui.chatHistory = tview.NewTextView().
		SetDynamicColors(true).
		SetRegions(true).
		SetWordWrap(true).
		SetScrollable(true).
		SetChangedFunc(func() {
			ui.app.Draw()
		})
	ui.chatHistory.SetBorder(true).SetTitle(" Conversation ").SetBorderColor(tcell.ColorBlue)

	// Configure the loading spinner
	ui.loadingSpinner = tview.NewTextView()
	ui.loadingSpinner.SetTextAlign(tview.AlignCenter)

	// Configure input field
	ui.inputField = tview.NewInputField().
		SetLabel("You: ").
		SetFieldWidth(0).
		SetFieldBackgroundColor(tcell.ColorBlack)
	ui.inputField.SetBorder(true).SetTitle(" Input ").SetTitleAlign(tview.AlignLeft).SetBorderColor(tcell.ColorGreen)

	// Configure status bar
	ui.statusBar = tview.NewTextView()
	ui.statusBar.SetTextAlign(tview.AlignRight).SetTextColor(tcell.ColorYellow)
	ui.UpdateStatus(fmt.Sprintf("Model: %s | Status: Ready", ui.cfg.OpenRouter.Model))

	// Layout
	ui.flex = tview.NewFlex().
		SetDirection(tview.FlexRow).
		AddItem(ui.chatHistory, 0, 1, false).
		AddItem(ui.loadingSpinner, 1, 0, false).
		AddItem(ui.inputField, 3, 1, true).
		AddItem(ui.statusBar, 1, 1, false)

	// Set input handler
	ui.inputField.SetDoneFunc(func(key tcell.Key) {
		if key == tcell.KeyEnter {
			text := ui.inputField.GetText()
			if text != "" {
				ui.handleInput(text)
			}
			ui.inputField.SetText("")
		}
	})

	// Ctrl-C to quit
	ui.app.SetInputCapture(func(event *tcell.EventKey) *tcell.EventKey {
		if event.Key() == tcell.KeyCtrlC {
			ui.app.Stop()
			return nil
		}
		return event
	})
}

func (ui *ChatUI) Run() error {
	ui.SetupUI()
	return ui.app.SetRoot(ui.flex, true).EnableMouse(true).Run()
}

func (ui *ChatUI) UpdateStatus(text string) {
	ui.statusBar.SetText(text)
}

func (ui *ChatUI) StartLoading() {
	ui.mu.Lock()
	defer ui.mu.Unlock()

	ui.loadingActive = true
	ui.inputField.SetDisabled(true)
	ui.assistantText = &strings.Builder{}

	// Start spinner animation
	go func() {
		frames := []string{"⠋", "⠙", "⠹", "⠸", "⢰", "⣠", "⣄", "⣆", "⡆", "⠇"}
		frameIdx := 0

		for ui.loadingActive {
			text := fmt.Sprintf(" %s Generating... %s ", frames[frameIdx], frames[frameIdx])
			ui.app.QueueUpdateDraw(func() {
				ui.loadingSpinner.SetText(text)
			})
			frameIdx = (frameIdx + 1) % len(frames)
			time.Sleep(100 * time.Millisecond)
		}
	}()
}

func (ui *ChatUI) StopLoading() {
	ui.mu.Lock()
	defer ui.mu.Unlock()
	ui.loadingActive = false
	ui.inputField.SetDisabled(false)
	ui.app.SetFocus(ui.inputField)
}

func (ui *ChatUI) AddMessage(role, content string) {
	ui.messages = append(ui.messages, Message{Role: role, Content: content})
}

func (ui *ChatUI) AppendToChat(role, text string) {
	switch role {
	case "You":
		fmt.Fprintf(ui.chatHistory, "[purple]You:[-] [white]%s[-]\n", text)
	case "Assistant":
		fmt.Fprintf(ui.chatHistory, "[blue]Assistant:[-] [white]%s[-]\n", text)
	case "System":
		fmt.Fprintf(ui.chatHistory, "[red]System:[-] %s\n", text)
	default:
		fmt.Fprintf(ui.chatHistory, "[white]%s:[-] %s\n", role, text)
	}
	ui.chatHistory.ScrollToEnd()
}

func (ui *ChatUI) AppendToChatPlain(role, text string) {
	fmt.Fprintf(ui.chatHistory, "[blue]Assistant:[-] [yellow]%s", text)
	ui.chatHistory.ScrollToEnd()
}

func (ui *ChatUI) handleInput(input string) {
	// Add user message to history and display (purple role)
	ui.AddMessage("user", input)
	ui.AppendToChat("You", input)

	// Start loading animation and begin assistant response line
	ui.StartLoading()
	ui.AppendToChatPlain("Assistant", "")

	// Send request to OpenRouter in a separate goroutine
	go func() {
		// Prepare request payload
		reqBody := CompletionRequest{
			Model:     ui.cfg.OpenRouter.Model,
			Messages:  ui.messages,
			Stream:    true,
			MaxTokens: ui.cfg.OpenRouter.MaxTokens,
		}

		jsonBody, err := json.Marshal(reqBody)
		if err != nil {
			ui.handleStreamError("Request serialization error: " + err.Error())
			return
		}

		// Create HTTP request
		req, err := http.NewRequest("POST", "https://openrouter.ai/api/v1/chat/completions",
			bytes.NewReader(jsonBody))
		if err != nil {
			ui.handleStreamError("Request creation error: " + err.Error())
			return
		}

		req.Header.Set("Authorization", "Bearer "+ui.cfg.OpenRouter.APIKey)
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("HTTP-Referer", "github.com/reVost/go-openrouter")
		req.Header.Set("X-Title", "Go OpenRouter Client")

		// Make the streaming request
		resp, err := ui.client.Do(req)
		if err != nil {
			ui.handleStreamError("API request error: " + err.Error())
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			// Read the error response
			errBody, _ := io.ReadAll(resp.Body)
			errMsg := fmt.Sprintf("API error: %s - %s", resp.Status, string(errBody))
			ui.handleStreamError(errMsg)
			return
		}

		// Process SSE streaming response
		reader := bufio.NewReader(resp.Body)
		isFirst := true

		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				if errors.Is(err, io.EOF) {
					break
				}
				ui.appendSystemError("Stream read error: " + err.Error())
				break
			}

			// Skip empty lines and SSE comment lines
			if strings.TrimSpace(line) == "" || strings.HasPrefix(line, ":") {
				continue
			}

			// Check for data prefix
			if strings.HasPrefix(line, "data:") {
				jsonStr := strings.TrimPrefix(line, "data:")
				jsonStr = strings.TrimSpace(jsonStr)

				// Check for special "[DONE]" message
				if jsonStr == "[DONE]" {
					break
				}

				var chunk CompletionResponse
				if err := json.Unmarshal([]byte(jsonStr), &chunk); err != nil {
					ui.appendSystemError("JSON parse error: " + err.Error())
					continue
				}

				if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
					delta := chunk.Choices[0].Delta.Content
					ui.assistantText.WriteString(delta)

					// Update UI with the new content
					ui.app.QueueUpdateDraw(func() {
						// For the first chunk, setup the assistant line
						if isFirst {
							ui.AppendToChatPlain("Assistant", "[yellow]")
							isFirst = false
						}

						ui.appendDelta(delta)
					})
				}
			}
		}

		// Add final assistant response
		finalResponse := ui.assistantText.String()
		if finalResponse != "" {
			ui.app.QueueUpdateDraw(func() {
				// Save assistant response to message history
				ui.AddMessage("assistant", finalResponse)
				ui.StopLoading()

				// Remove the streaming marker and reset to white
				ui.finalizeAssistantMessage()
			})
		} else {
			ui.StopLoading()
		}
	}()
}

// appendDelta adds a new chunk to the assistant response in yellow
func (ui *ChatUI) appendDelta(delta string) {
	currentText := ui.chatHistory.GetText(true)

	// Find last line and append new content
	lines := strings.Split(currentText, "\n")
	if len(lines) > 0 {
		lastLine := lines[len(lines)-1]

		// Handle the case where we might have ANSI codes at the end
		if strings.HasSuffix(lastLine, "[-]") {
			lastLine = strings.TrimSuffix(lastLine, "[-]")
		}

		// Append the new delta
		lastLine += delta

		// Note: We explicitly keep yellow open for additional tokens
		lastLine += "[yellow]"

		// Update the last line
		lines[len(lines)-1] = lastLine
		ui.chatHistory.SetText(strings.Join(lines, "\n"))
	}

	ui.chatHistory.ScrollToEnd()
}

// finalizeAssistantMessage removes yellow coloring from the assistant's message
func (ui *ChatUI) finalizeAssistantMessage() {
	currentText := ui.chatHistory.GetText(true)
	lines := strings.Split(currentText, "\n")

	if len(lines) == 0 {
		return
	}

	// Process the last line (assistant's response)
	lastLine := lines[len(lines)-1]

	// Remove any open yellow tags and their endings
	lastLine = strings.ReplaceAll(lastLine, "[yellow]", "")
	lastLine = strings.ReplaceAll(lastLine, "[-]", "")

	// Recolor assistant response to white
	lines[len(lines)-1] = lastLine
	ui.chatHistory.SetText(strings.Join(lines, "\n") + "\n")
	ui.chatHistory.ScrollToEnd()
}

func (ui *ChatUI) handleStreamError(msg string) {
	ui.app.QueueUpdateDraw(func() {
		ui.StopLoading()
		ui.AppendToChat("System", "Error: "+msg)
	})
}

func (ui *ChatUI) appendSystemError(msg string) {
	ui.app.QueueUpdateDraw(func() {
		ui.AppendToChat("System", msg)
	})
}

func main() {
	cfg, err := loadConfig()
	if err != nil {
		log.Fatalf("Config Error: %v", err)
	}

	ui := NewChatUI(cfg)
	if err := ui.Run(); err != nil {
		log.Fatalf("TUI Error: %v", err)
	}
}
