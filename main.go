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
	"os"
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

	v.SetDefault("openrouter.model", "openai/gpt-3.5-turbo")
	v.SetDefault("openrouter.timeout", 30)
	v.SetDefault("openrouter.max_tokens", 512)

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
	ui.chatHistory.SetText("Welcome to OpenRouter Chat! Enter your message below and press Enter to send.")

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
		fmt.Fprintf(ui.chatHistory, "[purple]You:[-] [white]%s\n", text)
	case "Assistant":
		fmt.Fprintf(ui.chatHistory, "[blue]Assistant:[-] [white]%s\n", text)
	case "System":
		fmt.Fprintf(ui.chatHistory, "[red]System:[-] %s\n", text)
	default:
		fmt.Fprintf(ui.chatHistory, "[white]%s:[-] %s\n", role, text)
	}
	ui.chatHistory.ScrollToEnd()
}

func (ui *ChatUI) AppendPartialAssistant(prefix, text string) {
	fmt.Fprintf(ui.chatHistory, "[blue]Assistant:[-] %s%s", prefix, text)
	ui.chatHistory.ScrollToEnd()
}

func (ui *ChatUI) handleInput(input string) {
	// Add user message to history and display (purple role)
	ui.AddMessage("user", input)
	ui.AppendToChat("You", input)

	// Start loading animation and begin assistant response line
	ui.StartLoading()

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
			errBody, err := io.ReadAll(resp.Body)
			if err != nil {
				ui.handleStreamError("Failed to read error response: " + err.Error())
			} else {
				ui.handleStreamError(fmt.Sprintf("API error: %s - %s", resp.Status, string(errBody)))
			}
			return
		}

		// Process SSE streaming response
		reader := bufio.NewReader(resp.Body)
		var responseStarted bool

		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				if errors.Is(err, io.EOF) {
					break
				}
				log.Printf("Stream read error: %v", err)
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

				// Check for special "[极ONE]" message
				if jsonStr == "[DONE]" {
					break
				}

				var chunk CompletionResponse
				if err := json.Unmarshal([]byte(jsonStr), &chunk); err != nil {
					log.Printf("JSON parse error: %v", err)
					continue
				}

				if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
					delta := chunk.Choices[0].Delta.Content
					ui.assistantText.WriteString(delta)

					if !responseStarted {
						responseStarted = true
						ui.app.QueueUpdateDraw(func() {
							ui.AppendPartialAssistant("[yellow]", delta)
						})
					} else {
						ui.app.QueueUpdateDraw(func() {
							fmt.Fprint(ui.chatHistory, "[yellow]"+delta)
							ui.chatHistory.ScrollToEnd()
						})
					}
				}
			}
		}

		// Handle the final response
		ui.app.QueueUpdateDraw(func() {
			finalResponse := ui.assistantText.String()
			responseText := strings.TrimSpace(finalResponse)

			if responseText != "" {
				// Update the assistant message with final formatting
				ui.addCompletedAssistantMessage(finalResponse)
			} else if !responseStarted {
				log.Println("Assistant returned an empty response")
				ui.AppendToChat("System", "Assistant returned an empty response")
				fmt.Fprint(ui.chatHistory, "\n") // Ensure new line
			}

			ui.StopLoading()
			ui.chatHistory.ScrollToEnd()
		})
	}()
}

func (ui *ChatUI) addCompletedAssistantMessage(text string) {
	// Get current text content
	current := ui.chatHistory.GetText(true)

	// Remove yellow formatting from last line
	lines := strings.Split(current, "\n")

	// Only process if we have at least one line
	if len(lines) > 0 {
		lastLine := lines[len(lines)-1]

		// Clean up yellow styling markers
		lastLine = strings.TrimSuffix(lastLine, "[yellow]")
		lines = lines[:len(lines)-1]

		// Reconstruct the chat history text
		ui.chatHistory.SetText(strings.Join(lines, "\n") + "\n")
	}

	text = strings.TrimSuffix(text, "\n")
	ui.AppendToChat("Assistant", text)
}

func (ui *ChatUI) handleStreamError(msg string) {
	ui.app.QueueUpdateDraw(func() {
		ui.StopLoading()
		ui.AppendToChat("System", msg)
	})
}

func main() {
	cfg, err := loadConfig()
	if err != nil {
		log.Printf("Config error: %v", err)

		// Try to create default config if it doesn't exist
		if _, ok := err.(viper.ConfigFileNotFoundError); ok {
			log.Println("Creating default config file...")

			// Create file if it doesn't exist
			if _, err := os.Stat("config.yaml"); os.IsNotExist(err) {
				if _, createErr := os.Create("config.yaml"); createErr != nil {
					log.Fatalf("Failed to create config file: %v", createErr)
				}
			}

			// Set default values
			v := viper.New()
			v.SetConfigFile("config.yaml")
			v.Set("openrouter", map[string]interface{}{
				"api_key":    "your-api-key-here",
				"model":      "openai/gpt-3.5-turbo",
				"timeout":    30,
				"max_tokens": 512,
			})

			if err := v.WriteConfig(); err != nil {
				log.Fatalf("Failed to write config: %v", err)
			}

			log.Println("Created default config.yaml file. Please update with your API key.")
			log.Println("Rerun the application after editing config.yaml")
			os.Exit(0)
		} else {
			log.Fatalf("Fatal config error: %v", err)
		}
	}

	ui := NewChatUI(cfg)
	if err := ui.Run(); err != nil {
		log.Fatalf("TUI Error: %v", err)
	}
}
