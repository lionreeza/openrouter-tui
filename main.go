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
	"unicode"

	"github.com/gdamore/tcell/v2"
	"github.com/rivo/tview"
	"github.com/spf13/viper"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type CompletionRequest struct {
	Model     string    `json:"model"`
	Messages  []Message `json:"messages"`
	Stream    bool      `json:"stream"`
	MaxTokens int       `json:"max_tokens,omitempty"`
}

type CompletionResponse struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
	} `json:"choices"`
}

type Config struct {
	OpenRouter struct {
		APIKey    string `mapstructure:"api_key"`
		Model     string `mapstructure:"model"`
		Timeout   int    `mapstructure:"timeout"`
		MaxTokens int    `mapstructure:"max_tokens"`
	} `mapstructure:"openrouter"`
}

// MarkdownParser handles Markdown rendering for assistant responses
type MarkdownParser struct {
	inBold      bool
	inItalic    bool
	inCode      bool
	inQuote     bool
	inList      bool
	buffer      *strings.Builder
	partialMode bool // For streaming mode
}

func NewMarkdownParser() *MarkdownParser {
	return &MarkdownParser{
		buffer: &strings.Builder{},
	}
}

func (p *MarkdownParser) Reset() {
	p.inBold = false
	p.inItalic = false
	p.inCode = false
	p.inQuote = false
	p.inList = false
	p.buffer.Reset()
}

// RenderMarkdown renders complete text
func (p *MarkdownParser) RenderMarkdown(text string) []byte {
	return p.renderInternal(text, false)
}

// RenderPartial renders text incrementally (for streaming)
func (p *MarkdownParser) RenderPartial(text string) []byte {
	return p.renderInternal(text, true)
}

func (p *MarkdownParser) renderInternal(text string, partial bool) []byte {
	p.partialMode = partial
	p.Reset()
	lines := strings.Split(text, "\n")
	output := &strings.Builder{}
	prevLineEmpty := true

	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		lineEmpty := trimmed == ""

		// Skip consecutive empty lines in partial mode
		if lineEmpty && prevLineEmpty && partial {
			continue
		}
		prevLineEmpty = lineEmpty

		if strings.HasPrefix(trimmed, "```") {
			continue
		}

		if strings.HasPrefix(trimmed, "|") && strings.Contains(trimmed, "|") {
			// Handle tables
			if i > 0 && strings.HasPrefix(strings.TrimSpace(lines[i-1]), "|") {
				p.buffer.Reset()
				cells := strings.Split(trimmed, "|")
				for _, cell := range cells {
					cell = strings.TrimSpace(cell)
					if cell != "" {
						fmt.Fprintf(p.buffer, "[::b]%s[::-] ", cell)
					}
				}
				output.WriteString(p.buffer.String() + "\n")
			}
		} else if strings.HasPrefix(trimmed, "> ") {
			if !p.inQuote {
				p.buffer.WriteString("[darkcyan]│ [::-]")
				p.inQuote = true
			}
			content := filteredString(trimmed[2:])
			p.markdownLine(content)
			output.WriteString(p.buffer.String() + "\n")
		} else if strings.HasPrefix(trimmed, "- ") || strings.HasPrefix(trimmed, "* ") {
			if !p.inList {
				p.buffer.WriteString(" • ")
				p.inList = true
			}
			content := filteredString(trimmed[2:])
			p.markdownLine(content)
			output.WriteString(p.buffer.String() + "\n")
		} else if lineEmpty {
			if p.inList {
				p.inList = false
			}
			if p.inQuote {
				p.inQuote = false
			}
			output.WriteString("\n")
		} else {
			content := filteredString(line)
			p.markdownLine(content)
			output.WriteString(p.buffer.String() + "\n")
		}
	}

	return []byte(output.String())
}

func filteredString(s string) string {
	return strings.Map(func(r rune) rune {
		if unicode.IsPrint(r) {
			return r
		}
		return -1
	}, s)
}

func (p *MarkdownParser) markdownLine(line string) {
	p.buffer.Reset()
	active := false

	for i := 0; i < len(line); i++ {
		if i > 0 && line[i-1] == '\\' {
			continue
		}

		switch {
		case strings.HasPrefix(line[i:], "**") && !p.inCode:
			if active {
				p.buffer.WriteString("[::-][white]")
				active = false
				i++
			} else {
				p.buffer.WriteString("[::b][white]")
				active = true
				i++
			}
		case strings.HasPrefix(line[i:], "__") && !p.inCode:
			if active {
				p.buffer.WriteString("[::-][white]")
				active = false
				i++
			} else {
				p.buffer.WriteString("[::u][white]")
				active = true
				i++
			}
		case line[i] == '*' && !p.inCode:
			if active {
				p.buffer.WriteString("[::-][white]")
				active = false
			} else {
				p.buffer.WriteString("[::i][white]")
				active = true
			}
		case line[i] == '_' && !p.inCode:
			if active {
				p.buffer.WriteString("[::-][white]")
				active = false
			} else {
				p.buffer.WriteString("[::i][white]")
				active = true
			}
		case strings.HasPrefix(line[i:], "`") && !p.inCode && !p.partialMode:
			// Only handle code blocks in non-streaming mode
			if !p.inCode {
				p.buffer.WriteString("[::r]")
				p.inCode = true
				active = !active
			} else {
				p.buffer.WriteString("[::-][white]")
				p.inCode = false
				active = !active
			}
			i += 1
		default:
			p.buffer.WriteByte(line[i])
		}
	}

	if active {
		p.buffer.WriteString("[::-]")
	}
}

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
	assistantText  *strings.Builder
	markdownParser *MarkdownParser
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

	// Validate API key
	if cfg.OpenRouter.APIKey == "" || cfg.OpenRouter.APIKey == "your-api-key-here" {
		return nil, fmt.Errorf("API key is not configured. Please update config.yaml")
	}

	return &cfg, nil
}

func NewChatUI(cfg *Config) *ChatUI {
	return &ChatUI{
		app:            tview.NewApplication(),
		cfg:            cfg,
		messages:       []Message{},
		markdownParser: NewMarkdownParser(),
		client: &http.Client{
			Timeout: time.Duration(cfg.OpenRouter.Timeout) * time.Second,
		},
	}
}

func (ui *ChatUI) SetupUI() {
	ui.chatHistory = tview.NewTextView().
		SetDynamicColors(true).
		SetRegions(true).
		SetWordWrap(true).
		SetScrollable(true).
		SetChangedFunc(func() {
			ui.app.Draw()
		})
	ui.chatHistory.SetBorder(true).SetTitle(" Conversation ").SetBorderColor(tcell.ColorBlue)
	ui.chatHistory.SetText("Welcome to OpenRouter Chat!\nEnter your message below and press Enter to send.")

	ui.loadingSpinner = tview.NewTextView()
	ui.loadingSpinner.SetTextAlign(tview.AlignCenter)

	ui.inputField = tview.NewInputField().
		SetLabel("You: ").
		SetFieldWidth(0).
		SetFieldBackgroundColor(tcell.ColorBlack)
	ui.inputField.SetBorder(true).SetTitle(" Input ").SetTitleAlign(tview.AlignLeft).SetBorderColor(tcell.ColorGreen)

	ui.statusBar = tview.NewTextView()
	ui.statusBar.SetTextAlign(tview.AlignRight).SetTextColor(tcell.ColorYellow)
	ui.UpdateStatus(fmt.Sprintf("Model: %s | Status: Ready", ui.cfg.OpenRouter.Model))

	ui.flex = tview.NewFlex().
		SetDirection(tview.FlexRow).
		AddItem(ui.chatHistory, 0, 1, false).
		AddItem(ui.loadingSpinner, 1, 0, false).
		AddItem(ui.inputField, 3, 1, true).
		AddItem(ui.statusBar, 1, 1, false)

	ui.inputField.SetDoneFunc(func(key tcell.Key) {
		if key == tcell.KeyEnter {
			text := ui.inputField.GetText()
			if text != "" {
				ui.handleInput(text)
			}
			ui.inputField.SetText("")
		}
	})

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
	return ui.app.SetRoot(ui.flex, true).SetFocus(ui.inputField).EnableMouse(true).Run()
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

// AppendToChat renders and displays a message in the chat view
func (ui *ChatUI) AppendToChat(role, text string) {
	switch role {
	case "You":
		fmt.Fprintf(ui.chatHistory, "[purple]You:[-] [white]%s\n", text)
	case "Assistant":
		formatted := ui.markdownParser.RenderMarkdown(text)
		fmt.Fprintf(ui.chatHistory, "[blue]Assistant:[-] %s\n", formatted)
	case "System":
		fmt.Fprintf(ui.chatHistory, "[red]System:[-] %s\n", text)
	default:
		fmt.Fprintf(ui.chatHistory, "[white]%s:[-] %s\n", role, text)
	}
	ui.chatHistory.ScrollToEnd()
}

// AppendPartial appends streaming text with markdown applied
func (ui *ChatUI) AppendPartialAssistant(text string) {
	// Render partial markdown for streaming
	formatted := ui.markdownParser.RenderPartial(text)
	fmt.Fprintf(ui.chatHistory, "[blue]Assistant:[-] %s", formatted)
	ui.chatHistory.ScrollToEnd()
}

func (ui *ChatUI) handleInput(input string) {
	ui.AddMessage("user", input)
	ui.AppendToChat("You", input)
	ui.StartLoading()

	go func() {
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

		req, err := http.NewRequest("POST", "https://openrouter.ai/api/v1/chat/completions",
			bytes.NewReader(jsonBody))
		if err != nil {
			ui.handleStreamError("Request creation error: " + err.Error())
			return
		}

		// Trim API key
		apiKey := strings.TrimSpace(ui.cfg.OpenRouter.APIKey)

		req.Header.Set("Authorization", "Bearer "+apiKey)
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("HTTP-Referer", "github.com/reVost/go-openrouter")
		req.Header.Set("X-Title", "Go OpenRouter Client")

		log.Printf("Using model: %s", ui.cfg.OpenRouter.Model)
		if len(apiKey) > 8 {
			log.Printf("Using API key: %s...%s", apiKey[:4], apiKey[len(apiKey)-4:])
		}

		resp, err := ui.client.Do(req)
		if err != nil {
			ui.handleStreamError("API request error: " + err.Error())
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			errBody, _ := io.ReadAll(resp.Body)
			ui.handleStreamError(fmt.Sprintf("API error (%d): %s", resp.StatusCode, string(errBody)))
			return
		}

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

			// Skip empty lines and SSE comments
			if strings.TrimSpace(line) == "" || strings.HasPrefix(line, ":") {
				continue
			}

			if strings.HasPrefix(line, "data:") {
				jsonStr := strings.TrimPrefix(line, "data:")
				jsonStr = strings.TrimSpace(jsonStr)

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
							ui.AppendPartialAssistant(delta)
						})
					} else {
						ui.app.QueueUpdateDraw(func() {
							ui.AppendPartialAssistant(delta)
						})
					}
				}
			}
		}

		ui.app.QueueUpdateDraw(func() {
			// Add full message with final markdown rendering
			finalResponse := ui.assistantText.String()
			if finalResponse != "" {
				ui.AddMessage("assistant", finalResponse)
				ui.AddCompletedAssistantMessage(finalResponse)
			} else if !responseStarted {
				ui.AppendToChat("System", "Assistant returned an empty response")
			}

			ui.StopLoading()
		})
	}()
}

func (ui *ChatUI) AddCompletedAssistantMessage(text string) {
	ui.AppendToChat("Assistant", text)
}

func (ui *ChatUI) handleStreamError(msg string) {
	ui.app.QueueUpdateDraw(func() {
		ui.StopLoading()
		ui.AppendToChat("System", "Error: "+msg)
	})
}

func main() {
	cfg, err := loadConfig()
	if err != nil {
		log.Printf("Config error: %v", err)

		if strings.Contains(err.Error(), "API key is not configured") {
			log.Println("Please update config.yaml with your API key and restart")
			log.Println("You can get your API key at: https://openrouter.ai/keys")
			os.Exit(1)
		}

		if _, ok := err.(viper.ConfigFileNotFoundError); ok {
			log.Println("Creating default config file...")
			if _, err := os.Stat("config.yaml"); os.IsNotExist(err) {
				file, createErr := os.Create("config.yaml")
				if createErr != nil {
					log.Fatalf("Failed to create config file: %v", createErr)
				}
				file.Close()
			}

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

			log.Println("Created config.yaml. Please update with your API key")
			log.Println("Rerun the application after setup")
			os.Exit(0)
		} else {
			log.Fatalf("Fatal config error: %v", err)
		}
	}

	log.Printf("Loaded model: %s", cfg.OpenRouter.Model)
	if len(cfg.OpenRouter.APIKey) > 8 {
		log.Printf("Using API key: %s...%s", cfg.OpenRouter.APIKey[:4], cfg.OpenRouter.APIKey[len(cfg.OpenRouter.APIKey)-4:])
	}

	ui := NewChatUI(cfg)
	if err := ui.Run(); err != nil {
		log.Fatalf("UI Error: %v", err)
	}
}
