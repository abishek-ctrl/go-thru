// Package server provides an OpenAI-compatible HTTP API.
package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/abishekm/go-thru/engine/generate"
)

// Config for HTTP server.
type Config struct {
	Addr           string
	ModelName      string
	MaxConcurrency int
}

// Server wraps the generation engine.
type Server struct {
	engine *generate.Engine
	cfg    Config
	sem    chan struct{}
	mu     sync.Mutex
}

// New creates a server.
func New(eng *generate.Engine, cfg Config) *Server {
	if cfg.MaxConcurrency <= 0 {
		cfg.MaxConcurrency = 4
	}
	if cfg.ModelName == "" {
		cfg.ModelName = "smollm2"
	}
	return &Server{
		engine: eng,
		cfg:    cfg,
		sem:    make(chan struct{}, cfg.MaxConcurrency),
	}
}

// Handler returns the root http.Handler.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models", s.handleModels)
	mux.HandleFunc("POST /v1/chat/completions", s.handleChatCompletions)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})
	return mux
}

// ListenAndServe starts the HTTP server.
func (s *Server) ListenAndServe() error {
	return http.ListenAndServe(s.cfg.Addr, s.Handler())
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	resp := map[string]any{
		"object": "list",
		"data": []map[string]string{
			{"id": s.cfg.ModelName, "object": "model", "owned_by": "go-thru"},
		},
	}
	writeJSON(w, resp)
}

type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens"`
	Temperature float32       `json:"temperature"`
	TopP        float32       `json:"top_p"`
	Stream      bool          `json:"stream"`
	Seed        int64         `json:"seed"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	select {
	case s.sem <- struct{}{}:
		defer func() { <-s.sem }()
	default:
		http.Error(w, "server busy", http.StatusServiceUnavailable)
		return
	}

	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	prompt := buildPrompt(req.Messages)
	genReq := generate.Request{
		Prompt:      prompt,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Seed:        req.Seed,
	}
	if genReq.MaxTokens == 0 {
		genReq.MaxTokens = 256
	}
	if genReq.Temperature == 0 {
		genReq.Temperature = 0.8
	}
	if genReq.TopP == 0 {
		genReq.TopP = 0.9
	}

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Minute)
	defer cancel()

	events, err := s.engine.Generate(ctx, genReq)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if req.Stream {
		s.streamResponse(w, events, req.Model)
		return
	}
	s.jsonResponse(w, events, req.Model)
}

func buildPrompt(msgs []chatMessage) string {
	var b stringsBuilder
	for _, m := range msgs {
		switch m.Role {
		case "system":
			b.WriteString("<|im_start|>system\n")
			b.WriteString(m.Content)
			b.WriteString("\n")
		case "user":
			b.WriteString("<|im_start|>user\n")
			b.WriteString(m.Content)
			b.WriteString("\n")
		case "assistant":
			b.WriteString("<|im_start|>assistant\n")
			b.WriteString(m.Content)
			b.WriteString("\n")
		}
	}
	b.WriteString("<|im_start|>assistant\n")
	return b.String()
}

// stringsBuilder avoids importing strings in hot path file — thin wrapper.
type stringsBuilder struct{ buf []byte }

func (b *stringsBuilder) WriteString(s string) { b.buf = append(b.buf, s...) }
func (b *stringsBuilder) String() string       { return string(b.buf) }

func (s *Server) jsonResponse(w http.ResponseWriter, events <-chan generate.Event, model string) {
	var content stringsBuilder
	for ev := range events {
		if ev.Err != nil {
			http.Error(w, ev.Err.Error(), http.StatusInternalServerError)
			return
		}
		if ev.Done {
			break
		}
		content.WriteString(ev.Token)
	}
	resp := map[string]any{
		"id":      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []map[string]any{
			{
				"index": 0,
				"message": map[string]string{
					"role":    "assistant",
					"content": content.String(),
				},
				"finish_reason": "stop",
			},
		},
	}
	writeJSON(w, resp)
}

func (s *Server) streamResponse(w http.ResponseWriter, events <-chan generate.Event, model string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	id := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	for ev := range events {
		if ev.Err != nil {
			fmt.Fprintf(w, "data: {\"error\":%q}\n\n", ev.Err.Error())
			flusher.Flush()
			return
		}
		if ev.Done {
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			return
		}
		chunk := map[string]any{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": time.Now().Unix(),
			"model":   model,
			"choices": []map[string]any{
				{
					"index": 0,
					"delta": map[string]string{"content": ev.Token},
				},
			},
		}
		data, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}
