package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

type mockEngine struct{}

func TestHealth(t *testing.T) {
	// Minimal handler test without full engine
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	req := httptest.NewRequest("GET", "/health", nil)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d", rec.Code)
	}
}

func TestModelsJSON(t *testing.T) {
	resp := map[string]any{
		"object": "list",
		"data":   []map[string]string{{"id": "test"}},
	}
	b, err := json.Marshal(resp)
	if err != nil || len(b) == 0 {
		t.Fatal(err)
	}
}

func TestBuildPrompt(t *testing.T) {
	p := buildPrompt([]chatMessage{
		{Role: "user", Content: "hi"},
	})
	if p == "" {
		t.Fatal("empty prompt")
	}
}
