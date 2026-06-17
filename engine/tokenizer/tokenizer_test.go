package tokenizer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestParseMinimalBPE(t *testing.T) {
	// Minimal tokenizer.json fixture
	fixture := `{
		"model": {
			"type": "BPE",
			"vocab": {"a": 0, "b": 1, "ab": 2, "c": 3},
			"merges": [["a", "b"]]
		},
		"added_tokens": []
	}`
	tok, err := Parse([]byte(fixture))
	if err != nil {
		t.Fatal(err)
	}
	if tok.VocabSize() < 4 {
		t.Fatalf("vocab size %d", tok.VocabSize())
	}
}

func TestGoldenFixture(t *testing.T) {
	path := filepath.Join("..", "..", "testdata", "tokenizer_minimal.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skip("testdata not present")
	}
	tok, err := Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	goldenPath := filepath.Join("..", "..", "testdata", "golden_encode.json")
	goldenRaw, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Skip("golden fixture not present")
	}
	var cases []struct {
		Text string  `json:"text"`
		IDs  []int32 `json:"ids"`
	}
	if err := json.Unmarshal(goldenRaw, &cases); err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		got := tok.Encode(c.Text)
		if len(got) != len(c.IDs) {
			t.Fatalf("text %q: len %d want %d", c.Text, len(got), len(c.IDs))
		}
		for i := range c.IDs {
			if got[i] != c.IDs[i] {
				t.Fatalf("text %q: id[%d]=%d want %d", c.Text, i, got[i], c.IDs[i])
			}
		}
	}
}
