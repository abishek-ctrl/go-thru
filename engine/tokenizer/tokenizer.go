// Package tokenizer implements HuggingFace BPE tokenization from tokenizer.json.
package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"golang.org/x/text/unicode/norm"
)

// Tokenizer encodes and decodes text using byte-level BPE.
type Tokenizer struct {
	vocab       map[string]int
	idToToken   []string
	merges      map[string]int // "a b" -> priority (lower = earlier)
	specialIDs  map[int]bool
	bosID       int
	eosID       int
	unkID       int
	byteLevel   bool
	byteEncoder [256]int
	byteDecoder map[int]byte
}

type tokenizerJSON struct {
	Model struct {
		Type   string         `json:"type"`
		Vocab  map[string]int `json:"vocab"`
		Merges [][]string     `json:"merges"`
	} `json:"model"`
	AddedTokens []struct {
		ID      int    `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens"`
	PreTokenizer *struct {
		Type string `json:"type"`
	} `json:"pre_tokenizer"`
}

// Load reads tokenizer.json from a model directory.
func Load(modelDir string) (*Tokenizer, error) {
	path := filepath.Join(modelDir, "tokenizer.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Parse(raw)
}

// Parse builds a tokenizer from JSON bytes.
func Parse(raw []byte) (*Tokenizer, error) {
	var tj tokenizerJSON
	if err := json.Unmarshal(raw, &tj); err != nil {
		return nil, err
	}
	if tj.Model.Type != "BPE" {
		return nil, fmt.Errorf("tokenizer: unsupported model type %q", tj.Model.Type)
	}

	t := &Tokenizer{
		vocab:       tj.Model.Vocab,
		merges:      make(map[string]int),
		specialIDs:  make(map[int]bool),
		byteDecoder: make(map[int]byte),
	}

	// Build idToToken sorted by id
	maxID := 0
	for _, id := range tj.Model.Vocab {
		if id > maxID {
			maxID = id
		}
	}
	t.idToToken = make([]string, maxID+1)
	for tok, id := range tj.Model.Vocab {
		t.idToToken[id] = tok
	}

	for i, pair := range tj.Model.Merges {
		if len(pair) != 2 {
			continue
		}
		t.merges[pair[0]+" "+pair[1]] = i
	}

	for _, at := range tj.AddedTokens {
		if at.Special {
			t.specialIDs[at.ID] = true
		}
	}

	t.buildByteEncoder()
	t.byteLevel = len(tj.Model.Vocab) >= 256

	// Defaults for Llama/SmolLM
	if id, ok := t.vocab["<|endoftext|>"]; ok {
		t.eosID = id
	}
	if id, ok := t.vocab["<|im_start|>"]; ok {
		t.bosID = id
	}
	if t.bosID == 0 {
		if id, ok := t.vocab["<s>"]; ok {
			t.bosID = id
		}
	}
	if t.eosID == 0 {
		if id, ok := t.vocab["</s>"]; ok {
			t.eosID = id
		}
	}
	if id, ok := t.vocab["<unk>"]; ok {
		t.unkID = id
	}

	return t, nil
}

func (t *Tokenizer) buildByteEncoder() {
	// GPT-2 byte-level BPE mapping
	var bs []int
	for i := 0; i < 256; i++ {
		bs = append(bs, i)
	}
	isPrintable := func(b int) bool {
		return (b >= 33 && b <= 126) || b == 9 || b == 10 || b == 13
	}
	sort.Slice(bs, func(i, j int) bool {
		bi, bj := bs[i], bs[j]
		if isPrintable(bi) && !isPrintable(bj) {
			return true
		}
		if !isPrintable(bi) && isPrintable(bj) {
			return false
		}
		return bi < bj
	})
	for i, b := range bs {
		t.byteEncoder[b] = i
		t.byteDecoder[i] = byte(b)
	}
}

func (t *Tokenizer) bytesToTokens(b []byte) []string {
	out := make([]string, len(b))
	for i, by := range b {
		out[i] = t.idToToken[t.byteEncoder[int(by)]]
	}
	return out
}

func (t *Tokenizer) bpe(token string) []string {
	if _, ok := t.vocab[token]; ok {
		return []string{token}
	}
	parts := strings.Split(token, "")
	for len(parts) > 1 {
		bestIdx := -1
		bestRank := int(1e9)
		for i := 0; i < len(parts)-1; i++ {
			key := parts[i] + " " + parts[i+1]
			if rank, ok := t.merges[key]; ok && rank < bestRank {
				bestRank = rank
				bestIdx = i
			}
		}
		if bestIdx < 0 {
			break
		}
		merged := parts[bestIdx] + parts[bestIdx+1]
		parts = append(parts[:bestIdx], append([]string{merged}, parts[bestIdx+2:]...)...)
	}
	return parts
}

// Encode tokenizes text to IDs.
func (t *Tokenizer) Encode(text string) []int32 {
	text = norm.NFC.String(text)
	if !t.byteLevel {
		return t.encodeSimple(text)
	}
	bytes := []byte(text)
	if len(bytes) == 0 {
		return nil
	}
	var pieces []string
	for _, b := range bytes {
		pieces = append(pieces, t.idToToken[t.byteEncoder[int(b)]])
	}
	merged := strings.Join(pieces, "")
	bpeTokens := t.bpe(merged)
	return t.tokensToIDs(bpeTokens)
}

func (t *Tokenizer) encodeSimple(text string) []int32 {
	parts := t.bpe(text)
	return t.tokensToIDs(parts)
}

func (t *Tokenizer) tokensToIDs(tokens []string) []int32 {
	ids := make([]int32, 0, len(tokens))
	for _, p := range tokens {
		if id, ok := t.vocab[p]; ok {
			ids = append(ids, int32(id))
		} else if t.unkID > 0 {
			ids = append(ids, int32(t.unkID))
		}
	}
	return ids
}

// Decode converts token IDs back to text.
func (t *Tokenizer) Decode(ids []int32) string {
	if !t.byteLevel {
		var b strings.Builder
		for _, id := range ids {
			if int(id) < len(t.idToToken) {
				b.WriteString(t.idToToken[id])
			}
		}
		return b.String()
	}
	var tokenStrs []string
	for _, id := range ids {
		if int(id) < len(t.idToToken) {
			tokenStrs = append(tokenStrs, t.idToToken[id])
		}
	}
	merged := strings.Join(tokenStrs, "")
	// Reverse byte mapping: each token string maps back to a byte
	var out []byte
	for i := 0; i < len(merged); {
		found := false
		for byteVal, encID := range t.byteEncoder {
			tok := t.idToToken[encID]
			if strings.HasPrefix(merged[i:], tok) {
				out = append(out, byte(byteVal))
				i += len(tok)
				found = true
				break
			}
		}
		if !found {
			i++
		}
	}
	return string(out)
}

// BOS returns beginning-of-sequence token id.
func (t *Tokenizer) BOS() int32 { return int32(t.bosID) }

// EOS returns end-of-sequence token id.
func (t *Tokenizer) EOS() int32 { return int32(t.eosID) }

// VocabSize returns vocabulary size.
func (t *Tokenizer) VocabSize() int { return len(t.idToToken) }
