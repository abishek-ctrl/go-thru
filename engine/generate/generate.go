// Package generate provides the high-level text generation loop.
package generate

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/abishekm/go-thru/engine/kv"
	"github.com/abishekm/go-thru/engine/model"
	"github.com/abishekm/go-thru/engine/sampler"
	"github.com/abishekm/go-thru/engine/tokenizer"
)

// Request is a generation request.
type Request struct {
	Prompt      string
	MaxTokens   int
	Temperature float32
	TopP        float32
	TopK        int
	Seed        int64
	Stop        []string
}

// Event is streamed during generation.
type Event struct {
	Token   string
	TokenID int32
	Done    bool
	Err     error
	Stats   *Stats
}

// Stats holds generation metrics.
type Stats struct {
	PromptTokens int
	OutputTokens int
	Duration     time.Duration
	TokensPerSec float64
}

// Engine wraps model + tokenizer for generation.
type Engine struct {
	Model     *model.Model
	Tokenizer *tokenizer.Tokenizer
	Cache     *kv.Cache
	logits    []float32
}

// NewEngine loads model and tokenizer from directory.
func NewEngine(modelDir string) (*Engine, error) {
	m, err := model.Load(modelDir)
	if err != nil {
		return nil, fmt.Errorf("model: %w", err)
	}
	tok, err := tokenizer.Load(modelDir)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: %w", err)
	}
	cfg := m.Cfg
	cache := kv.New(cfg.NumLayers, cfg.MaxSeqLen, cfg.NumKVHeads, cfg.HeadDim)
	logits := make([]float32, cfg.VocabSize)
	return &Engine{
		Model:     m,
		Tokenizer: tok,
		Cache:     cache,
		logits:    logits,
	}, nil
}

// Tokenize encodes text.
func (e *Engine) Tokenize(text string) []int32 {
	return e.Tokenizer.Encode(text)
}

// Detokenize decodes ids.
func (e *Engine) Detokenize(ids []int32) string {
	return e.Tokenizer.Decode(ids)
}

// Generate runs autoregressive generation, streaming events.
func (e *Engine) Generate(ctx context.Context, req Request) (<-chan Event, error) {
	if req.MaxTokens <= 0 {
		req.MaxTokens = 256
	}
	ch := make(chan Event, 16)
	go func() {
		defer close(ch)
		start := time.Now()
		promptIDs := e.Tokenizer.Encode(req.Prompt)
		if len(promptIDs) >= e.Model.Cfg.MaxSeqLen {
			ch <- Event{Err: fmt.Errorf("prompt too long: %d tokens", len(promptIDs))}
			return
		}

		e.Cache.Reset()
		if err := e.Model.Prefill(promptIDs, e.Cache, e.logits); err != nil {
			ch <- Event{Err: err}
			return
		}

		pos := len(promptIDs)
		samp := sampler.New(sampler.Config{
			Temperature: req.Temperature,
			TopP:        req.TopP,
			TopK:        req.TopK,
			Seed:        req.Seed,
		})

		var output strings.Builder
		outputTokens := 0
		eosID := e.Model.Cfg.EOSID
		if eosID == 0 {
			eosID = int32(e.Tokenizer.EOS())
		}

		for i := 0; i < req.MaxTokens; i++ {
			select {
			case <-ctx.Done():
				ch <- Event{Err: ctx.Err()}
				return
			default:
			}

			if pos >= e.Model.Cfg.MaxSeqLen {
				break
			}

			tokID := int32(samp.Sample(e.logits))
			if tokID == eosID {
				break
			}

			piece := e.Tokenizer.Decode([]int32{tokID})
			output.WriteString(piece)
			outputTokens++

			ch <- Event{Token: piece, TokenID: tokID}

			if matchesStop(output.String(), req.Stop) {
				break
			}

			if err := e.Model.Forward(tokID, pos, e.Cache, e.logits); err != nil {
				ch <- Event{Err: err}
				return
			}
			pos++
		}

		elapsed := time.Since(start)
		tps := float64(outputTokens) / elapsed.Seconds()
		ch <- Event{
			Done: true,
			Stats: &Stats{
				PromptTokens: len(promptIDs),
				OutputTokens: outputTokens,
				Duration:     elapsed,
				TokensPerSec: tps,
			},
		}
	}()
	return ch, nil
}

func matchesStop(text string, stops []string) bool {
	for _, s := range stops {
		if s != "" && strings.HasSuffix(text, s) {
			return true
		}
	}
	return false
}

// ApplyChatTemplate wraps user message in ChatML format (SmolLM2 default).
func ApplyChatTemplate(user string) string {
	return "<|im_start|>user\n" + user + "\n<|im_start|>assistant\n"
}
