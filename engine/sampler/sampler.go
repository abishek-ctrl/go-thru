// Package sampler provides token sampling strategies.
package sampler

import (
	"math"
	"math/rand"
	"sort"
)

// Config controls sampling behavior.
type Config struct {
	Temperature float32
	TopP        float32
	TopK        int
	Seed        int64
}

// DefaultConfig returns sensible defaults.
func DefaultConfig() Config {
	return Config{Temperature: 0.8, TopP: 0.9, TopK: 40, Seed: 0}
}

// Sampler holds RNG state.
type Sampler struct {
	cfg Config
	rng *rand.Rand
}

// New creates a sampler.
func New(cfg Config) *Sampler {
	src := rand.NewSource(cfg.Seed)
	return &Sampler{cfg: cfg, rng: rand.New(src)}
}

type scored struct {
	id    int
	score float32
}

// Sample selects a token index from logits.
func (s *Sampler) Sample(logits []float32) int {
	if s.cfg.Temperature <= 0 {
		return argmax(logits)
	}

	scores := make([]scored, len(logits))
	maxV := logits[0]
	for i, v := range logits {
		if v > maxV {
			maxV = v
		}
		scores[i] = scored{id: i, score: v}
	}

	invT := 1.0 / s.cfg.Temperature
	var sum float32
	for i := range scores {
		e := float32(math.Exp(float64((scores[i].score - maxV) * invT)))
		scores[i].score = e
		sum += e
	}

	// Top-K filter
	if s.cfg.TopK > 0 && s.cfg.TopK < len(scores) {
		sort.Slice(scores, func(i, j int) bool { return scores[i].score > scores[j].score })
		scores = scores[:s.cfg.TopK]
		sum = 0
		for i := range scores {
			sum += scores[i].score
		}
	}

	// Top-P (nucleus)
	if s.cfg.TopP > 0 && s.cfg.TopP < 1 {
		sort.Slice(scores, func(i, j int) bool { return scores[i].score > scores[j].score })
		var cum float32
		cut := len(scores)
		for i, sc := range scores {
			cum += sc.score / sum
			if cum >= s.cfg.TopP {
				cut = i + 1
				break
			}
		}
		scores = scores[:cut]
		sum = 0
		for i := range scores {
			sum += scores[i].score
		}
	}

	r := s.rng.Float32() * sum
	var cum float32
	for _, sc := range scores {
		cum += sc.score
		if r <= cum {
			return sc.id
		}
	}
	return scores[len(scores)-1].id
}

func argmax(x []float32) int {
	best := 0
	for i := 1; i < len(x); i++ {
		if x[i] > x[best] {
			best = i
		}
	}
	return best
}
