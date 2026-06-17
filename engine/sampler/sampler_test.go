package sampler

import "testing"

func TestGreedy(t *testing.T) {
	s := New(Config{Temperature: 0})
	logits := []float32{0.1, 0.9, 0.2}
	if got := s.Sample(logits); got != 1 {
		t.Fatalf("got %d want 1", got)
	}
}

func TestStochastic(t *testing.T) {
	s := New(Config{Temperature: 1, TopP: 1, Seed: 42})
	logits := []float32{1, 2, 3}
	tok := s.Sample(logits)
	if tok < 0 || tok > 2 {
		t.Fatalf("bad token %d", tok)
	}
}
