package model

import (
	"math"
	"testing"

	"github.com/abishekm/go-thru/engine/kv"
	"github.com/abishekm/go-thru/engine/loader"
	"github.com/abishekm/go-thru/engine/ops"
	"github.com/abishekm/go-thru/engine/tensor"
)

func TestForwardTiny(t *testing.T) {
	hidden, heads, kvHeads, headDim := 8, 2, 1, 4
	intermediate, vocab, layers := 16, 10, 1
	cfg := Config{
		HiddenSize: hidden, IntermediateSize: intermediate, NumLayers: layers,
		NumHeads: heads, NumKVHeads: kvHeads, HeadDim: headDim,
		VocabSize: vocab, MaxSeqLen: 32, RMSNormEps: 1e-5, RopeTheta: 10000,
	}
	ones := func(n int) []float32 {
		o := make([]float32, n)
		for i := range o {
			o[i] = 0.01 * float32(i%5+1)
		}
		return o
	}
	lw := LayerWeights{
		InputNorm: ones(hidden), AttnNorm: ones(hidden),
		QProj: ones(hidden * heads * headDim), KProj: ones(hidden * kvHeads * headDim),
		VProj: ones(hidden * kvHeads * headDim), OProj: ones(heads * headDim * hidden),
		GateProj: ones(hidden * intermediate), UpProj: ones(hidden * intermediate),
		DownProj: ones(intermediate * hidden),
	}
	m := &Model{
		Cfg: cfg,
		W: Weights{
			Embed:  ones(vocab * hidden),
			LMHead: ones(vocab * hidden),
			Norm:   ones(hidden),
			Layers: []LayerWeights{lw},
		},
	}
	m.CosTable, m.SinTable = ops.BuildRoPETables(cfg.MaxSeqLen, headDim, cfg.RopeTheta)
	m.Workspace = tensor.NewWorkspace(hidden, heads, kvHeads, headDim, intermediate, vocab, cfg.MaxSeqLen)

	cache := kv.New(layers, cfg.MaxSeqLen, kvHeads, headDim)
	logits := make([]float32, vocab)
	if err := m.Forward(1, 0, cache, logits); err != nil {
		t.Fatal(err)
	}
	var sum float32
	for _, v := range logits {
		sum += float32(math.Abs(float64(v)))
	}
	if sum == 0 {
		t.Fatal("all zero logits")
	}
}

func TestConfigFromLlama(t *testing.T) {
	cfg := ConfigFromLlama(&loader.LlamaConfig{
		HiddenSize: 64, NumAttentionHeads: 4, NumKeyValueHeads: 2,
	})
	if cfg.HeadDim != 16 {
		t.Fatalf("headDim %d", cfg.HeadDim)
	}
}
