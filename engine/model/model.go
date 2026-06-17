// Package model implements Llama-family decoder forward pass.
package model

import (
	"fmt"
	"math"

	"github.com/abishekm/go-thru/engine/kv"
	"github.com/abishekm/go-thru/engine/loader"
	"github.com/abishekm/go-thru/engine/ops"
	"github.com/abishekm/go-thru/engine/tensor"
)

// Config holds model hyperparameters.
type Config struct {
	HiddenSize       int
	IntermediateSize int
	NumLayers        int
	NumHeads         int
	NumKVHeads       int
	HeadDim          int
	VocabSize        int
	MaxSeqLen        int
	RMSNormEps       float32
	RopeTheta        float32
	TieEmbeddings    bool
	BOSID            int32
	EOSID            int32
}

// ConfigFromLlama converts loader config.
func ConfigFromLlama(c *loader.LlamaConfig) Config {
	headDim := c.HiddenSize / c.NumAttentionHeads
	return Config{
		HiddenSize:       c.HiddenSize,
		IntermediateSize: c.IntermediateSize,
		NumLayers:        c.NumHiddenLayers,
		NumHeads:         c.NumAttentionHeads,
		NumKVHeads:       c.NumKeyValueHeads,
		HeadDim:          headDim,
		VocabSize:        c.VocabSize,
		MaxSeqLen:        c.MaxPositionEmbeddings,
		RMSNormEps:       float32(c.RMSNormEps),
		RopeTheta:        float32(c.RopeTheta),
		TieEmbeddings:    c.TieWordEmbeddings,
		BOSID:            int32(c.BOSID),
		EOSID:            int32(c.EOSID),
	}
}

// LayerWeights holds one transformer block.
type LayerWeights struct {
	InputNorm []float32
	AttnNorm  []float32
	QProj     []float32
	KProj     []float32
	VProj     []float32
	OProj     []float32
	GateProj  []float32
	UpProj    []float32
	DownProj  []float32
}

// Weights holds all model parameters.
type Weights struct {
	Embed  []float32
	LMHead []float32
	Norm   []float32
	Layers []LayerWeights
}

// Model runs inference.
type Model struct {
	Cfg       Config
	W         Weights
	CosTable  []float32
	SinTable  []float32
	Workspace *tensor.Workspace
}

// Load reads weights from a HuggingFace model directory.
func Load(modelDir string) (*Model, error) {
	cfg, err := loader.LoadConfig(modelDir)
	if err != nil {
		return nil, fmt.Errorf("config: %w", err)
	}
	stPath, err := loader.FindSafetensors(modelDir)
	if err != nil {
		return nil, err
	}
	sf, err := loader.OpenSafetensors(stPath)
	if err != nil {
		return nil, err
	}
	mcfg := ConfigFromLlama(cfg)
	w, err := loadWeights(sf, mcfg)
	if err != nil {
		return nil, err
	}
	cos, sin := ops.BuildRoPETables(mcfg.MaxSeqLen, mcfg.HeadDim, mcfg.RopeTheta)
	ws := tensor.NewWorkspace(
		mcfg.HiddenSize, mcfg.NumHeads, mcfg.NumKVHeads, mcfg.HeadDim,
		mcfg.IntermediateSize, mcfg.VocabSize, mcfg.MaxSeqLen,
	)
	return &Model{
		Cfg:       mcfg,
		W:         w,
		CosTable:  cos,
		SinTable:  sin,
		Workspace: ws,
	}, nil
}

func loadWeights(sf *loader.SafetensorsFile, cfg Config) (Weights, error) {
	get := func(name string) ([]float32, error) {
		data, _, err := sf.GetTensor(name)
		return data, err
	}
	embed, err := get("model.embed_tokens.weight")
	if err != nil {
		return Weights{}, err
	}
	norm, err := get("model.norm.weight")
	if err != nil {
		return Weights{}, err
	}
	var lmHead []float32
	if cfg.TieEmbeddings {
		lmHead = embed
	} else {
		lmHead, err = get("lm_head.weight")
		if err != nil {
			return Weights{}, err
		}
	}
	layers := make([]LayerWeights, cfg.NumLayers)
	for i := 0; i < cfg.NumLayers; i++ {
		p := fmt.Sprintf("model.layers.%d.", i)
		lw := LayerWeights{}
		var e error
		lw.InputNorm, e = get(p + "input_layernorm.weight")
		if e != nil {
			return Weights{}, e
		}
		lw.AttnNorm, e = get(p + "post_attention_layernorm.weight")
		if e != nil {
			return Weights{}, e
		}
		lw.QProj, e = get(p + "self_attn.q_proj.weight")
		if e != nil {
			return Weights{}, e
		}
		lw.KProj, e = get(p + "self_attn.k_proj.weight")
		if e != nil {
			return Weights{}, e
		}
		lw.VProj, e = get(p + "self_attn.v_proj.weight")
		if e != nil {
			return Weights{}, e
		}
		lw.OProj, e = get(p + "self_attn.o_proj.weight")
		if e != nil {
			return Weights{}, e
		}
		lw.GateProj, e = get(p + "mlp.gate_proj.weight")
		if e != nil {
			return Weights{}, e
		}
		lw.UpProj, e = get(p + "mlp.up_proj.weight")
		if e != nil {
			return Weights{}, e
		}
		lw.DownProj, e = get(p + "mlp.down_proj.weight")
		if e != nil {
			return Weights{}, e
		}
		layers[i] = lw
	}
	return Weights{Embed: embed, LMHead: lmHead, Norm: norm, Layers: layers}, nil
}

// Forward runs one token at position pos, updating cache. Returns logits [vocabSize].
func (m *Model) Forward(token int32, pos int, cache *kv.Cache, logits []float32) error {
	if pos >= m.Cfg.MaxSeqLen {
		return fmt.Errorf("position %d exceeds max %d", pos, m.Cfg.MaxSeqLen)
	}
	cfg := m.Cfg
	h := cfg.HiddenSize
	ws := m.Workspace
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	qDim := cfg.NumHeads * cfg.HeadDim
	repeat := cfg.NumHeads / cfg.NumKVHeads
	scale := float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))

	ops.Copy(ws.Hidden, m.W.Embed[int(token)*h:(int(token)+1)*h])

	for layer := 0; layer < cfg.NumLayers; layer++ {
		lw := m.W.Layers[layer]
		ops.Copy(ws.Residual, ws.Hidden)

		ops.RMSNorm(ws.Hidden, lw.InputNorm, cfg.RMSNormEps, ws.FFN)

		ops.MatVec(qDim, h, lw.QProj, ws.FFN, ws.Q)
		ops.MatVec(kvDim, h, lw.KProj, ws.FFN, ws.K)
		ops.MatVec(kvDim, h, lw.VProj, ws.FFN, ws.V)

		for head := 0; head < cfg.NumHeads; head++ {
			qHead := ws.Q[head*cfg.HeadDim : (head+1)*cfg.HeadDim]
			kvHead := head / repeat
			kHead := ws.K[kvHead*cfg.HeadDim : (kvHead+1)*cfg.HeadDim]
			ops.RoPE(qHead, kHead, cfg.HeadDim, pos, m.CosTable, m.SinTable)
		}
		cache.Write(layer, pos, ws.K, ws.V)

		seqLen := pos + 1
		attnOut := ws.Attn[:qDim]
		kBase := cache.KView(layer)
		vBase := cache.VView(layer)

		for head := 0; head < cfg.NumHeads; head++ {
			qHead := ws.Q[head*cfg.HeadDim : (head+1)*cfg.HeadDim]
			kvHead := head / repeat
			scores := ws.AttnScore[head*cfg.MaxSeqLen : head*cfg.MaxSeqLen+seqLen]

			for t := 0; t < seqLen; t++ {
				kOff := t*kvDim + kvHead*cfg.HeadDim
				kHead := kBase[kOff : kOff+cfg.HeadDim]
				var dot float32
				for d := 0; d < cfg.HeadDim; d++ {
					dot += qHead[d] * kHead[d]
				}
				scores[t] = dot * scale
			}
			ops.SoftmaxInPlace(scores)

			outHead := attnOut[head*cfg.HeadDim : (head+1)*cfg.HeadDim]
			for d := range outHead {
				outHead[d] = 0
			}
			for t := 0; t < seqLen; t++ {
				vOff := t*kvDim + kvHead*cfg.HeadDim
				vHead := vBase[vOff : vOff+cfg.HeadDim]
				s := scores[t]
				for d := 0; d < cfg.HeadDim; d++ {
					outHead[d] += s * vHead[d]
				}
			}
		}

		ops.MatVec(h, qDim, lw.OProj, attnOut, ws.FFN)
		ops.Copy(ws.Hidden, ws.Residual)
		ops.AddInPlace(ws.Hidden, ws.FFN)

		ops.Copy(ws.Residual, ws.Hidden)
		ops.RMSNorm(ws.Hidden, lw.AttnNorm, cfg.RMSNormEps, ws.FFN)
		ops.MatVec(cfg.IntermediateSize, h, lw.GateProj, ws.FFN, ws.Gate)
		ops.MatVec(cfg.IntermediateSize, h, lw.UpProj, ws.FFN, ws.Up)
		ops.SiLUInPlace(ws.Gate)
		ops.MulElemInPlace(ws.Gate, ws.Up)
		ops.MatVec(h, cfg.IntermediateSize, lw.DownProj, ws.Gate, ws.FFN)
		ops.Copy(ws.Hidden, ws.Residual)
		ops.AddInPlace(ws.Hidden, ws.FFN)
	}

	ops.RMSNorm(ws.Hidden, m.W.Norm, cfg.RMSNormEps, ws.FFN)
	ops.MatVec(cfg.VocabSize, h, m.W.LMHead, ws.FFN, logits)
	return nil
}

// Prefill processes prompt tokens; logits hold output for the last token.
func (m *Model) Prefill(tokens []int32, cache *kv.Cache, logits []float32) error {
	cache.Reset()
	for i, tok := range tokens {
		if err := m.Forward(tok, i, cache, logits); err != nil {
			return err
		}
	}
	cache.Advance(len(tokens))
	return nil
}
