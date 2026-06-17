// Package loader reads model artifacts (safetensors, config.json).
package loader

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Dtype represents safetensors dtype strings.
type Dtype string

const (
	F32  Dtype = "F32"
	F16  Dtype = "F16"
	BF16 Dtype = "BF16"
)

type tensorMeta struct {
	Dtype       string `json:"dtype"`
	Shape       []int  `json:"shape"`
	DataOffsets [2]int `json:"data_offsets"`
}

// SafetensorsFile holds parsed tensor metadata and raw file bytes.
type SafetensorsFile struct {
	path   string
	data   []byte
	meta   map[string]tensorMeta
	offset int // start of tensor data region in file
}

// OpenSafetensors parses a .safetensors file.
func OpenSafetensors(path string) (*SafetensorsFile, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(raw) < 8 {
		return nil, fmt.Errorf("safetensors: file too short")
	}
	headerLen := binary.LittleEndian.Uint64(raw[:8])
	if 8+int(headerLen) > len(raw) {
		return nil, fmt.Errorf("safetensors: header length out of bounds")
	}
	headerJSON := raw[8 : 8+headerLen]
	var header map[string]json.RawMessage
	if err := json.Unmarshal(headerJSON, &header); err != nil {
		return nil, fmt.Errorf("safetensors: header json: %w", err)
	}
	meta := make(map[string]tensorMeta)
	for k, v := range header {
		if k == "__metadata__" {
			continue
		}
		var tm tensorMeta
		if err := json.Unmarshal(v, &tm); err != nil {
			return nil, fmt.Errorf("safetensors: tensor %q: %w", k, err)
		}
		meta[k] = tm
	}
	return &SafetensorsFile{
		path:   path,
		data:   raw,
		meta:   meta,
		offset: 8 + int(headerLen),
	}, nil
}

// Names returns sorted tensor names.
func (f *SafetensorsFile) Names() []string {
	names := make([]string, 0, len(f.meta))
	for n := range f.meta {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// GetTensor returns tensor data as float32 slice (converts F16/BF16 if needed).
func (f *SafetensorsFile) GetTensor(name string) ([]float32, []int, error) {
	tm, ok := f.meta[name]
	if !ok {
		return nil, nil, fmt.Errorf("safetensors: tensor %q not found", name)
	}
	start := f.offset + tm.DataOffsets[0]
	end := f.offset + tm.DataOffsets[1]
	if start < 0 || end > len(f.data) || start >= end {
		return nil, nil, fmt.Errorf("safetensors: bad offsets for %q", name)
	}
	raw := f.data[start:end]
	switch Dtype(tm.Dtype) {
	case F32:
		n := len(raw) / 4
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			bits := binary.LittleEndian.Uint32(raw[i*4:])
			out[i] = math.Float32frombits(bits)
		}
		return out, tm.Shape, nil
	case F16:
		n := len(raw) / 2
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = float16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
		return out, tm.Shape, nil
	case BF16:
		n := len(raw) / 2
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			u16 := binary.LittleEndian.Uint16(raw[i*2:])
			out[i] = math.Float32frombits(uint32(u16) << 16)
		}
		return out, tm.Shape, nil
	default:
		return nil, nil, fmt.Errorf("safetensors: unsupported dtype %q", tm.Dtype)
	}
}

func float16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := (h >> 10) & 0x1f
	frac := uint32(h & 0x3ff)
	if exp == 0 {
		if frac == 0 {
			return math.Float32frombits(sign)
		}
		e := exp
		for e == 0 && (frac&0x400) == 0 {
			frac <<= 1
			e--
		}
		e++
		frac &= 0x3ff
		return math.Float32frombits(sign | ((uint32(e) + 112) << 23) | (frac << 13))
	}
	if exp == 31 {
		if frac == 0 {
			return math.Float32frombits(sign | 0x7f800000)
		}
		return math.Float32frombits(sign | 0x7fc00000)
	}
	return math.Float32frombits(sign | ((uint32(exp) + 112) << 23) | (frac << 13))
}

// LlamaConfig mirrors HuggingFace config.json for Llama-family models.
type LlamaConfig struct {
	HiddenSize            int     `json:"hidden_size"`
	IntermediateSize      int     `json:"intermediate_size"`
	NumHiddenLayers       int     `json:"num_hidden_layers"`
	NumAttentionHeads     int     `json:"num_attention_heads"`
	NumKeyValueHeads      int     `json:"num_key_value_heads"`
	VocabSize             int     `json:"vocab_size"`
	MaxPositionEmbeddings int     `json:"max_position_embeddings"`
	RMSNormEps            float64 `json:"rms_norm_eps"`
	RopeTheta             float64 `json:"rope_theta"`
	HiddenAct             string  `json:"hidden_act"`
	TieWordEmbeddings     bool    `json:"tie_word_embeddings"`
	BOSID                 int     `json:"bos_token_id"`
	EOSID                 int     `json:"eos_token_id"`
}

// LoadConfig reads config.json from a model directory.
func LoadConfig(modelDir string) (*LlamaConfig, error) {
	path := filepath.Join(modelDir, "config.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg LlamaConfig
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, err
	}
	if cfg.NumKeyValueHeads == 0 {
		cfg.NumKeyValueHeads = cfg.NumAttentionHeads
	}
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 10000
	}
	return &cfg, nil
}

// FindSafetensors returns the path to model weights in dir.
func FindSafetensors(modelDir string) (string, error) {
	// Single file
	p := filepath.Join(modelDir, "model.safetensors")
	if _, err := os.Stat(p); err == nil {
		return p, nil
	}
	// Sharded: model-00001-of-00002.safetensors — v1: first shard only for tiny models
	entries, err := os.ReadDir(modelDir)
	if err != nil {
		return "", err
	}
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".safetensors") {
			return filepath.Join(modelDir, e.Name()), nil
		}
	}
	return "", fmt.Errorf("no .safetensors found in %s", modelDir)
}

// WriteSafetensorsFixture writes a minimal safetensors file for tests.
func WriteSafetensorsFixture(w io.Writer, tensors map[string]struct {
	Dtype string
	Shape []int
	Data  []float32
}) error {
	type info struct {
		Dtype       string `json:"dtype"`
		Shape       []int  `json:"shape"`
		DataOffsets [2]int `json:"data_offsets"`
	}
	header := make(map[string]any)
	var data []byte
	offset := 0
	names := make([]string, 0, len(tensors))
	for n := range tensors {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, name := range names {
		t := tensors[name]
		nBytes := len(t.Data) * 4
		header[name] = info{Dtype: string(F32), Shape: t.Shape, DataOffsets: [2]int{offset, offset + nBytes}}
		for _, v := range t.Data {
			var buf [4]byte
			binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
			data = append(data, buf[:]...)
		}
		offset += nBytes
	}
	hj, _ := json.Marshal(header)
	pad := (8 - len(hj)%8) % 8
	for i := 0; i < pad; i++ {
		hj = append(hj, ' ')
	}
	var hdr [8]byte
	binary.LittleEndian.PutUint64(hdr[:], uint64(len(hj)))
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	if _, err := w.Write(hj); err != nil {
		return err
	}
	_, err := w.Write(data)
	return err
}
