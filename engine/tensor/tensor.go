// Package tensor provides a minimal contiguous float32 tensor type for inference.
package tensor

import (
	"fmt"
)

// Tensor is a row-major contiguous float32 buffer with shape metadata.
type Tensor struct {
	Data  []float32
	Shape []int
}

// New allocates a zeroed tensor with the given shape.
func New(shape ...int) Tensor {
	size := Size(shape)
	return Tensor{
		Data:  make([]float32, size),
		Shape: append([]int(nil), shape...),
	}
}

// FromSlice wraps an existing slice (must match shape product).
func FromSlice(data []float32, shape ...int) (Tensor, error) {
	if Size(shape) != len(data) {
		return Tensor{}, fmt.Errorf("tensor: data len %d != shape size %d", len(data), Size(shape))
	}
	return Tensor{
		Data:  data,
		Shape: append([]int(nil), shape...),
	}, nil
}

// Size returns the product of shape dimensions.
func Size(shape []int) int {
	if len(shape) == 0 {
		return 0
	}
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

// Numel returns the number of elements.
func (t Tensor) Numel() int {
	return len(t.Data)
}

// Clone returns a deep copy.
func (t Tensor) Clone() Tensor {
	out := New(t.Shape...)
	copy(out.Data, t.Data)
	return out
}

// View returns a tensor sharing data with offset (no copy). Panics on invalid offset.
func (t Tensor) View(offset int, shape ...int) Tensor {
	if offset+Size(shape) > len(t.Data) {
		panic(fmt.Sprintf("tensor view out of bounds: offset=%d size=%d len=%d", offset, Size(shape), len(t.Data)))
	}
	return Tensor{
		Data:  t.Data[offset : offset+Size(shape)],
		Shape: append([]int(nil), shape...),
	}
}

// Workspace holds reusable scratch buffers sized for a model config.
type Workspace struct {
	Buf []float32
	// Named regions are views into Buf; populated by model.NewWorkspace.
	Hidden    []float32
	Q         []float32
	K         []float32
	V         []float32
	Attn      []float32
	AttnScore []float32
	FFN       []float32
	Gate      []float32
	Up        []float32
	Logits    []float32
	Residual  []float32
}

// NewWorkspace allocates a single flat buffer and slices named regions.
func NewWorkspace(hiddenDim, nHeads, nKVHeads, headDim, intermediateDim, vocabSize, maxSeq int) *Workspace {
	// Per-token scratch (decode uses seq=1; prefill uses maxSeq for some paths).
	h := hiddenDim
	q := nHeads * headDim
	k := nKVHeads * headDim
	v := nKVHeads * headDim
	attn := nHeads * headDim
	score := nHeads * maxSeq
	ffn := h
	gate := intermediateDim
	up := intermediateDim
	logits := vocabSize

	residual := h
	total := h + q + k + v + attn + score + ffn + gate + up + logits + residual
	buf := make([]float32, total)
	ws := &Workspace{Buf: buf}
	off := 0
	slice := func(n int) []float32 {
		s := buf[off : off+n]
		off += n
		return s
	}
	ws.Hidden = slice(h)
	ws.Q = slice(q)
	ws.K = slice(k)
	ws.V = slice(v)
	ws.Attn = slice(attn)
	ws.AttnScore = slice(score)
	ws.FFN = slice(ffn)
	ws.Gate = slice(gate)
	ws.Up = slice(up)
	ws.Logits = slice(logits)
	ws.Residual = slice(residual)
	return ws
}
