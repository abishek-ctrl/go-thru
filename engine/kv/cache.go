// Package kv provides a pre-allocated key-value cache for autoregressive decoding.
package kv

// Cache stores K and V tensors per layer: [nLayers][2][maxSeq][nKVHeads*headDim].
type Cache struct {
	nLayers int
	maxSeq  int
	kvDim   int // nKVHeads * headDim
	k       []float32
	v       []float32
	curLen  int
}

// New allocates a KV cache for the given dimensions.
func New(nLayers, maxSeq, nKVHeads, headDim int) *Cache {
	kvDim := nKVHeads * headDim
	size := nLayers * maxSeq * kvDim
	return &Cache{
		nLayers: nLayers,
		maxSeq:  maxSeq,
		kvDim:   kvDim,
		k:       make([]float32, size),
		v:       make([]float32, size),
	}
}

// Reset clears the cache length (does not zero memory).
func (c *Cache) Reset() {
	c.curLen = 0
}

// Len returns the number of cached positions.
func (c *Cache) Len() int {
	return c.curLen
}

// layerOffset returns base index for layer at position pos.
func (c *Cache) layerOffset(layer, pos int) int {
	return (layer*c.maxSeq + pos) * c.kvDim
}

// Write stores k and v for one token at layer (vectors length kvDim).
func (c *Cache) Write(layer int, pos int, kVec, vVec []float32) {
	off := c.layerOffset(layer, pos)
	copy(c.k[off:off+c.kvDim], kVec)
	copy(c.v[off:off+c.kvDim], vVec)
}

// KView returns K vectors for layer for positions [0, seqLen).
func (c *Cache) KView(layer int) []float32 {
	return c.k[layer*c.maxSeq*c.kvDim : (layer+1)*c.maxSeq*c.kvDim]
}

// VView returns V vectors for layer.
func (c *Cache) VView(layer int) []float32 {
	return c.v[layer*c.maxSeq*c.kvDim : (layer+1)*c.maxSeq*c.kvDim]
}

// Advance marks that one more position has been written.
func (c *Cache) Advance(n int) {
	c.curLen += n
}

// MaxSeq returns capacity.
func (c *Cache) MaxSeq() int {
	return c.maxSeq
}

// KVDim returns nKVHeads * headDim.
func (c *Cache) KVDim() int {
	return c.kvDim
}
