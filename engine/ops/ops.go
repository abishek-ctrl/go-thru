// Package ops provides pure-Go inference math primitives.
package ops

import (
	"math"
	"runtime"
	"sync"
)

const tileSize = 64

// MatMulNaive computes C = A @ B where A is [M,K], B is [K,N], C is [M,N] row-major.
func MatMulNaive(m, k, n int, a, b, c []float32) {
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			aRow := a[i*k:]
			for l := 0; l < k; l++ {
				sum += aRow[l] * b[l*n+j]
			}
			c[i*n+j] = sum
		}
	}
}

// MatMul computes C = A @ B using tiled blocked multiplication with goroutine parallelism.
// A: [M,K], B: [K,N], C: [M,N] row-major.
func MatMul(m, k, n int, a, b, c []float32) {
	if m <= tileSize && n <= tileSize {
		MatMulNaive(m, k, n, a, b, c)
		return
	}
	// Zero C
	for i := range c {
		c[i] = 0
	}

	nRowTiles := (m + tileSize - 1) / tileSize
	workers := runtime.GOMAXPROCS(0)
	if workers > nRowTiles {
		workers = nRowTiles
	}
	if workers < 1 {
		workers = 1
	}

	var wg sync.WaitGroup
	tilesPerWorker := (nRowTiles + workers - 1) / workers

	for w := 0; w < workers; w++ {
		startTile := w * tilesPerWorker
		endTile := startTile + tilesPerWorker
		if endTile > nRowTiles {
			endTile = nRowTiles
		}
		if startTile >= endTile {
			continue
		}
		wg.Add(1)
		go func(st, et int) {
			defer wg.Done()
			for iTile := st; iTile < et; iTile++ {
				i0 := iTile * tileSize
				i1 := i0 + tileSize
				if i1 > m {
					i1 = m
				}
				for jTile := 0; jTile < (n+tileSize-1)/tileSize; jTile++ {
					j0 := jTile * tileSize
					j1 := j0 + tileSize
					if j1 > n {
						j1 = n
					}
					for lTile := 0; lTile < (k+tileSize-1)/tileSize; lTile++ {
						l0 := lTile * tileSize
						l1 := l0 + tileSize
						if l1 > k {
							l1 = k
						}
						for i := i0; i < i1; i++ {
							aRow := a[i*k:]
							cRow := c[i*n:]
							for l := l0; l < l1; l++ {
								aval := aRow[l]
								bRow := b[l*n:]
								for j := j0; j < j1; j++ {
									cRow[j] += aval * bRow[j]
								}
							}
						}
					}
				}
			}
		}(startTile, endTile)
	}
	wg.Wait()
}

// MatVec computes y = A @ x where A is [M,K], x is [K], y is [M].
func MatVec(m, k int, a, x, y []float32) {
	for i := 0; i < m; i++ {
		var sum float32
		aRow := a[i*k:]
		for j := 0; j < k; j++ {
			sum += aRow[j] * x[j]
		}
		y[i] = sum
	}
}

// MatVecTransposed computes y = W @ x where W is stored as [outDim, inDim] row-major.
func MatVecTransposed(outDim, inDim int, w, x, y []float32) {
	MatVec(outDim, inDim, w, x, y)
}

// AddInPlace: dst += src
func AddInPlace(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}

// Copy copies src to dst.
func Copy(dst, src []float32) {
	copy(dst, src)
}

// MulElemInPlace: dst[i] *= src[i]
func MulElemInPlace(dst, src []float32) {
	for i := range dst {
		dst[i] *= src[i]
	}
}

// ScaleInPlace: dst[i] *= s
func ScaleInPlace(dst []float32, s float32) {
	for i := range dst {
		dst[i] *= s
	}
}

// SiLUInPlace applies x * sigmoid(x) in place.
func SiLUInPlace(x []float32) {
	for i := range x {
		v := x[i]
		x[i] = v / (1 + float32(math.Exp(float64(-v))))
	}
}

// RMSNorm computes out = (x / rsqrt(mean(x^2)+eps)) * weight.
func RMSNorm(x, weight []float32, eps float32, out []float32) {
	var sum float32
	for _, v := range x {
		sum += v * v
	}
	inv := float32(1.0 / math.Sqrt(float64(sum/float32(len(x))+eps)))
	for i := range x {
		out[i] = x[i] * inv * weight[i]
	}
}

// SoftmaxInPlace applies stable softmax to x.
func SoftmaxInPlace(x []float32) {
	if len(x) == 0 {
		return
	}
	maxV := x[0]
	for _, v := range x[1:] {
		if v > maxV {
			maxV = v
		}
	}
	var sum float32
	for i, v := range x {
		e := float32(math.Exp(float64(v - maxV)))
		x[i] = e
		sum += e
	}
	inv := float32(1.0) / sum
	for i := range x {
		x[i] *= inv
	}
}

// RoPE applies rotary position embedding to q and k vectors for one head.
// headDim must be even. cos/sin tables are precomputed for all positions.
func RoPE(q, k []float32, headDim int, pos int, cosTable, sinTable []float32) {
	half := headDim / 2
	base := pos * headDim
	for i := 0; i < half; i++ {
		c := cosTable[base+i]
		s := sinTable[base+i]
		q0, q1 := q[i], q[i+half]
		q[i] = q0*c - q1*s
		q[i+half] = q0*s + q1*c
		k0, k1 := k[i], k[i+half]
		k[i] = k0*c - k1*s
		k[i+half] = k0*s + k1*c
	}
}

// BuildRoPETables precomputes cos/sin for positions [0, maxSeq) and headDim.
func BuildRoPETables(maxSeq, headDim int, theta float32) (cos, sin []float32) {
	n := maxSeq * headDim
	cos = make([]float32, n)
	sin = make([]float32, n)
	half := headDim / 2
	for pos := 0; pos < maxSeq; pos++ {
		for i := 0; i < half; i++ {
			freq := float32(1.0) / float32(math.Pow(float64(theta), float64(2*i)/float64(headDim)))
			angle := float32(pos) * freq
			c := float32(math.Cos(float64(angle)))
			s := float32(math.Sin(float64(angle)))
			idx := pos*headDim + i
			cos[idx] = c
			sin[idx] = s
			cos[pos*headDim+i+half] = c
			sin[pos*headDim+i+half] = s
		}
	}
	return cos, sin
}

// EmbeddingLookup copies rows from embed table into out [seqLen, hiddenDim].
func EmbeddingLookup(embed []float32, hiddenDim int, tokens []int32, out []float32) {
	for t, tok := range tokens {
		src := embed[int(tok)*hiddenDim:]
		dst := out[t*hiddenDim:]
		copy(dst, src)
	}
}

// Argmax returns index of maximum value.
func Argmax(x []float32) int {
	best := 0
	for i := 1; i < len(x); i++ {
		if x[i] > x[best] {
			best = i
		}
	}
	return best
}
