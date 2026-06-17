package ops

import (
	"math"
	"testing"
)

// ReferenceMatMul for verification tests.
func ReferenceMatMul(m, k, n int, a, b, c []float32) {
	for i := range c {
		c[i] = 0
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for l := 0; l < k; l++ {
				sum += a[i*k+l] * b[l*n+j]
			}
			c[i*n+j] = sum
		}
	}
}

func TestMatMulReferenceParity(t *testing.T) {
	sizes := [][3]int{{4, 4, 4}, {7, 5, 3}, {33, 17, 25}}
	for _, sz := range sizes {
		m, k, n := sz[0], sz[1], sz[2]
		a := make([]float32, m*k)
		b := make([]float32, k*n)
		for i := range a {
			a[i] = float32(i%11) * 0.1
		}
		for i := range b {
			b[i] = float32(i%7) * 0.2
		}
		want := make([]float32, m*n)
		got := make([]float32, m*n)
		ReferenceMatMul(m, k, n, a, b, want)
		MatMul(m, k, n, a, b, got)
		for i := range want {
			if math.Abs(float64(got[i]-want[i])) > 1e-2 {
				t.Fatalf("size %v mismatch at %d: %v vs %v", sz, i, got[i], want[i])
			}
		}
	}
}

func TestMatVecReference(t *testing.T) {
	m, k := 5, 4
	a := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
	x := []float32{1, 0, 1, 0}
	y := make([]float32, m)
	MatVec(m, k, a, x, y)
	if y[0] != 4 {
		t.Fatalf("y[0]=%v want 4", y[0])
	}
}
