package ops

import "testing"

func BenchmarkMatVec(b *testing.B) {
	m, k := 2048, 2048
	a := make([]float32, m*k)
	x := make([]float32, k)
	y := make([]float32, m)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatVec(m, k, a, x, y)
	}
}

func BenchmarkSoftmax(b *testing.B) {
	x := make([]float32, 32000)
	for i := range x {
		x[i] = float32(i % 100)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cp := append([]float32(nil), x...)
		SoftmaxInPlace(cp)
	}
}
