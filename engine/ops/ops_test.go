package ops

import (
	"math"
	"testing"
)

func TestMatMulSmall(t *testing.T) {
	// 2x2 @ 2x2
	a := []float32{1, 2, 3, 4}
	b := []float32{5, 6, 7, 8}
	want := []float32{19, 22, 43, 50}
	got := make([]float32, 4)
	MatMul(2, 2, 2, a, b, got)
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Fatalf("MatMul[%d] = %v, want %v", i, got, want)
		}
	}
}

func TestMatMulMatchesNaive(t *testing.T) {
	m, k, n := 32, 48, 64
	a := make([]float32, m*k)
	b := make([]float32, k*n)
	for i := range a {
		a[i] = float32(i%7) * 0.1
	}
	for i := range b {
		b[i] = float32(i%5) * 0.2
	}
	want := make([]float32, m*n)
	got := make([]float32, m*n)
	MatMulNaive(m, k, n, a, b, want)
	MatMul(m, k, n, a, b, got)
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-3 {
			t.Fatalf("mismatch at %d: got %v want %v", i, got[i], want[i])
		}
	}
}

func TestRMSNorm(t *testing.T) {
	x := []float32{1, 2, 3}
	w := []float32{1, 1, 1}
	out := make([]float32, 3)
	RMSNorm(x, w, 1e-5, out)
	var sum float32
	for _, v := range out {
		sum += v * v
	}
	// normalized vector should have RMS ~ 1
	rms := float32(math.Sqrt(float64(sum / 3)))
	if math.Abs(float64(rms-1)) > 0.01 {
		t.Fatalf("rms = %v", rms)
	}
}

func TestSoftmax(t *testing.T) {
	x := []float32{1, 2, 3}
	SoftmaxInPlace(x)
	var sum float32
	for _, v := range x {
		sum += v
	}
	if math.Abs(float64(sum-1)) > 1e-5 {
		t.Fatalf("sum = %v", sum)
	}
}

func TestSiLU(t *testing.T) {
	x := []float32{0, 1, -1}
	SiLUInPlace(x)
	if x[0] != 0 {
		t.Fatalf("silu(0) = %v", x[0])
	}
}

func TestRoPE(t *testing.T) {
	headDim := 4
	cos, sin := BuildRoPETables(4, headDim, 10000)
	q := []float32{1, 0, 1, 0}
	k := []float32{0, 1, 0, 1}
	q0 := append([]float32(nil), q...)
	RoPE(q, k, headDim, 1, cos, sin)
	if q[0] == q0[0] && q[1] == q0[1] {
		t.Fatal("RoPE should modify q")
	}
}

func BenchmarkMatMul(b *testing.B) {
	m, k, n := 256, 256, 256
	a := make([]float32, m*k)
	bb := make([]float32, k*n)
	c := make([]float32, m*n)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatMul(m, k, n, a, bb, c)
	}
}
