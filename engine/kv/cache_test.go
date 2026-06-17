package kv

import "testing"

func TestCacheWriteRead(t *testing.T) {
	c := New(2, 8, 2, 4)
	k := make([]float32, 8)
	v := make([]float32, 8)
	for i := range k {
		k[i] = float32(i)
		v[i] = float32(i + 100)
	}
	c.Write(0, 0, k, v)
	c.Advance(1)
	kv := c.KView(0)
	if kv[0] != 0 || kv[7] != 7 {
		t.Fatalf("k view %v", kv[:8])
	}
}
