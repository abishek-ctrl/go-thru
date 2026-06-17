package tensor

import "testing"

func TestPoolPutGet(t *testing.T) {
	p := NewPool(2)
	ws := NewWorkspace(8, 2, 1, 4, 16, 10, 32)
	p.Put(ws)
	got := p.Get()
	if got == nil {
		t.Fatal("expected workspace")
	}
}
