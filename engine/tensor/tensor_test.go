package tensor

import "testing"

func TestSize(t *testing.T) {
	if got := Size([]int{2, 3, 4}); got != 24 {
		t.Fatalf("Size = %d, want 24", got)
	}
}

func TestNewAndView(t *testing.T) {
	tens := New(2, 3)
	if tens.Numel() != 6 {
		t.Fatalf("Numel = %d", tens.Numel())
	}
	v := tens.View(3, 1, 3)
	if len(v.Data) != 3 {
		t.Fatalf("view len = %d", len(v.Data))
	}
}

func TestWorkspaceRegions(t *testing.T) {
	ws := NewWorkspace(64, 8, 4, 8, 128, 1000, 512)
	if len(ws.Buf) == 0 {
		t.Fatal("empty workspace")
	}
	sum := len(ws.Hidden) + len(ws.Q) + len(ws.K) + len(ws.V) +
		len(ws.Attn) + len(ws.AttnScore) + len(ws.FFN) + len(ws.Gate) + len(ws.Up) + len(ws.Logits) + len(ws.Residual)
	if sum != len(ws.Buf) {
		t.Fatalf("region sum %d != buf %d", sum, len(ws.Buf))
	}
}
