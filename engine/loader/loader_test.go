package loader

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestSafetensorsRoundtrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.safetensors")
	data := map[string]struct {
		Dtype string
		Shape []int
		Data  []float32
	}{
		"weight": {Dtype: "F32", Shape: []int{2, 2}, Data: []float32{1, 2, 3, 4}},
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := WriteSafetensorsFixture(f, data); err != nil {
		t.Fatal(err)
	}
	f.Close()

	sf, err := OpenSafetensors(path)
	if err != nil {
		t.Fatal(err)
	}
	got, shape, err := sf.GetTensor("weight")
	if err != nil {
		t.Fatal(err)
	}
	if len(shape) != 2 || shape[0] != 2 {
		t.Fatalf("shape %v", shape)
	}
	want := []float32{1, 2, 3, 4}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func TestLoadConfigMissing(t *testing.T) {
	_, err := LoadConfig(t.TempDir())
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestWriteFixtureInMemory(t *testing.T) {
	var buf bytes.Buffer
	err := WriteSafetensorsFixture(&buf, map[string]struct {
		Dtype string
		Shape []int
		Data  []float32
	}{"a": {Dtype: "F32", Shape: []int{1}, Data: []float32{42}}})
	if err != nil {
		t.Fatal(err)
	}
	sf, err := OpenSafetensorsFromBytes(buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	v, _, err := sf.GetTensor("a")
	if err != nil || v[0] != 42 {
		t.Fatalf("got %v err %v", v, err)
	}
}

// OpenSafetensorsFromBytes parses from memory (for tests).
func OpenSafetensorsFromBytes(raw []byte) (*SafetensorsFile, error) {
	tmp := filepath.Join(os.TempDir(), "st-test.safetensors")
	if err := os.WriteFile(tmp, raw, 0600); err != nil {
		return nil, err
	}
	defer os.Remove(tmp)
	return OpenSafetensors(tmp)
}
