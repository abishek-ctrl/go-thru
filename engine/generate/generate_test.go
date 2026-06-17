package generate

import (
	"testing"
)

func TestApplyChatTemplate(t *testing.T) {
	s := ApplyChatTemplate("hello")
	if s == "" {
		t.Fatal("empty template")
	}
	if !contains(s, "hello") {
		t.Fatal("missing user content")
	}
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (s == sub || len(sub) == 0 ||
		(len(s) > 0 && findSub(s, sub)))
}

func findSub(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

func TestMatchesStop(t *testing.T) {
	if !matchesStop("hello world", []string{"world"}) {
		t.Fatal("should match stop")
	}
}
