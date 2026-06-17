package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/abishekm/go-thru/engine/generate"
	"github.com/abishekm/go-thru/server"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}
	switch os.Args[1] {
	case "generate":
		cmdGenerate(os.Args[2:])
	case "chat":
		cmdChat(os.Args[2:])
	case "bench":
		cmdBench(os.Args[2:])
	case "serve":
		cmdServe(os.Args[2:])
	default:
		usage()
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintf(os.Stderr, `thru - pure Go LLM inference

Usage:
  thru generate -model <dir> -prompt <text> [flags]
  thru chat     -model <dir> [flags]
  thru bench    -model <dir> -prompt <text> [flags]
  thru serve    -model <dir> -addr :8080 [flags]

Flags (generate/chat/bench):
  -model string       Path to HuggingFace model directory
  -prompt string      Input prompt (generate/bench)
  -max-tokens int     Max tokens to generate (default 128)
  -temperature float  Sampling temperature (default 0.8)
  -top-p float        Nucleus sampling (default 0.9)
  -seed int           RNG seed (default 0)

Flags (serve):
  -addr string        Listen address (default :8080)
  -name string        Model name for API (default smollm2)
`)
}

func parseModelFlags(args []string) (modelDir string, rest []string) {
	fs := flag.NewFlagSet("model", flag.ExitOnError)
	fs.StringVar(&modelDir, "model", "", "model directory")
	_ = fs.Parse(args)
	return modelDir, fs.Args()
}

func cmdGenerate(args []string) {
	fs := flag.NewFlagSet("generate", flag.ExitOnError)
	modelDir := fs.String("model", "", "model directory")
	prompt := fs.String("prompt", "Hello", "prompt")
	maxTokens := fs.Int("max-tokens", 128, "max tokens")
	temp := fs.Float64("temperature", 0.8, "temperature")
	topP := fs.Float64("top-p", 0.9, "top-p")
	seed := fs.Int64("seed", 0, "seed")
	_ = fs.Parse(args)

	if *modelDir == "" {
		fmt.Fprintln(os.Stderr, "-model required")
		os.Exit(1)
	}

	eng, err := generate.NewEngine(*modelDir)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	req := generate.Request{
		Prompt:      *prompt,
		MaxTokens:   *maxTokens,
		Temperature: float32(*temp),
		TopP:        float32(*topP),
		Seed:        *seed,
	}

	ctx := context.Background()
	events, err := eng.Generate(ctx, req)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	fmt.Print(*prompt)
	for ev := range events {
		if ev.Err != nil {
			fmt.Fprintln(os.Stderr, ev.Err)
			os.Exit(1)
		}
		if ev.Done {
			if ev.Stats != nil {
				fmt.Fprintf(os.Stderr, "\n[%d prompt + %d output tokens, %.1f tok/s, %v]\n",
					ev.Stats.PromptTokens, ev.Stats.OutputTokens, ev.Stats.TokensPerSec, ev.Stats.Duration)
			}
			break
		}
		fmt.Print(ev.Token)
	}
	fmt.Println()
}

func cmdChat(args []string) {
	fs := flag.NewFlagSet("chat", flag.ExitOnError)
	modelDir := fs.String("model", "", "model directory")
	maxTokens := fs.Int("max-tokens", 256, "max tokens")
	temp := fs.Float64("temperature", 0.8, "temperature")
	topP := fs.Float64("top-p", 0.9, "top-p")
	_ = fs.Parse(args)

	if *modelDir == "" {
		fmt.Fprintln(os.Stderr, "-model required")
		os.Exit(1)
	}

	eng, err := generate.NewEngine(*modelDir)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	scanner := bufio.NewScanner(os.Stdin)
	fmt.Println("Chat (Ctrl-D to exit). Model:", *modelDir)
	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		prompt := generate.ApplyChatTemplate(line)
		events, err := eng.Generate(context.Background(), generate.Request{
			Prompt:      prompt,
			MaxTokens:   *maxTokens,
			Temperature: float32(*temp),
			TopP:        float32(*topP),
		})
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			continue
		}
		fmt.Print("assistant: ")
		for ev := range events {
			if ev.Err != nil {
				fmt.Fprintln(os.Stderr, ev.Err)
				break
			}
			if ev.Done {
				if ev.Stats != nil {
					fmt.Fprintf(os.Stderr, "\n[%.1f tok/s]\n", ev.Stats.TokensPerSec)
				}
				break
			}
			fmt.Print(ev.Token)
		}
		fmt.Println()
	}
}

func cmdBench(args []string) {
	fs := flag.NewFlagSet("bench", flag.ExitOnError)
	modelDir := fs.String("model", "", "model directory")
	prompt := fs.String("prompt", "The meaning of life is", "prompt")
	maxTokens := fs.Int("max-tokens", 64, "max tokens")
	_ = fs.Parse(args)

	if *modelDir == "" {
		fmt.Fprintln(os.Stderr, "-model required")
		os.Exit(1)
	}

	eng, err := generate.NewEngine(*modelDir)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	start := time.Now()
	events, err := eng.Generate(context.Background(), generate.Request{
		Prompt:      *prompt,
		MaxTokens:   *maxTokens,
		Temperature: 0.8,
		TopP:        0.9,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	var stats *generate.Stats
	for ev := range events {
		if ev.Err != nil {
			fmt.Fprintln(os.Stderr, ev.Err)
			os.Exit(1)
		}
		if ev.Done {
			stats = ev.Stats
		}
	}
	elapsed := time.Since(start)
	if stats != nil {
		fmt.Printf("bench: %d output tokens in %v (%.2f tok/s)\n",
			stats.OutputTokens, elapsed, stats.TokensPerSec)
	}
}

func cmdServe(args []string) {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	modelDir := fs.String("model", "", "model directory")
	addr := fs.String("addr", ":8080", "listen address")
	name := fs.String("name", "smollm2", "model name")
	_ = fs.Parse(args)

	if *modelDir == "" {
		fmt.Fprintln(os.Stderr, "-model required")
		os.Exit(1)
	}

	eng, err := generate.NewEngine(*modelDir)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	srv := server.New(eng, server.Config{
		Addr:           *addr,
		ModelName:      *name,
		MaxConcurrency: 4,
	})
	fmt.Println("Listening on", *addr)
	if err := srv.ListenAndServe(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
