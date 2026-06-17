# go-thru

Pure-Go LLM inference engine with an OpenAI-compatible HTTP backend.

**No cgo. No Python. No llama.cpp.** Load SmolLM2 (Llama-family) weights from HuggingFace safetensors and run inference in a single static binary.

## Documentation

- [ADR-0001: Matmul Strategy](docs/adr/0001-matmul-strategy.md)

## Quick Start

### 1. Download a model

```bash
# Example: SmolLM2-135M (requires huggingface-cli or manual download)
pip install huggingface_hub
huggingface-cli download HuggingFaceTB/SmolLM2-135M --local-dir ./models/SmolLM2-135M
```

### 2. Build

```bash
go build -o thru ./cmd/thru
```

### 3. Generate text

```bash
./thru generate -model ./models/SmolLM2-135M -prompt "The meaning of life is" -max-tokens 64
```

### 4. Interactive chat

```bash
./thru chat -model ./models/SmolLM2-135M
```

### 5. Benchmark

```bash
./thru bench -model ./models/SmolLM2-135M -max-tokens 128
```

### 6. OpenAI-compatible API server

```bash
./thru serve -model ./models/SmolLM2-135M -addr :8080
```

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "smollm2",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64,
    "stream": true
  }'
```

## Architecture

```
cmd/thru          CLI + server entrypoint
server/           OpenAI-compatible HTTP adapter
engine/
  tensor/         Contiguous float32 tensors + workspace pool
  ops/            matmul (tiled+parallel), softmax, rmsnorm, rope, silu
  loader/         safetensors + config.json parser
  tokenizer/      BPE from tokenizer.json
  kv/             Pre-allocated KV cache
  model/          Llama-family forward pass
  sampler/        temperature, top-k, top-p
  generate/       Generation loop + streaming events
```

## Memory expectations

| Model | Weight RAM (F32) | Typical total |
|-------|------------------|---------------|
| SmolLM2-135M | ~540 MB | ~600–700 MB |
| SmolLM2-360M | ~1.4 GB | ~1.5–1.8 GB |

## Performance (CPU, tiled matmul)

| Model | Expected tok/s (M-series) |
|-------|---------------------------|
| SmolLM2-135M | 15–30 |
| SmolLM2-360M | 8–15 |

Run `./thru bench` on your hardware for actual numbers.

## Testing

```bash
go test ./...
go test -race ./...
go test -bench=. ./engine/ops/...
```

## Troubleshooting

**Tokenizer mismatch:** Ensure `tokenizer.json` is from the same model revision. Golden tests live in `testdata/`.

**OOM:** Reduce context length or use SmolLM2-135M. Weights are loaded as F32.

**Slow inference:** Expected on CPU with pure-Go matmul. Tiled parallelism is enabled by default; see ADR-0001 for optimization roadmap.

## License

MIT
