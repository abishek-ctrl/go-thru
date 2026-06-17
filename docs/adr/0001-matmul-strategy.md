# ADR-0001: Matmul Strategy

## Status

Accepted

## Context

Matrix multiplication dominates inference time (>90% of forward pass). Go has no SIMD in the standard language and no built-in BLAS. We need a default strategy that balances correctness, performance, and the "pure Go static binary" goal.

## Decision

Use **cache-tiled matmul with goroutine parallelism** as the default implementation in `engine/ops`.

- Tile size: 64×64 (tunable via constant)
- Parallelism: split output rows across `min(GOMAXPROCS, numRowTiles)` workers
- Fast path: `MatVec` for decode steps where batch dimension is 1
- Naive matmul retained for small matrices in unit tests

## Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| Naive only | Simple | 5–10× too slow for demo |
| cgo + OpenBLAS | Near-optimal speed | Breaks static binary, adds native dep |
| Go assembly (AVX2/NEON) | Best pure-Go perf | High complexity; deferred to Phase 2 build tag |

## Consequences

- Expected 15–30 tok/s on SmolLM2-135M (M-series CPU) with tiled matmul
- Assembly SIMD can be added behind `//go:build amd64 || arm64` without changing model code
- Quantization (int8/int4) is a separate follow-up ADR

## References

- [docs/GO_THRU_INFERENCE_ENGINE_PLAN.md](../GO_THRU_INFERENCE_ENGINE_PLAN.md) §4
