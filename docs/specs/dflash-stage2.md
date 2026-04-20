# DFlash Stage 2 Spec

## Scope

Stage 2 adds native DFlash speculative decoding to `MLXLMServer` for one
supported BF16 target/draft pair and exposes it through the existing
OpenAI-compatible `/v1/chat/completions` surface.

## Supported model pair

- Target model: `mlx-community/Qwen3-4B-bf16`
- Draft model: `z-lab/Qwen3-4B-DFlash-b16`
- Server model id: `dflash:qwen3-4b`

This is the only Stage 2 configuration that is expected to work out of the
box.

## Expected performance

- Expected decode speedup: about `3x` on `512`-token generations
- Source of expectation: Aryagm's `dflash-mlx` Qwen3-4B BF16 benchmark on M4 Max
- Actual throughput varies with prompt shape, hardware, and whether the model
  weights are already warm in the local cache

Use `scripts/bench-dflash.sh` to measure your local baseline vs DFlash speedup.

## Correctness guarantee

At `temperature=0`, DFlash is intended to be lossless relative to baseline
autoregressive decoding: committed tokens are verifier-approved, so the output
should remain bit-identical to the baseline target model for the supported pair.

## Memory profile

- Target weights: about `8 GB`
- Draft weights: about `1 GB`
- Additional memory: target + draft KV caches, which grow with prompt length

On Apple Silicon, plan around roughly `9+ GB` before accounting for KV cache
growth and general process overhead.

## CLI and HTTP usage

Start the server with the Stage 2 target and draft:

```bash
xcodebuild -scheme mlx-lm-server -configuration Debug -destination 'platform=macOS' build
.build-xcode/Build/Products/Debug/mlx-lm-server \
  --host 127.0.0.1 \
  --port 8080 \
  --model mlx-community/Qwen3-4B-bf16 \
  --dflash-target mlx-community/Qwen3-4B-bf16 \
  --dflash-draft z-lab/Qwen3-4B-DFlash-b16
```

Request a DFlash completion:

```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "dflash:qwen3-4b",
    "messages": [
      {"role": "user", "content": "Solve: 2x + 5 = 17. Show your work step by step."}
    ],
    "max_tokens": 48,
    "stream": false
  }'
```

Validate the route and warm the model cache:

```bash
scripts/smoke-dflash.sh --dflash
```

Benchmark baseline vs DFlash on the same prompt:

```bash
scripts/bench-dflash.sh
```

## Limitations

- BF16 only in Stage 2
- Qwen3 dense model family only in Stage 2
- Quantized targets are deferred to Stage 3+
- Qwen3.5 hybrid attention support is deferred to Stage 3.5+
- The documented performance target is based on external benchmarking, not a
  fixed guarantee for every machine

## References

- Implementation plan: `docs/plans/2026-04-19-dflash-stage2-bf16-mvp.md`
- Ecosystem comparison: `docs/research/2026-04-19-dflash-ecosystem-comparison.md`
