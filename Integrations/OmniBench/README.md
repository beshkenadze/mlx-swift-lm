# omni-bench integration

`MLXLMOmniBench` is an optional Swift host adapter for omni-bench Foundation
0.6. It is a nested package so the main `mlx-swift-lm` dependency graph does not
force omni-bench onto applications that do not benchmark models.

The adapter:

- applies the identity-bearing `min_output_tokens`, `max_output_tokens`,
  `temperature`, `top_p`, and nullable `seed` controls;
- tokenizes the final manifest-bound prompt directly and never applies a model
  chat template or other host-specific prompt wrapper;
- uses a fresh MLX KV cache for every sample;
- supports batch and true raw-token streaming;
- emits exactly one `TokenEvent` per generated model token, including an empty
  `text_delta` when the tokenizer has no complete text for that token;
- advertises `max_concurrency = 1` until concurrent shared-model execution has
  separate load qualification.

omni-bench owns clocks, TTFT/ITL/finalization, RSS, scoring, parity, analysis,
and publication. The adapter never reports latency or quality itself.
Passing this package's tests establishes seam conformance, not model performance
qualification or permission to publish benchmark results.

Add `Integrations/OmniBench` as a local package dependency and construct
`MLXLMGenerator` with the loaded `MLXLMCommon.ModelContainer`. The nested
manifest pins the reviewed omni-bench core revision that defines the host seam.

```bash
swift test --package-path Integrations/OmniBench
```

## Private model qualification

The nested package also exposes `omni-bench-mlx-lm`, a local-only runner that
loads an already downloaded model directory and writes a canonical core
RunArtifact. Prepare the production Task with core first, then run the batch and
streaming profiles separately:

```bash
swift run --package-path Integrations/OmniBench omni-bench-mlx-lm \
  --model-directory /absolute/path/to/model \
  --model-id organization/model \
  --model-artifact-sha256 sha256:<digest> \
  --quantization 4bit \
  --manifest /absolute/path/to/textgen.performance.standard.v1/manifest.json \
  --registry-bundle /absolute/path/to/omni-bench/fixtures/consumer-bundle \
  --out /private/output/run-artifact.jsonl \
  --mode streaming \
  --backend-version <mlx-swift-lm-commit> \
  --implementation MLXLMGenerator \
  --environment-label private-local-qualification \
  --output-tokens 256 \
  --chunk-ms 100
```

Use `--mode batch` for `text_generation.batch_single.v1`; `--chunk-ms` only
affects streaming identity. The runner fixes concurrency to `1` and defaults to
the Task card's deterministic 256-token controls. It deliberately does not
support load claims while the adapter advertises `max_concurrency = 1`.

MLX command-line tools must be able to locate the compiled Metal library. When
running outside Xcode, follow the upstream `mlx-swift` command-line guidance:
make the build framework visible through `DYLD_FRAMEWORK_PATH`, or place the
matching generated `mlx.metallib` beside the executable. Never commit the model,
Metal library, prepared data, RunArtifact, or Result.

Score the artifact with the exact pinned core and private references. A single
successful Result is qualification evidence, not a stable or publishable
performance baseline; public claims require the registered repeated-run
analysis and publication policy.
