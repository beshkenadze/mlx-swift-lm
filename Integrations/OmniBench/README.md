# omni-bench integration

`MLXLMOmniBench` is an optional Swift host adapter for omni-bench Foundation
0.6. It is a nested package so the main `mlx-swift-lm` dependency graph does not
force omni-bench onto applications that do not benchmark models.

The adapter:

- applies the identity-bearing `min_output_tokens`, `max_output_tokens`,
  `temperature`, `top_p`, and nullable `seed` controls;
- tokenizes the final manifest-bound prompt directly and never applies a model
  chat template or other host-specific prompt wrapper;
- normalizes the optional initial detokenizer space consistently with MLX LM
  streaming detokenizers while preserving all later whitespace;
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
`min_output_tokens=0` and `max_output_tokens=256`. Use `--output-tokens N` for
performance Tasks that require exactly `N` generated tokens, or set
`--min-output-tokens` and `--max-output-tokens` separately for quality Tasks
whose output may terminate naturally. It deliberately does not support load
claims while the adapter advertises `max_concurrency = 1`.

For example, a prepared FLORES+ translation Task should allow EOS rather than
forcing the performance profile's fixed output length:

```bash
swift run --package-path Integrations/OmniBench omni-bench-mlx-lm \
  --model-directory /absolute/path/to/model \
  --model-id organization/model \
  --model-artifact-sha256 sha256:<digest> \
  --quantization 4bit \
  --manifest /absolute/path/to/textgen.mt.flores_plus.en-de.v1/manifest.json \
  --registry-bundle /absolute/path/to/omni-bench/fixtures/consumer-bundle \
  --out /private/output/en-de.run-artifact.jsonl \
  --mode streaming \
  --backend-version <mlx-swift-lm-commit> \
  --implementation MLXLMGenerator \
  --environment-label private-local-qualification \
  --min-output-tokens 0 \
  --max-output-tokens 128 \
  --chunk-ms 100
```

MLX command-line tools must be able to locate the compiled Metal library. When
running outside Xcode, either make the build framework visible through
`DYLD_FRAMEWORK_PATH`, or place `mlx.metallib` beside the executable. The Metal
package version must match the `MLX_VERSION` declared by the resolved
`.build/checkouts/mlx-swift/Package.swift`; that value can differ from the
`mlx-swift` package tag. One private, reproducible setup is:

```bash
uv pip install --target /private/mlx-metal --no-deps \
  'mlx-metal==<resolved-MLX_VERSION>'
ln -s /private/mlx-metal/mlx/lib/mlx.metallib \
  "$(swift build --package-path Integrations/OmniBench --show-bin-path)/mlx.metallib"
```

Never commit the model, Metal library, prepared data, RunArtifact, or Result.

## Private token diagnostics

When parity fails, `omni-bench-mlx-lm-trace` records the exact raw prompt token
IDs and a bounded greedy generated-token prefix without changing the benchmark
artifact contract:

```bash
swift run --package-path Integrations/OmniBench omni-bench-mlx-lm-trace \
  --model-directory /absolute/path/to/model \
  --manifest /absolute/path/to/manifest.json \
  --sample-id sample-id \
  --max-tokens 64 \
  --out /private/output/token-trace.json
```

Token traces are reversible model inputs and outputs. Keep them private, compare
them only against a trace from the exact same model artifact and prompt, and
never upload them as RunArtifact, Result, or publication evidence.

Score the artifact with the exact pinned core and private references. A single
successful Result is qualification evidence, not a stable or publishable
performance baseline; public claims require the registered repeated-run
analysis and publication policy.
