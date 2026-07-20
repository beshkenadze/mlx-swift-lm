# omni-bench integration

`MLXLMOmniBench` is an optional Swift host adapter for omni-bench Foundation
0.6. It is a nested package so the main `mlx-swift-lm` dependency graph does not
force omni-bench onto applications that do not benchmark models.

The adapter:

- applies the identity-bearing `min_output_tokens`, `max_output_tokens`,
  `temperature`, `top_p`, and nullable `seed` controls;
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
