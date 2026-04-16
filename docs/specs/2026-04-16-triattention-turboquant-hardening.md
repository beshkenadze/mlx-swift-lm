# TriAttention & TurboQuant Hardening Spec

## Goal

Close the three known post-port gaps without expanding scope:

1. TriAttention compression bug on long transcript
2. Gemma4 TriAttention EOS/runtime issue
3. TurboQuant quality degradation on real models

This is a **gap-closure spec**, not a feature-expansion spec.

---

## Current supported state

### Working today

- `llm-tool` in `mlx-swift-examples` builds and runs through `xcodebuild` + `mlx-run`
- baseline generation works on:
  - `mlx-community/Qwen3-0.6B-4bit`
  - `mlx-community/Qwen3.5-2B-4bit`
  - `mlx-community/gemma-4-e2b-it-4bit`
- TriAttention CLI/demo path exists:
  - `triattention calibrate`
  - `eval --tri-attention-calibration ...`
- TriAttention is demo-ready for **short prompts** on:
  - Qwen3
  - Qwen3.5
- Gemma4 baseline path now works after upgrading `swift-transformers` / `swift-jinja`
- TurboQuant CLI/demo wiring exists:
  - `--turbo-quant`
  - `--turbo-quant-bits`
  - `--turbo-quant-seed`
- TurboQuant has:
  - packed storage
  - packed attention fast path
  - prompt-cache save/load

### Not working yet

- TriAttention compression path on long transcripts (real pruning active) crashes on Qwen3.5
- Gemma4 + TriAttention generation is still unstable / exits early under certain paths
- TurboQuant produces degraded / garbled output on real models despite correct plumbing

---

## Scope boundaries

### In scope

- Fix the three named issues only
- Add whatever validation and regression coverage is needed to close them safely
- Harden current code paths in `mlx-swift-lm` and, if necessary, their existing `llm-tool` usage path in `mlx-swift-examples`

### Out of scope

- New model families
- New CLI/product surface beyond the already-added `llm-tool` flags/commands
- New demo apps
- General long-context optimization work outside the known TriAttention bug
- TurboQuant fused-kernel redesign
- Joint TriAttention + TurboQuant mode
- Large benchmarking program across many models/tasks

---

## Sequencing

Work in this order:

1. **TriAttention long-transcript compression bug**
2. **Gemma4 TriAttention runtime issue**
3. **TurboQuant quality**

Why this order:

- Shared TriAttention correctness must be fixed before model-specific runtime support is trusted.
- Gemma4 runtime work should happen on top of a stable TriAttention core.
- TurboQuant quality is the only item in this list that is primarily a *quality* problem rather than a *correctness* problem.

---

## Issue 1: TriAttention compression bug on long transcript

### Symptom

Long-context summarization on `mlx-community/Qwen3.5-2B-4bit` works without TriAttention and with TriAttention when compression is effectively disabled, but fails when the budget is low enough to activate pruning.

Observed failure:

- baseline transcript summarization: works
- TriAttention calibration on the same transcript: works
- TriAttention summarization with `budget = 1024`: crashes with reshape error

Representative error:

```text
[reshape] Cannot reshape array of size 1024 into shape (2,4,32)
```

### Likely root cause

The failure is in the **compression/scoring path**, not in calibration collection or in the basic wrapper hookup.

Strongest evidence from current code:

- `TriAttentionCache.compress()` only runs when `cachedKeys.dim(2) > configuration.budget`
- the crash appears only when pruning activates
- `scoreKeys()` reshapes calibration tensors using runtime `kvHeads` and derived `repeats`
- this path assumes calibration head layout and runtime grouped-query layout are fully compatible

Current risk points:

- calibration `qHeads` and runtime `kvHeads` are not validated before reshape
- grouped-query assumption is implicit, not enforced
- test coverage only exercises small 1:1 head fixtures, not real `repeats > 1` grouped-query models

### Files most likely involved

- `Libraries/MLXLMCommon/TriAttention.swift`
  - `TriAttentionCache.compress()`
  - `scoreKeys(...)`
  - calibration shape handling
- `Libraries/MLXLLM/Models/Qwen35.swift`
  - grouped-query attention layout
- `Tests/MLXLMTests/TriAttentionTests.swift`
- `Tests/MLXLMTests/KVCacheTests.swift`

### Required design changes

1. Make calibration/runtime shape compatibility explicit.
2. Validate head-count invariants before any reshape.
3. Fail with a targeted TriAttention shape/configuration error instead of raw MLX reshape failure.
4. Add grouped-query coverage where `qHeads / kvHeads > 1`.

### Acceptance criteria

- Long transcript summarization on `Qwen3.5-2B-4bit` with TriAttention budget below prompt length completes without crash.
- Compression activates at least once during the run.
- Output remains usable and recognizably close to the baseline summary.
- A grouped-query regression test exists and fails with a clear error on invalid calibration/runtime head mismatch.

---

## Issue 2: Gemma4 TriAttention EOS/runtime issue

### Symptom

Gemma4 baseline generation works. TriAttention calibration can be made to work. But Gemma4 generation with TriAttention enabled is still not reliably demo-ready.

Historically observed states in this branch included:

- baseline Gemma4 parser/runtime failure due to old `swift-jinja` dependency (fixed)
- missing capture hook for Gemma4 attention path (fixed)
- model-level RoPE/headDim discovery gaps (partially fixed)
- TriAttention runtime still vulnerable to Gemma4-specific shared-KV / cache topology issues

### Likely root cause bucket

Gemma4 is not a plain one-layer-one-cache topology.

Current design risks:

1. **Shared-KV topology**
   - some Gemma4 layers reuse `(K,V)` tuples produced by earlier layers instead of owning their own cache update path
   - TriAttention wrapper logic is cache-centric, so “shared but not cache-owning” layers may diverge from assumptions

2. **Wrapper transparency**
   - `TriAttentionCache` must behave as a transparent proxy when compression is inactive
   - Oracle specifically flagged `TriAttentionCache` as suspect if offset or `innerState()` do not mirror the base cache perfectly

3. **Flat kvHeads exposure vs actual Gemma4 topology**
   - Gemma4 may expose flat `kvHeads` arrays while operational behavior differs across full-attention and shared-KV layers

### Files most likely involved

- `Libraries/MLXLLM/Models/Gemma4Text.swift`
- `Libraries/MLXLMCommon/TriAttention.swift`
- `Libraries/MLXLMCommon/KVCache.swift`
- `Libraries/MLXLMCommon/RoPEApplication.swift`

### Required design changes

1. Document Gemma4 cache ownership vs shared-KV consumption explicitly.
2. Make “transparent when compression inactive” a hard invariant for `TriAttentionCache`.
3. Separate Gemma4-specific support rules from general TriAttention guarantees.

### Acceptance criteria

- `gemma-4-e2b-it-4bit` baseline remains unchanged.
- TriAttention calibration succeeds.
- TriAttention generation on Gemma4 no longer crashes or exits immediately from runtime pathology.
- Gemma4 text path and wrapper path are both covered by a smoke matrix.

---

## Issue 3: TurboQuant quality

### Symptom

TurboQuant wiring is complete:

- generation-time opt-in
- packed storage
- packed attention fast path
- prompt-cache save/load
- CLI integration in `llm-tool`

But real-model text quality is degraded. On smoke-tested models, output is garbled or significantly worse than baseline.

### Likely root cause buckets

This is a **quality** issue, not a plumbing issue.

Most likely sources, in order:

1. **Quantization quality itself**
   - fixed 3-bit centroid table
   - no model-aware calibration stage
   - deterministic sign/Hadamard path may be too coarse as currently applied

2. **Packed attention numerics**
   - packed/materialized paths may match on toy tensors but drift on real-model distributions
   - chunked online softmax path adds another source of numeric mismatch

3. **Model-specific path differences**
   - GPTOSS, MiMoV2Flash, Gemma4 all have direct TurboQuant branches
   - unsupported features (sinks, certain masks) are already rejected, but remaining supported paths may still diverge in subtle ways

4. **Insufficient quality validation**
   - current tests prove structural correctness and toy-path parity
   - they do not yet prove real-model text quality parity

### Files most likely involved

- `Libraries/MLXLMCommon/KVCache.swift`
  - quantize/dequantize path
  - packed storage
  - sign/Hadamard transform
  - centroid table
- `Libraries/MLXLMCommon/AttentionUtils.swift`
  - packed attention fast path
  - chunked streaming attention
- `Libraries/MLXLLM/Models/GPTOSS.swift`
- `Libraries/MLXLLM/Models/MiMoV2Flash.swift`
- `Libraries/MLXVLM/Models/Gemma4.swift`

### Required design changes

1. Define a **supported TurboQuant contract** first.
2. Separate:
   - pack/unpack/storage correctness
   - packed attention numerical parity
   - real-model output quality
3. Add a small representative quality suite instead of broad benchmarking.

### Acceptance criteria

- Packed attention matches materialized TurboQuant attention within tight numeric tolerance on synthetic coverage set.
- Real-model generation on a small agreed model set remains readable and close enough to baseline for demo use.
- Prompt-cache save/load/copy/trim continue to preserve continuation behavior.
- Unsupported settings are rejected early and clearly.

---

## External references that should guide implementation

### TriAttention

- paper: `arXiv:2604.04921`
- MLX reference path: `Blaizzy/mlx-vlm` PR `#985`
- important pattern:
  - offline calibration
  - runtime budget
  - protected sinks + recent tokens
  - trigonometric scoring over pre-RoPE query statistics

### TurboQuant

- paper: `arXiv:2504.19874`
- MLX reference path:
  - `mlx-vlm` TurboQuant implementation and related PRs
- important pattern:
  - randomized Hadamard preconditioning
  - Lloyd-Max scalar quantization
  - packed runtime path
  - staged path before full fused kernels is acceptable

### Broader public usage patterns

Outside MLX, the dominant public TA usage pattern is:

1. calibrate offline
2. run with fixed KV budget
3. use long-context inference backend

That validates the current `llm-tool` direction (`triattention calibrate` + runtime calibration file + budget), even if the remaining correctness/quality gaps are still being closed.

---

## Validation matrix

### Shared TriAttention

- Qwen3 short prompt
- Qwen3.5 short prompt
- Qwen3.5 long transcript

### Gemma4 TriAttention

- Gemma4 baseline
- Gemma4 calibration
- Gemma4 short generation with TriAttention
- Gemma4 long generation only if short path is stable

### TurboQuant

- synthetic pack/unpack parity
- synthetic packed-vs-materialized attention parity
- real-model smoke generation:
  - one Qwen model
  - one additional model only after first passes

---

## Defer list

These should **not** be folded into this hardening spec unless a discovered root cause forces it:

- new CLI features
- new example apps
- new model families
- full benchmark suite
- fused TurboQuant kernel redesign
- generalized long-context optimization outside the known TriAttention bug

---

## Exit condition

This spec is complete when:

1. TriAttention long-transcript compression works on Qwen3.5 without crash
2. Gemma4 + TriAttention is stable enough for smoke/demo use
3. TurboQuant quality is good enough for a small real-model demo set
4. All three items have explicit regression coverage and clearly bounded unsupported cases
