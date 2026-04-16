# TriAttention & TurboQuant Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the three remaining post-port gaps: TriAttention long-transcript compression crash, Gemma4 TriAttention runtime instability, and TurboQuant quality degradation on real models.

**Architecture:** Treat this as a hardening pass, not feature work. Fix shared TriAttention correctness first, then Gemma4-specific runtime correctness, then TurboQuant quality. Every step must be proven by a failing test or reproducible smoke command before implementation, and every fix must preserve already-working demo paths in `llm-tool`.

**Tech Stack:** Swift 6, `mlx-swift-lm`, `mlx-swift-examples`, `xcodebuild`, `mlx-run`, `MLX`, `MLXNN`, `swift-transformers`, `swift-jinja`.

---

### Task 1: Add grouped-query TriAttention shape validation tests

**Files:**
- Modify: `Tests/MLXLMTests/TriAttentionTests.swift`
- Modify: `Tests/MLXLMTests/KVCacheTests.swift`

**Step 1: Write the failing grouped-query regression test**

Add a test that creates calibration tensors with `qHeads = 16`, `kvHeads = 8`, `nFreqs = 32` and asserts that grouped-query shape handling is valid when `qHeads % kvHeads == 0`.

Example test:

```swift
@Test
func testTriAttentionGroupedQueryCalibrationShapeIsAccepted() throws {
    let layer = TriAttentionLayerCalibration(
        qCenterReal: MLXArray.zeros([16, 32], dtype: .float32),
        qCenterImag: MLXArray.zeros([16, 32], dtype: .float32),
        qMeanNorm: MLXArray.ones([16, 32], dtype: .float32)
    )
    let calibration = TriAttentionCalibrationData(layers: [layer], qHeads: 16, kvHeads: 8)
    let rope = TriAttentionRoPEConfig(
        headDim: 128,
        rotatedDims: 64,
        traditional: false,
        omega: MLXArray.ones([32], dtype: .float32)
    )

    let keys = MLXArray.zeros([1, 8, 1025, 128], dtype: .float32)

    _ = scoreKeys(
        cachedKeys: keys,
        currentPosition: 1025,
        layerCalibration: layer,
        calibration: calibration,
        rope: rope,
        offsets: MLXArray([1, 2, 4], dtype: .float32)
    )
}
```

**Step 2: Write the failing invalid-shape regression test**

Add a second test that uses mismatched calibration/runtime heads and expects a **clear TriAttention error**, not an MLX reshape crash.

```swift
@Test
func testTriAttentionRejectsGroupedQueryCalibrationMismatch() {
    // qHeads 8, kvHeads 8 in calibration, but runtime keys imply a different grouped-query layout
    // Expect a typed TriAttention error before reshape.
}
```

**Step 3: Run the tests to verify failure**

Run:

```bash
xcodebuild test -scheme "mlx-swift-lm-Package" -destination "platform=macOS" -configuration Debug \
  -only-testing "MLXLMTests/testTriAttentionGroupedQueryCalibrationShapeIsAccepted()" \
  -only-testing "MLXLMTests/testTriAttentionRejectsGroupedQueryCalibrationMismatch()"
```

Expected: one or both fail with the current reshape-based behavior.

**Step 4: Commit**

```bash
GIT_MASTER=1 git add Tests/MLXLMTests/TriAttentionTests.swift Tests/MLXLMTests/KVCacheTests.swift
GIT_MASTER=1 git commit -m "test: add grouped-query TriAttention shape regressions" -m "Ultraworked with [Sisyphus](https://github.com/code-yeongyu/oh-my-openagent)" -m "Co-authored-by: Sisyphus <clio-agent@sisyphuslabs.ai>"
```

---

### Task 2: Add explicit TriAttention calibration/runtime compatibility validation

**Files:**
- Modify: `Libraries/MLXLMCommon/TriAttention.swift`
- Test: `Tests/MLXLMTests/TriAttentionTests.swift`

**Step 1: Implement a dedicated validation helper**

Add a private helper in `TriAttention.swift`, called before any reshape in `scoreKeys(...)`:

```swift
private func validateTriAttentionShapeCompatibility(
    calibration: TriAttentionCalibrationData,
    layerCalibration: TriAttentionLayerCalibration,
    runtimeKVHeads: Int,
    runtimeHeadDim: Int,
    rope: TriAttentionRoPEConfig
) throws
```

It must validate:
- `calibration.qHeads % calibration.kvHeads == 0`
- `runtimeKVHeads == calibration.kvHeads`
- `layerCalibration.qCenterReal.dim(0) == calibration.qHeads`
- `layerCalibration.qCenterReal.dim(1) == rope.rotatedDims / 2`
- `runtimeHeadDim >= rope.rotatedDims`

**Step 2: Add a typed error case**

Extend `TriAttentionError` with a shape/configuration mismatch case, e.g.:

```swift
case incompatibleCalibration(String)
```

**Step 3: Call validation before reshape**

In `scoreKeys(...)`, validate before:

```swift
let qCenterMagGrouped = qCenterMag.reshaped(kvHeads, repeats, nFreqs)
```

**Step 4: Run tests to verify pass**

Run the two tests from Task 1.

**Step 5: Add one long-context smoke guard**

Add a narrow regression test that ensures long-context compression failure becomes a typed error if shapes are invalid.

**Step 6: Commit**

```bash
GIT_MASTER=1 git add Libraries/MLXLMCommon/TriAttention.swift Tests/MLXLMTests/TriAttentionTests.swift
GIT_MASTER=1 git commit -m "fix: validate TriAttention grouped-query shapes before compression" -m "Ultraworked with [Sisyphus](https://github.com/code-yeongyu/oh-my-openagent)" -m "Co-authored-by: Sisyphus <clio-agent@sisyphuslabs.ai>"
```

---

### Task 3: Reproduce and fix long-transcript TriAttention compression on Qwen3.5

**Files:**
- Modify: `Libraries/MLXLMCommon/TriAttention.swift`
- Verify: `mlx-swift-examples/Tools/llm-tool` runtime only

**Step 1: Reproduce with the real transcript**

Run the known failing command:

```bash
cd /Volumes/DATA/mlx-swift-examples

./mlx-run llm-tool triattention calibrate \
  --model mlx-community/Qwen3.5-2B-4bit \
  --prompt-file "/Users/akira/Downloads/Session Mar 4, 3:30 PM - Transcript.md" \
  --output /tmp/qwen35-transcript-triattention.safetensors \
  --prefill-step-size 128

/usr/bin/time -l ./mlx-run llm-tool eval \
  --model mlx-community/Qwen3.5-2B-4bit \
  --system "You are an expert meeting summarizer. Summarize the transcript into four sections: Summary, Decisions, Action Items, Open Questions. Be concise and factual." \
  --prompt-file "/Users/akira/Downloads/Session Mar 4, 3:30 PM - Transcript.md" \
  --max-tokens 256 \
  --tri-attention-calibration /tmp/qwen35-transcript-triattention.safetensors \
  --tri-attention-budget 1024 \
  --tri-attention-divide-length 128 \
  --tri-attention-protect-recent 128 \
  --tri-attention-protect-initial 4
```

Expected today: crash or invalid runtime behavior.

**Step 2: Make the smallest fix implied by validation results**

Do not guess. Use the error produced after Task 2 validation to patch the exact grouped-query assumption. If the grouped-query layout itself is correct and only the reshape math is wrong, patch the reshape math only.

**Step 3: Re-run the exact real transcript command**

Success criteria:
- no crash
- output remains usable and recognizably close to baseline summarization

**Step 4: Capture A/B numbers**

Run baseline and TriAttention again and record:
- real time
- peak memory footprint
- prompt tokens
- generation tokens

**Step 5: Commit**

```bash
GIT_MASTER=1 git add Libraries/MLXLMCommon/TriAttention.swift
GIT_MASTER=1 git commit -m "fix: restore long-context TriAttention compression on Qwen3.5" -m "Ultraworked with [Sisyphus](https://github.com/code-yeongyu/oh-my-openagent)" -m "Co-authored-by: Sisyphus <clio-agent@sisyphuslabs.ai>"
```

---

### Task 4: Add a dedicated Gemma4 TriAttention smoke regression

**Files:**
- Modify: `IntegrationTesting/` or `Tests/MLXLMTests/TriAttentionTests.swift` (choose the existing place that already hosts real-model smoke coverage)
- Modify: `Libraries/MLXLMCommon/TriAttention.swift` only if root cause requires it

**Step 1: Add a failing Gemma4 smoke expectation**

Create a narrow smoke test or documented smoke script for:
- baseline Gemma4 eval
- `triattention calibrate`
- `eval --tri-attention-calibration ...`

If a real automated test is too expensive, add an integration helper mirroring the Qwen3.5 smoke command sequence.

**Step 2: Investigate transparency invariants**

Specifically review and, if needed, patch `TriAttentionCache` to behave as a true proxy when compression is inactive:
- `innerState()` should mirror base cache if currently missing
- `offset` should mirror `base.offset` after update/trim
- no manual offset drift

**Step 3: Re-run Gemma4 smoke path**

Use:

```bash
./mlx-run llm-tool triattention calibrate \
  --model mlx-community/gemma-4-e2b-it-4bit \
  --prompt-file /tmp/gemma4-triattention.txt \
  --output /tmp/gemma4-triattention.safetensors \
  --prefill-step-size 32

./mlx-run llm-tool eval \
  --model mlx-community/gemma-4-e2b-it-4bit \
  --prompt "Say hello in one short sentence." \
  --max-tokens 32 \
  --tri-attention-calibration /tmp/gemma4-triattention.safetensors \
  --tri-attention-budget 2048
```

Success criteria:
- no crash
- no immediate pathological EOS/runtime failure
- output is at least baseline-like and usable

**Step 4: Commit**

```bash
GIT_MASTER=1 git add Libraries/MLXLMCommon/TriAttention.swift Libraries/MLXLLM/Models/Gemma4Text.swift Tests/MLXLMTests/TriAttentionTests.swift IntegrationTesting
GIT_MASTER=1 git commit -m "fix: stabilize Gemma4 TriAttention runtime path" -m "Ultraworked with [Sisyphus](https://github.com/code-yeongyu/oh-my-openagent)" -m "Co-authored-by: Sisyphus <clio-agent@sisyphuslabs.ai>"
```

---

### Task 5: Add TurboQuant synthetic parity tests beyond current toy coverage

**Files:**
- Modify: `Tests/MLXLMTests/KVCacheTests.swift`
- Modify: `Libraries/MLXLMCommon/AttentionUtils.swift` and/or `KVCache.swift` only if tests expose an issue

**Step 1: Add a failing packed-vs-materialized parity matrix test**

Cover multiple synthetic axes:
- `headDim` values that stress pack/unpack boundaries
- more than one `sequenceChunkSize`
- multiple `qHeads / kvHeads` repeat factors
- more than one seed

Example structure:

```swift
@Test
func testTurboQuantPackedParityMatrix() throws {
    for chunk in [1, 8, 32] {
        for seed in [0, 7] {
            // build synthetic q/k/v
            // compare packed helper vs materialized helper within tolerance
        }
    }
}
```

**Step 2: Run the test to see the first real divergence**

**Step 3: Fix only the first exposed numeric inconsistency**

Likely areas:
- pack/unpack bit boundary logic
- `materializeTurboQuantHeadRange(...)`
- chunked online softmax accumulation in `turboQuantChunkedHeadAttention(...)`
- repeated-head broadcast logic

**Step 4: Re-run matrix test until green**

**Step 5: Commit**

```bash
GIT_MASTER=1 git add Tests/MLXLMTests/KVCacheTests.swift Libraries/MLXLMCommon/AttentionUtils.swift Libraries/MLXLMCommon/KVCache.swift
GIT_MASTER=1 git commit -m "test: expand TurboQuant packed attention parity coverage" -m "Ultraworked with [Sisyphus](https://github.com/code-yeongyu/oh-my-openagent)" -m "Co-authored-by: Sisyphus <clio-agent@sisyphuslabs.ai>"
```

---

### Task 6: Add real-model TurboQuant quality smoke checks

**Files:**
- Modify: `IntegrationTesting/` helpers or add a small runtime smoke script under `scripts/`

**Step 1: Reproduce the current quality problem on a real model**

Use `mlx-community/Qwen3-0.6B-4bit` and/or `mlx-community/Qwen3.5-2B-4bit`.

Command pattern:

```bash
./mlx-run llm-tool eval \
  --model mlx-community/Qwen3.5-2B-4bit \
  --prompt "Say hello in one short sentence." \
  --max-tokens 32 \
  --turbo-quant \
  --turbo-quant-bits 3 \
  --turbo-quant-seed 7
```

Capture baseline vs TurboQuant outputs.

**Step 2: Define a minimal quality bar in code/docs**

For this hardening pass, quality target is not “perfect match.” It is:
- output remains readable
- does not devolve into garbled text on the smoke set

**Step 3: Fix the dominant quality source only**

Use the synthetic parity matrix from Task 5 to narrow down whether the issue is:
- quantization table / normalization
- chunked attention accumulation
- model-specific direct integration path

Fix the dominant issue first; do not redesign fused kernels here.

**Step 4: Re-run real-model smoke checks**

Success criteria:
- TurboQuant output on the smoke set is readable and baseline-like enough for demo use.

**Step 5: Commit**

```bash
GIT_MASTER=1 git add Libraries/MLXLMCommon/KVCache.swift Libraries/MLXLMCommon/AttentionUtils.swift Libraries/MLXLLM/Models/GPTOSS.swift Libraries/MLXLLM/Models/MiMoV2Flash.swift Libraries/MLXVLM/Models/Gemma4.swift IntegrationTesting scripts
GIT_MASTER=1 git commit -m "fix: improve TurboQuant quality on real-model smoke set" -m "Ultraworked with [Sisyphus](https://github.com/code-yeongyu/oh-my-openagent)" -m "Co-authored-by: Sisyphus <clio-agent@sisyphuslabs.ai>"
```

---

### Task 7: Final verification and demo matrix

**Files:**
- Verify all touched files above

**Step 1: Run focused unit/regression tests**

Run:

```bash
xcodebuild test -scheme "mlx-swift-lm-Package" -destination "platform=macOS" -configuration Debug
```

If too slow, at minimum run the exact tests added in Tasks 1, 2, and 5 plus the previously green TriAttention/TurboQuant regressions.

**Step 2: Re-run `llm-tool` smoke paths**

From `/Volumes/DATA/mlx-swift-examples`:

```bash
xcodebuild build -project "mlx-swift-examples.xcodeproj" -scheme "llm-tool" -destination 'platform=macOS'

./mlx-run llm-tool eval --model mlx-community/Qwen3.5-2B-4bit --prompt "Say hello in one short sentence." --max-tokens 32
./mlx-run llm-tool triattention calibrate --model mlx-community/Qwen3.5-2B-4bit --prompt-file /tmp/qwen35-triattention.txt --output /tmp/qwen35-triattention.safetensors --prefill-step-size 32
./mlx-run llm-tool eval --model mlx-community/Qwen3.5-2B-4bit --prompt "Say hello in one short sentence." --max-tokens 32 --tri-attention-calibration /tmp/qwen35-triattention.safetensors --tri-attention-budget 2048
./mlx-run llm-tool eval --model mlx-community/gemma-4-e2b-it-4bit --prompt "Say hello in one short sentence." --max-tokens 32
./mlx-run llm-tool eval --model mlx-community/Qwen3.5-2B-4bit --prompt "Say hello in one short sentence." --max-tokens 32 --turbo-quant --turbo-quant-bits 3 --turbo-quant-seed 7
```

**Step 3: Re-run transcript A/B for TriAttention**

Baseline and TriAttention-on-long-transcript should both complete, and TriAttention should no longer crash when `budget = 1024`.

**Step 4: Document final supported demo state**

Update `docs/specs/2026-04-16-triattention-turboquant-hardening.md` or a follow-up note with the final truth table:
- Qwen3 + TriAttention
- Qwen3.5 + TriAttention
- Gemma4 + TriAttention
- Qwen3/Qwen3.5 + TurboQuant

**Step 5: Final commit**

```bash
GIT_MASTER=1 git add Libraries Tests IntegrationTesting docs/specs/2026-04-16-triattention-turboquant-hardening.md
GIT_MASTER=1 git commit -m "fix: harden TriAttention and TurboQuant demo paths" -m "Ultraworked with [Sisyphus](https://github.com/code-yeongyu/oh-my-openagent)" -m "Co-authored-by: Sisyphus <clio-agent@sisyphuslabs.ai>"
```
