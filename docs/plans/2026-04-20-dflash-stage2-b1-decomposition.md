# DFlash Stage 2 — Wave B Task B1 Decomposition

> **For Claude/Codex:** This decomposes the original plan's Task B1 (`DFlashIterator`) into 5 atomic sub-tasks. Each is its own commit. Execute via `superpowers:executing-plans` methodology.

**Parent plan:** `docs/plans/2026-04-19-dflash-stage2-bf16-mvp.md` Wave B Task B1

**Why decompose:** B1 is the highest-risk task of Stage 2 — ~400 LOC of off-by-one math in the speculative verify+rollback loop. Monolithic implementation makes bisection painful when parity test fails. Five small commits, each with its own test, give us clean rollback points and make each error surface at its own boundary.

**Branch state at start:** `feat/dflash-stage2-bf16` at `fc8f150` (Wave A complete: A1+A2+A3+A4 all committed and pushed).

---

## Architecture recap (from `dflash.py.spec_generate`)

```
prefill → pendingTokens=[first_sampled_token], committedLength=prompt_len, targetHidden=captured

loop until done:
  block = [last_committed, MASK, MASK, ..., MASK]  # 16 tokens
  noise = target.embedTokens(block)
  draft_hidden = drafter(noise, targetHidden)
  draft_logits = target.lmHead(draft_hidden[:, 1..<16])  # 15 logits
  draft_tokens = argmax(draft_logits)                    # 15 proposed
  reset draft_cache to committedLength
  block[1..<16] = draft_tokens                           # fill masks
  (verify_logits, verify_hiddens) = target(block)        # 1 forward
  posterior = argmax(verify_logits)                      # 16 positions
  acceptance_len = longest_prefix(block[1:] == posterior[:-1])
  commit: accept draft_tokens[0..<acc_len] + posterior[acc_len]  # bonus
  trim target_cache: blockSize - (acc_len + 1) tokens
  committedLength += acc_len + 1
  targetHidden = verify_hiddens[:, 0..<(acc_len+1), :]   # slice
```

---

## Pre-B: `DFlashTargetModel` protocol + Qwen3 conformance

**Why:** `DFlashIterator` needs three operations from the target model that are NOT in `LanguageModel`: `embedTokens`, `lmHead`-or-fallback, and `captureHiddenStatesAndLogits`. Rather than make the iterator Qwen3-specific, define a narrow protocol and conform `Qwen3Model` to it. Keeps the door open for Qwen3.5, LLaMA 3.1 in later stages.

**Files:**
- New: `Libraries/MLXLMCommon/DFlash/DFlashTargetModel.swift` (~40 LOC)
- Modify: `Libraries/MLXLLM/Models/Qwen3.swift` (+~20 LOC — add `embedTokens` and `applyLMHead` public wrappers + `DFlashTargetModel` conformance)

**Step Pre-B.1 — Define protocol:**

```swift
// Libraries/MLXLMCommon/DFlash/DFlashTargetModel.swift
import MLX

/// Target model operations required by the DFlash speculative decoding loop.
/// Conformance exposes the three forward-pass variants DFlash needs:
/// - Token embedding lookup (for building the noise block)
/// - LM head projection (for converting drafter hidden states to vocab logits)
/// - Forward-with-captured-hidden-states (for seeding the drafter each round)
public protocol DFlashTargetModel: LanguageModel {
    /// Map token IDs to input embeddings. Shape: `[batch, seqLen] → [batch, seqLen, hiddenSize]`.
    func embedTokens(_ inputs: MLXArray) -> MLXArray

    /// Project the final hidden state to vocab logits via the LM head
    /// (or tied embeddings if `tie_word_embeddings`).
    /// Shape: `[batch, seqLen, hiddenSize] → [batch, seqLen, vocabSize]`.
    func applyLMHead(_ hidden: MLXArray) -> MLXArray

    /// Forward pass capturing hidden states at the specified layer indices AND logits.
    /// Used by DFlash to get both the verification logits and the drafter's conditioning hidden states in one pass.
    func captureHiddenStatesAndLogits(
        inputs: MLXArray,
        layerIndices: [Int],
        cache: [KVCache]?
    ) -> (hiddenStates: [MLXArray], logits: MLXArray)
}
```

**Step Pre-B.2 — Add wrappers on `Qwen3Model`:**

Inside `Libraries/MLXLLM/Models/Qwen3.swift`, in the `Qwen3Model` class body, add two small public methods:

```swift
public func embedTokens(_ inputs: MLXArray) -> MLXArray {
    model.embedTokens(inputs)
}

public func applyLMHead(_ hidden: MLXArray) -> MLXArray {
    if let lmHead {
        return lmHead(hidden)
    }
    return model.embedTokens.asLinear(hidden)
}
```

Add conformance (can be via extension, must be in a file that sees both Qwen3Model and DFlashTargetModel — place at the end of `Qwen3.swift`):

```swift
extension Qwen3Model: DFlashTargetModel {}
```

**Step Pre-B.3 — Unit test:** `Tests/MLXLMTests/DFlashTargetModelConformanceTests.swift`

Confirms Qwen3Model conforms and the methods dispatch correctly (no runtime MLX needed — compile-time conformance check is enough):

```swift
import MLXLLM
import MLXLMCommon
import Testing

struct DFlashTargetModelConformanceTests {
    @Test("Qwen3Model conforms to DFlashTargetModel")
    func testQwen3Conforms() {
        func accepts<T: DFlashTargetModel>(_ model: T) { _ = model }
        // Compile-time check: the function accepts Qwen3Model
        _ = accepts as (Qwen3Model) -> Void
    }
}
```

**Commit:** `feat(mlxlmcommon): DFlashTargetModel protocol + Qwen3Model conformance`

---

## B1.1: Iterator skeleton + prefill

**Files:**
- New: `Libraries/MLXLMCommon/DFlash/DFlashIterator.swift` (~150 LOC for this step)
- New: `Tests/MLXLMTests/DFlashIteratorPrefillTests.swift`

**What:** Set up `DFlashIterator` struct with state fields, init, and `prefill()` only. Do NOT implement speculation loop yet. `next()` returns buffered tokens from prefill only, returns nil when buffer empty.

**Signature:**
```swift
public struct DFlashIterator: TokenIteratorProtocol {
    public typealias Element = Int

    let target: any DFlashTargetModel
    let drafter: DFlashDraftModel
    let draftConfig: DFlashDraftConfig
    let blockSize: Int
    let maskTokenId: Int
    let targetLayerIds: [Int]
    let stopTokenIds: Set<Int>
    let maxTokens: Int?

    var targetCache: [KVCache]
    var draftCache: [KVCache]
    var committedLength: Int
    var lastTargetHidden: MLXArray?
    var pendingTokens: [Int]
    var pendingIndex: Int
    var emittedCount: Int
    var firstStepDone: Bool

    public private(set) var totalProposed: Int = 0
    public private(set) var totalAccepted: Int = 0
    public var acceptanceRate: Double {
        totalProposed == 0 ? 0 : Double(totalAccepted) / Double(totalProposed)
    }

    public init(
        promptTokens: MLXArray,    // [1, promptLen]
        target: any DFlashTargetModel,
        drafter: DFlashDraftModel,
        draftConfig: DFlashDraftConfig,
        stopTokenIds: Set<Int> = [],
        maxTokens: Int? = nil
    ) throws {
        // 1. Store refs, config values, stop tokens
        // 2. Allocate caches: target via target.newCache(), draft as [KVCacheSimple] x drafter layers count
        //    Verify both trimmable; throw if not (KVCacheError("Speculative decoding requires trimmable KV caches"))
        // 3. Init state: committedLength=0, pendingTokens=[], pendingIndex=0, firstStepDone=false
        // Do NOT run prefill here — deferred to first next() call
    }

    public mutating func next() -> Int? {
        if pendingIndex < pendingTokens.count {
            let t = pendingTokens[pendingIndex]; pendingIndex += 1
            emittedCount += 1
            return t
        }
        if let max = maxTokens, emittedCount >= max { return nil }
        if !firstStepDone {
            prefill()
            firstStepDone = true
            // Emit first token from prefill
            if !pendingTokens.isEmpty {
                let t = pendingTokens[pendingIndex]; pendingIndex += 1
                emittedCount += 1
                return t
            }
        }
        // B1.1 stops here — speculation loop not yet implemented
        return nil
    }

    private mutating func prefill() {
        // Call target.captureHiddenStatesAndLogits(inputs: promptTokens, layerIndices: targetLayerIds, cache: targetCache)
        // Sample first token from logits[:, -1, :]  (greedy: argmax)
        // pendingTokens = [firstToken]
        // committedLength = promptTokens.dim(1)
        // lastTargetHidden = concatenated(hiddenStates, axis: -1)  // [1, promptLen, 5*hidden]
    }
}
```

**Test:** 
- Gated on `dflashMetallibAvailable`
- Uses small synthetic target (easier: just run with real Qwen3 if HF env set, else skip) — document that metallib-needed tests will mostly skip until we have a proper mock target fixture
- Asserts: after `next()` once, `pendingIndex=1`, `committedLength == promptLen`, `lastTargetHidden.dim(-1) == 5*hiddenSize`

**Acceptance:** compiles, test discovered (skipped if no metallib is fine)

**Commit:** `feat(mlxlmcommon): DFlashIterator skeleton + prefill`

---

## B1.2: One speculation round (pure math, no EOS)

**Files:**
- Modify: `Libraries/MLXLMCommon/DFlash/DFlashIterator.swift` — implement `runOneSpeculationRound()`, invoke from `next()`
- New: `Tests/MLXLMTests/DFlashIteratorSpecRoundTests.swift`

**What:** Implement the full speculation round body EXCEPT EOS handling (deferred to B1.3). Focus on:
1. Block construction with mask tokens
2. Draft forward + draft cache management
3. Target verify forward
4. Greedy longest-prefix match
5. Commit accepted + bonus
6. Cache trimming (both)
7. `lastTargetHidden` update

**Critical math (copy-paste this into comments in the code — it's the spec):**

```
INVARIANTS after every round:
  committedLength = prompt_len + total_emitted_decode_tokens
  target_cache.offset == committedLength
  draft_cache.offset == committedLength - 1    (draft sees accepted tokens except last; see why below)
  lastTargetHidden.shape == [1, acceptance_len + 1, 5*hiddenSize]

OFF-BY-ONE SPEC:
  block[0] = last_committed_token (already in target_cache from prev round)
  block[1..<16] = mask OR draft proposals
  draft forward uses positions [committedLength..<committedLength+16]
  target verify forward uses positions [committedLength..<committedLength+16]
  BUT: target_cache already has block[0] from previous verify's last position.
       So verify runs block as a fresh 16-token forward; we trim blockSize-(acc_len+1) at end.
  posterior[i] = target's prediction for position (committedLength + i + 1) given [..., block[i]]
  block[1..<16][i] = draft_tokens[i]  (15 proposals)
  Greedy match:
    for i in 0..<15:
      if block[i+1] == posterior[i]: accept += 1; else: break
  Commit: block[1..<1+acc_len] + [posterior[acc_len]]  (bonus)
```

**Implementation sketch:**

```swift
private mutating func runOneSpeculationRound() {
    guard let lastHidden = lastTargetHidden else { return }
    let lastCommitted = pendingTokens.last!  // invariant: pendingTokens ends with last committed

    // 1. Build block
    var blockIds: [Int32] = [Int32(lastCommitted)]
    blockIds.append(contentsOf: repeatElement(Int32(maskTokenId), count: blockSize - 1))
    let block = MLXArray(blockIds).reshaped([1, blockSize])

    // 2. Draft forward
    let noiseEmb = target.embedTokens(block)
    let draftHidden = drafter(
        noiseEmbedding: noiseEmb,
        targetHidden: lastHidden,
        caches: draftCache.map { Optional($0) })
    // Take hidden[:, 1..<blockSize, :] because position 0's output is the embed-of-committed-token, not a proposal
    let draftLogits = target.applyLMHead(draftHidden[0..., 1..., 0...])  // [1, 15, vocab]
    let draftTokens = argMax(draftLogits, axis: -1).asType(.int32)       // [1, 15]

    // 3. Reset draft cache to committedLength (draft_cache ran ahead)
    for cache in draftCache {
        let n = cache.offset - committedLength
        if n > 0 { _ = cache.trim(n) }
    }

    // 4. Fill block with draft proposals
    var filled: [Int32] = [Int32(lastCommitted)]
    for i in 0..<(blockSize - 1) {
        filled.append(draftTokens[0, i].item(Int32.self))
    }
    let filledBlock = MLXArray(filled).reshaped([1, blockSize])

    // 5. Target verify
    let (verifyHiddens, verifyLogits) = target.captureHiddenStatesAndLogits(
        inputs: filledBlock,
        layerIndices: targetLayerIds,
        cache: targetCache)
    let posterior = argMax(verifyLogits, axis: -1).asType(.int32)  // [1, 16]

    // 6. Greedy match
    var acceptanceLen = 0
    for i in 0..<(blockSize - 1) {
        let draftTok = filled[i + 1]               // index 1..<16 in block
        let verTok = posterior[0, i].item(Int32.self)
        if draftTok == verTok { acceptanceLen += 1 } else { break }
    }
    totalProposed += blockSize - 1
    totalAccepted += acceptanceLen

    // 7. Commit accepted + bonus
    var newTokens: [Int] = []
    for i in 0..<acceptanceLen {
        newTokens.append(Int(filled[i + 1]))
    }
    let bonus = Int(posterior[0, acceptanceLen].item(Int32.self))
    newTokens.append(bonus)
    pendingTokens = newTokens
    pendingIndex = 0

    // 8. Trim target cache: keep acceptanceLen + 1 of the 16 we ran
    let extra = blockSize - (acceptanceLen + 1)
    if extra > 0 {
        for cache in targetCache { _ = cache.trim(extra) }
    }
    committedLength += acceptanceLen + 1

    // 9. Update lastTargetHidden
    let concatHidden = concatenated(verifyHiddens, axis: -1)
    lastTargetHidden = concatHidden[0..., 0..<(acceptanceLen + 1), 0...]
}
```

Wire into `next()`:
```swift
// After firstStepDone check, if pendingTokens exhausted after emit:
if pendingIndex >= pendingTokens.count {
    runOneSpeculationRound()
    pendingIndex = 0
    if pendingTokens.isEmpty { return nil }
    // then loop back to emit
}
```

Tests (`DFlashIteratorSpecRoundTests`):
- Math sanity with synthetic target that returns all-zero logits → posterior is all argmax=0, draft proposals are also all 0 (since lmHead over zero hidden = zero → argmax 0) → acceptance_len == blockSize-1 (perfect match edge case)
- Acceptance=0 edge: construct scenario where draft != posterior at position 0 → acceptance_len == 0, commit just bonus token
- Gated on dflashMetallibAvailable; mostly runs only when metallib is resolvable. Don't block on this.

**Acceptance:** compiles, math sanity test covers acceptance_len=0 and acceptance_len=blockSize-1

**Commit:** `feat(mlxlmcommon): DFlashIterator speculation round with greedy verify`

---

## B1.3: EOS detection + max_tokens + pendingTokens semantics

**Files:**
- Modify: `Libraries/MLXLMCommon/DFlash/DFlashIterator.swift` — add EOS + max checks
- New: `Tests/MLXLMTests/DFlashIteratorEOSTests.swift`

**What:** Handle two edge cases correctly:

1. **EOS mid-block:** If any token in `newTokens` is in `stopTokenIds`, truncate `newTokens` at that index (inclusive). Subsequent `next()` calls return nil.
2. **max_tokens cutoff:** Already in `next()` header; but must also prevent committing more than needed in the last round. If `emittedCount + newTokens.count > maxTokens!`, truncate.

Also handle: **`lastTargetHidden` sync** — after EOS, we don't need next round, but state should be consistent (committedLength reflects only emitted tokens).

Tests:
- EOS token appears at position 5 of a block-15 accept → `newTokens` has 6 items (5 accepted + EOS bonus) but only first N up to and including EOS emitted
- max_tokens=3 with acceptance that would give 10 tokens → only 3 emitted, rest discarded
- Gated on metallib

**Acceptance:** EOS + max_tokens behave per OpenAI-compatible streaming semantics

**Commit:** `feat(mlxlmcommon): DFlashIterator EOS and max_tokens handling`

---

## B1.4: Lossless parity regression test (HF-gated)

**Files:**
- New: `Tests/MLXLMTests/DFlashIteratorParityTests.swift` (~100 LOC)

**What:** The keystone test — gated on `DFLASH_TEST_HF=1` AND `dflashMetallibAvailable`. Downloads `mlx-community/Qwen3-4B-bf16` + `z-lab/Qwen3-4B-DFlash-b16`, runs baseline AR and DFlash on same prompt, asserts token-for-token equal at temp=0.

```swift
@Test(.enabled(if: dflashHFTestsEnabled && dflashMetallibAvailable))
func testDFlashLosslessAgainstBaselineQwen34B() async throws {
    // 1. Load target via ModelFactory
    let targetContainer = try await LLMModelFactory.shared.loadContainer(
        configuration: .init(id: "mlx-community/Qwen3-4B-bf16"))

    // 2. Load drafter
    let (drafter, dconfig) = try await DFlashWeightLoader.load()

    let prompt = "Solve 2x + 5 = 17 step by step."
    let maxTokens = 48

    // 3. Baseline greedy
    let baselineTokens = try await targetContainer.perform { context in
        let input = try await context.processor.prepare(input: .init(prompt: prompt))
        var cache = context.model.newCache(parameters: .init(temperature: 0))
        let iter = TokenIterator(
            input: input, model: context.model, cache: cache,
            parameters: .init(maxTokens: maxTokens, temperature: 0))
        return iter.prefix(maxTokens).map { $0 }
    }

    // 4. DFlash
    let dflashTokens = try await targetContainer.perform { context in
        guard let target = context.model as? any DFlashTargetModel else {
            Issue.record("Qwen3 target does not conform to DFlashTargetModel")
            return [Int]()
        }
        let promptTokens = try context.tokenizer.encode(
            text: prompt, addSpecialTokens: true)
        let promptArr = MLXArray(promptTokens.map { Int32($0) })
            .reshaped([1, promptTokens.count])
        var iter = try DFlashIterator(
            promptTokens: promptArr, target: target,
            drafter: drafter, draftConfig: dconfig,
            maxTokens: maxTokens)
        var out: [Int] = []
        while let t = iter.next() { out.append(t) }
        return out
    }

    #expect(baselineTokens == dflashTokens,
            "LOSSLESS VIOLATED: DFlash output diverged from baseline AR")

    // Also record acceptance for the record
    // (iter would need to be accessible outside perform; adjust as needed)
}
```

**Acceptance:** when `DFLASH_TEST_HF=1` on a metallib-capable machine, test downloads models and passes (tokens bit-identical). Without env vars, test skips.

**Commit:** `test(dflash): lossless parity regression gate (HF-online)`

---

## Execution order

1. Pre-B (protocol + conformance) — blocks everything else
2. B1.1 (skeleton + prefill)
3. B1.2 (spec round math)
4. B1.3 (EOS/max)
5. B1.4 (parity test — gated)

**Do not skip forward.** Each step's tests gate the next.

---

## After Wave B

Remaining Stage 2 work after B1.4:
- **B2** — `DFlashEngine` wrapping iterator as `InferenceEngine` + register in `EngineRegistry`
- **Wave C** — hardening (acceptance rate plumbing, parity CI gate, bench script, TQ/TA combo matrix, docs)

These stay as written in the parent plan.

---

## Commit policy

One commit per sub-task. Commit message exactly as specified above. No squashing. If a sub-task's build breaks, fix in the same commit before committing.

## Risks specific to this decomposition

| Risk | Mitigation |
|---|---|
| `captureHiddenStatesAndLogits` cache parameter is `[KVCache]?` (optional array) but we want to pass non-optional — API mismatch | Test Pre-B; if needed, add overload |
| `argMax` on MLXArray returns Int32 vs Int — item conversion needed | Plan sketches it; codex must verify |
| `KVCacheSimple` doesn't implement `trim(_:)` — check first | If missing, downgrade to `TrimmableKVCache` or add to protocol |
| `TokenIteratorProtocol` signature — verify matches ours | Grep before writing iterator |
| `target_hidden` concatenation axis — HF uses `dim=-1` (features); our MLX concatenated axis=-1 works | Covered in B1.1 test |
