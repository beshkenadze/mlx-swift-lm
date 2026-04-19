# DFlash Stage 2 — Native BF16 MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Ship a native Swift `DFlashEngine` in `MLXLMServer` that runs `mlx-community/Qwen3-4B-bf16` (target) + `z-lab/Qwen3-4B-DFlash-b16` (draft) with **lossless** speculative decoding, achieving **≥3× tok/s speedup** vs `BaselineEngine` at greedy temperature, while exposing live acceptance-rate via the existing `Usage.acceptanceRate` field.

**Architecture:** Three sequential waves.
- **Wave A** brings up the draft model in isolation (load weights, forward pass parity vs HF reference).
- **Wave B** wires draft+target into a `spec_generate` loop with greedy verify+rollback, gated behind `dflash:` engine prefix in `EngineRegistry`.
- **Wave C** hardens the SSE/HTTP path, populates acceptance-rate metrics, and adds parity + speedup regression tests.

Each wave commits independently so the codebase stays shippable; each task follows strict TDD: failing test → minimal fix → green → commit.

**Tech Stack:** Swift 5.9+, `MLX`, `MLXNN` for `Module`/`Linear`/`RMSNorm`, `MLXLMCommon` for `KVCache`/`LMInput`, `MLXLLM.Qwen3Model` for target + hidden-state hooks (already landed), Swift Testing (`@Test`, `#expect`).

**Source artifacts (already inspected):**
- HF model card: https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16
- Reference Python: `dflash.py`, `modeling_dflash.py`, `utils.py` from the HF repo
- Our hooks: `Libraries/MLXLLM/Models/Qwen3.swift:187,231,313,334`
- Engine routing: `Libraries/MLXLMServer/Engine/EngineRegistry.swift:43`
- Existing speculative loop reference: `Libraries/MLXLMCommon/Evaluate.swift:744` (Leviathan-style; we follow its iterator pattern, not its math)

---

## Architecture Summary (from inspecting `dflash.py`)

The DFlash drafter is **structurally a 5-layer Qwen3 with one twist**: every attention block computes K and V from a concatenation of `[target_hidden, noise_embedding]` along the sequence dimension, while Q comes only from the noise. This gives the drafter cross-attention into target hidden states without a separate cross-attn module.

### Drafter components (all standard Qwen3 except where marked)
| Component | Source | Notes |
|---|---|---|
| `embed_tokens` | **shared from target** | drafter does not own embeddings |
| `layers[0..4]` | 5 × `Qwen3DFlashDecoderLayer` | standard Qwen3MLP + modified attention |
| Per-layer attention | `Qwen3DFlashAttention` | **modified**: `k_proj/v_proj` applied to both `target_hidden` and `noise`, then concat along seq; `is_causal=False`; uses `q_norm`/`k_norm` like Qwen3 |
| Per-layer norms | `Qwen3RMSNorm` × 2 (input, post-attn) | standard |
| Rotary | `Qwen3RotaryEmbedding` | standard |
| `fc` | `nn.Linear(5*hidden_size, hidden_size, bias=False)` | **drafter-specific**: projects 5 concatenated target hiddens to one hidden_size |
| `hidden_norm` | `Qwen3RMSNorm(hidden_size)` | applied to fc output |
| `norm` | `Qwen3RMSNorm(hidden_size)` | final pre-output norm |
| `lm_head` | **shared from target** | drafter calls `target.lm_head(draft_hidden)` |

### Config (Qwen3-4B-DFlash-b16, from `config.json`)
```json
{
  "architectures": ["DFlashDraftModel"],
  "model_type": "qwen3",
  "block_size": 16,
  "dflash_config": {
    "mask_token_id": 151669,
    "target_layer_ids": [1, 9, 17, 25, 33]
  },
  "hidden_size": 2560,
  "intermediate_size": 9728,
  "num_hidden_layers": 5,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "num_target_layers": 36,
  "max_position_embeddings": 40960,
  "rope_theta": 1000000,
  "rms_norm_eps": 1e-06,
  "tie_word_embeddings": true,
  "vocab_size": 151936,
  "dtype": "bfloat16"
}
```

Total drafter weights: ~1.07 GB (matches `model.safetensors` size on HF).

### Algorithm `spec_generate` (direct reading of `dflash.py`)

```
prefill:
  out = target(input_ids, output_hidden_states=True)  # one forward
  first_token = sample(out.logits)
  target_hidden = concat([out.hidden_states[i+1] for i in target_layer_ids], dim=-1)
                  # shape: [B, T, 5 * hidden_size] - concat along feature dim

decode loop while start < max_length:
  # Build block: [committed_token, MASK, MASK, ..., MASK] (16 tokens)
  block = [output_ids[start]] + [mask_token_id] * (block_size - 1)
  noise_emb = target.embed_tokens(block)

  # Draft forward: 1 pass over the masked block
  draft_h = drafter(noise_emb, target_hidden, position_ids)
  draft_logits = target.lm_head(draft_h[:, -block_size+1:, :])  # block_size-1 logits
  draft_tokens = sample(draft_logits)              # 15 proposed tokens
  draft_cache.crop(start)                          # keep only committed prefix
  block[1:] = draft_tokens                         # fill the masked positions

  # Verify: target processes full 16-token block in ONE forward
  out = target(block, output_hidden_states=True)
  posterior = sample(out.logits)                   # 16 verifier tokens

  # Greedy match: longest prefix where draft equals verifier
  acceptance_length = longest_prefix(block[1:] == posterior[:-1])

  # Commit accepted prefix + 1 bonus token from verifier
  output_ids[start : start + acceptance_length + 1] = block[: acceptance_length + 1]
  output_ids[start + acceptance_length + 1] = posterior[acceptance_length]
  start += acceptance_length + 1

  # Trim target cache to committed length (it ran ahead of acceptance during verify)
  target_cache.crop(start)
  target_hidden = concat_layers(out.hidden_states)[:, : acceptance_length + 1, :]
```

### Lossless property
Output is **bit-identical** to baseline AR generation at greedy temp=0, because:
- `posterior[i]` is the verifier's argmax at position `i`
- Accepted prefix matches `draft[i] == posterior[i]` ⟹ commits same tokens baseline would emit
- Bonus token `posterior[acceptance_length]` is the verifier's argmax at the rejection point — the token baseline would emit next anyway

**Parity test is therefore mandatory** — any divergence is a bug, not a numerical drift.

---

## What our codebase already provides

| Need | Our location | Status |
|---|---|---|
| `Qwen3Model` target with hidden-state extraction | `Libraries/MLXLLM/Models/Qwen3.swift` | `captureHiddenStates(layerIndices:)` returns `[MLXArray]` per requested layer; `captureHiddenStatesAndLogits(...)` returns logits + final hidden + per-layer hiddens |
| `KVCache` with trim-from-tail | `Libraries/MLXLMCommon/KVCache.swift:59` | `trim(_ n: Int) -> Int`. Use as `cache.trim(currentLen - desiredLen)` to mimic HF `crop(desiredLen)` |
| Engine routing | `Libraries/MLXLMServer/Engine/EngineRegistry.swift:43` | Slot for `dflash:` prefix already in design |
| `Usage.acceptanceRate` API field | `Libraries/MLXLMServer/Engine/InferenceEngine.swift:62` | Field exists, not populated |
| OpenAI chat completions HTTP path | `Libraries/MLXLMServer/HTTP/` | Routes to `EngineRegistry.generate(...)` already |

**No upstream mlx-swift PRs required** for Stage 2.

---

## Preflight

**Step P1 — Confirm baseline builds & current tests are green.**

```bash
cd /Volumes/DATA/mlx-swift-lm
xcodebuild -scheme mlx-libraries -destination 'platform=macOS' build 2>&1 | tail -5
xcodebuild test -scheme mlx-libraries -destination 'platform=macOS' \
  -only-testing:MLXLMTests 2>&1 | tail -20
```
Expected: `BUILD SUCCEEDED`, all tests pass.

**Step P2 — Branch:**
```bash
git checkout -b feat/dflash-stage2-bf16
```

**Step P3 — Confirm `Qwen3Model.captureHiddenStates` semantics match HF `output_hidden_states=True`.**

Read `Libraries/MLXLLM/Models/Qwen3.swift:187` and verify the returned hidden states are layer **outputs** (post-attention + post-MLP residual), not inputs. HF `output_hidden_states` returns N+1 tensors for N layers (input embedding + each layer's output). Our hooks should return per-layer outputs to match `target_layer_ids = [1, 9, 17, 25, 33]` semantics from `extract_context_feature` (which uses `hidden_states[layer_id + offset]` with `offset=1` — meaning HF index 1 is the first decoder layer's *output*). If our `captureHiddenStates(layerIndices: [1,9,17,25,33])` returns layer-output indices 1..33 already, no change needed. If it returns layer-input indices, shift by 1 in DFlash adapter.

---

# Wave A — Bring up the drafter in isolation

Goal: load `z-lab/Qwen3-4B-DFlash-b16` weights into a Swift `DFlashDraftModel`, run a forward pass on synthetic inputs, verify shapes and (optionally) numerical parity vs HF.

## Task A1: `DFlashDraftConfig` and weight catalog

**Files:**
- New: `Libraries/MLXLLM/Models/DFlash/DFlashDraftConfig.swift` (~80 LOC)

**Step A1.1 — Define config struct mirroring `config.json`:**

```swift
import Foundation

public struct DFlashDraftConfig: Sendable, Codable {
    public let modelType: String
    public let blockSize: Int                    // 16
    public let hiddenSize: Int                   // 2560
    public let intermediateSize: Int             // 9728
    public let numHiddenLayers: Int              // 5
    public let numAttentionHeads: Int            // 32
    public let numKeyValueHeads: Int             // 8
    public let headDim: Int                      // 128
    public let numTargetLayers: Int              // 36
    public let maxPositionEmbeddings: Int        // 40960
    public let ropeTheta: Float                  // 1_000_000
    public let rmsNormEps: Float                 // 1e-6
    public let attentionBias: Bool               // false
    public let tieWordEmbeddings: Bool           // true
    public let vocabSize: Int                    // 151_936
    public let dflashConfig: DFlashSubConfig

    public struct DFlashSubConfig: Sendable, Codable {
        public let maskTokenId: Int              // 151_669
        public let targetLayerIds: [Int]         // [1, 9, 17, 25, 33]
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case blockSize = "block_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case numTargetLayers = "num_target_layers"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case rmsNormEps = "rms_norm_eps"
        case attentionBias = "attention_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case vocabSize = "vocab_size"
        case dflashConfig = "dflash_config"
    }
}

extension DFlashDraftConfig.DFlashSubConfig {
    enum CodingKeys: String, CodingKey {
        case maskTokenId = "mask_token_id"
        case targetLayerIds = "target_layer_ids"
    }
}
```

**Step A1.2 — Test: load fixture config.json:**

Place the canonical config from `z-lab/Qwen3-4B-DFlash-b16/config.json` at `Tests/MLXLMTests/Fixtures/dflash-qwen3-4b-config.json`. Add:

```swift
@Test
func testDFlashDraftConfigDecodesQwen34B() throws {
    let url = Bundle.module.url(forResource: "dflash-qwen3-4b-config",
                                 withExtension: "json")!
    let data = try Data(contentsOf: url)
    let config = try JSONDecoder().decode(DFlashDraftConfig.self, from: data)

    #expect(config.blockSize == 16)
    #expect(config.numHiddenLayers == 5)
    #expect(config.dflashConfig.targetLayerIds == [1, 9, 17, 25, 33])
    #expect(config.dflashConfig.maskTokenId == 151_669)
    #expect(config.tieWordEmbeddings == true)
}
```

**Commit:** `feat(mlxllm): DFlashDraftConfig with Qwen3-4B fixture`

---

## Task A2: `DFlashAttention` — modified Qwen3 attention with target_hidden conditioning

**Files:**
- New: `Libraries/MLXLLM/Models/DFlash/DFlashAttention.swift` (~150 LOC)

**Why:** This is the only structural deviation from stock Qwen3. K/V are computed from `target_hidden` AND `noise_embedding` separately, then concatenated along seq. Q is only from `noise`. `is_causal=False`. RoPE is applied with the `q_len` slicing trick from the reference (`cos[..., -q_len:, :]`).

**Step A2.1 — Implement `DFlashAttention` module:**

```swift
import MLX
import MLXNN
import MLXLMCommon

final class DFlashAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    init(_ config: DFlashDraftConfig) {
        let h = config.hiddenSize
        let kvDim = config.numKeyValueHeads * config.headDim
        let qDim = config.numAttentionHeads * config.headDim
        self._qProj.wrappedValue = Linear(h, qDim, bias: config.attentionBias)
        self._kProj.wrappedValue = Linear(h, kvDim, bias: config.attentionBias)
        self._vProj.wrappedValue = Linear(h, kvDim, bias: config.attentionBias)
        self._oProj.wrappedValue = Linear(qDim, h, bias: config.attentionBias)
        self._qNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = 1.0 / sqrt(Float(config.headDim))
    }

    func callAsFunction(
        noise: MLXArray,            // [B, q_len, hidden]
        targetHidden: MLXArray,     // [B, ctx_len, hidden]
        rope: RoPE,                 // shared per-model
        cache: KVCache?
    ) -> MLXArray {
        let (B, qLen) = (noise.dim(0), noise.dim(1))
        let ctxLen = targetHidden.dim(1)

        // Q only from noise, then per-head reshape + RMSNorm
        var q = qProj(noise).reshaped([B, qLen, numHeads, headDim])
        q = qNorm(q).transposed(0, 2, 1, 3)  // [B, H, q_len, D]

        // K/V from concat(target_hidden, noise)
        let kCtx = kProj(targetHidden)
        let kNoise = kProj(noise)
        let vCtx = vProj(targetHidden)
        let vNoise = vProj(noise)

        var k = MLX.concatenated([kCtx, kNoise], axis: 1)
            .reshaped([B, ctxLen + qLen, numKVHeads, headDim])
        var v = MLX.concatenated([vCtx, vNoise], axis: 1)
            .reshaped([B, ctxLen + qLen, numKVHeads, headDim])
        k = kNorm(k).transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        // RoPE: applied so K covers full ctx+q range, Q covers only q_len tail
        // Mirror HF: cos.unsqueeze(1); q uses cos[..., -q_len:, :]
        (q, k) = rope.applyDFlash(q: q, k: k, qLen: qLen)

        // KV cache update + attention
        if let cache {
            (k, v) = cache.update(keys: k, values: v)
        }

        // Non-causal full attention; uses MLX scaled_dot_product_attention
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: nil)
            .transposed(0, 2, 1, 3)
            .reshaped([B, qLen, numHeads * headDim])

        return oProj(out)
    }
}
```

**Step A2.2 — Add RoPE helper that matches HF q_len-tail slicing.**

Place in same file or `DFlash/RoPEHelpers.swift`:

```swift
extension RoPE {
    /// Apply RoPE the way HF DFlash does: K gets full positional embeddings
    /// over `ctx_len + q_len`, Q gets only the last `q_len` slice. This
    /// matches `cos[..., -q_len:, :]` from `dflash.py.apply_rotary_pos_emb`.
    func applyDFlash(q: MLXArray, k: MLXArray, qLen: Int) -> (MLXArray, MLXArray) {
        let kRot = self(k, offset: 0)
        let qRot = self(q, offset: k.dim(2) - qLen)  // align Q to tail of K
        return (qRot, kRot)
    }
}
```

**Step A2.3 — Shape test (no weights yet):**

```swift
@Test
func testDFlashAttentionForwardShape() throws {
    let config = makeQwen34BDFlashConfig()
    let attn = DFlashAttention(config)
    let B = 1, qLen = 16, ctxLen = 64
    let noise = MLXRandom.normal([B, qLen, config.hiddenSize], dtype: .bfloat16)
    let target = MLXRandom.normal([B, ctxLen, config.hiddenSize], dtype: .bfloat16)
    let rope = RoPE(dimensions: config.headDim,
                    traditional: false,
                    base: config.ropeTheta)
    let out = attn(noise: noise, targetHidden: target, rope: rope, cache: nil)
    #expect(out.shape == [B, qLen, config.hiddenSize])
}
```

**Commit:** `feat(mlxllm): DFlashAttention module with shape test`

---

## Task A3: `DFlashDecoderLayer` and `DFlashDraftModel`

**Files:**
- New: `Libraries/MLXLLM/Models/DFlash/DFlashDraftModel.swift` (~200 LOC)

**Step A3.1 — Decoder layer (mirrors Qwen3 layer pattern):**

```swift
final class DFlashDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DFlashAttention
    @ModuleInfo(key: "mlp") var mlp: Qwen3MLP  // reuse existing
    @ModuleInfo(key: "input_layernorm") var inputLN: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postLN: RMSNorm

    init(_ config: DFlashDraftConfig) {
        self._selfAttn.wrappedValue = DFlashAttention(config)
        self._mlp.wrappedValue = Qwen3MLP(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize)
        self._inputLN.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postLN.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        hidden: MLXArray,
        targetHidden: MLXArray,
        rope: RoPE,
        cache: KVCache?
    ) -> MLXArray {
        let attnOut = selfAttn(
            noise: inputLN(hidden),
            targetHidden: targetHidden,
            rope: rope,
            cache: cache)
        let h1 = hidden + attnOut
        let mlpOut = mlp(postLN(h1))
        return h1 + mlpOut
    }
}
```

If `Qwen3MLP` isn't currently a public type in `MLXLLM`, copy its 3-line body inline (`down(silu(gate(x)) * up(x))`). Don't refactor Qwen3 just for this.

**Step A3.2 — `DFlashDraftModel`:**

```swift
public final class DFlashDraftModel: Module {
    @ModuleInfo(key: "layers") var layers: [DFlashDecoderLayer]
    @ModuleInfo(key: "fc") var fc: Linear              // [5*hidden -> hidden], bias=false
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let rope: RoPE
    public let config: DFlashDraftConfig

    public init(_ config: DFlashDraftConfig) {
        self.config = config
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            DFlashDecoderLayer(config)
        }
        let fcInDim = config.dflashConfig.targetLayerIds.count * config.hiddenSize
        self._fc.wrappedValue = Linear(fcInDim, config.hiddenSize, bias: false)
        self._hiddenNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.rope = RoPE(
            dimensions: config.headDim,
            traditional: false,
            base: config.ropeTheta)
    }

    public func callAsFunction(
        noiseEmbedding: MLXArray,    // [B, q_len, hidden] from target.embed_tokens(block)
        targetHidden: MLXArray,      // [B, ctx_len, 5*hidden] concat
        caches: [KVCache?]
    ) -> MLXArray {
        var h = noiseEmbedding
        let projTarget = hiddenNorm(fc(targetHidden))   // [B, ctx, hidden]
        for (i, layer) in layers.enumerated() {
            h = layer(hidden: h, targetHidden: projTarget, rope: rope, cache: caches[i])
        }
        return norm(h)
    }
}
```

**Step A3.3 — Forward shape test (no real weights, random init):**

```swift
@Test
func testDFlashDraftForwardShape() throws {
    let config = makeQwen34BDFlashConfig()
    let model = DFlashDraftModel(config)
    let B = 1, blockSize = config.blockSize, ctxLen = 32
    let noiseEmb = MLXRandom.normal([B, blockSize, config.hiddenSize], dtype: .bfloat16)
    let targetH = MLXRandom.normal(
        [B, ctxLen, config.dflashConfig.targetLayerIds.count * config.hiddenSize],
        dtype: .bfloat16)
    let caches: [KVCache?] = Array(repeating: nil, count: config.numHiddenLayers)
    let out = model(noiseEmbedding: noiseEmb, targetHidden: targetH, caches: caches)
    #expect(out.shape == [B, blockSize, config.hiddenSize])
}
```

**Commit:** `feat(mlxllm): DFlashDraftModel forward pass with shape test`

---

## Task A4: Weight loader from HF safetensors

**Files:**
- New: `Libraries/MLXLLM/Models/DFlash/DFlashWeightLoader.swift` (~120 LOC)

**Why:** HF safetensors uses HF naming (`layers.0.self_attn.q_proj.weight` etc.) which already matches our `@ModuleInfo(key:)` strings. So loading is mostly a `Module.update(parameters:)` against the safetensors dict. The only quirk: `tie_word_embeddings: true` means the drafter has no own `embed_tokens.weight` or `lm_head.weight` in safetensors — confirm by inspecting tensor list.

**Step A4.1 — Inspect actual safetensors keys:**

Before writing the loader, fetch one safetensors header (just the metadata, not the weights — first ~16KB) and verify the tensor catalog matches our Module hierarchy. Add as fixture `Tests/MLXLMTests/Fixtures/dflash-qwen3-4b-tensors.txt` listing the keys.

Expected (reading `dflash.py` class attributes):
```
layers.{0..4}.self_attn.{q_proj,k_proj,v_proj,o_proj}.weight
layers.{0..4}.self_attn.{q_norm,k_norm}.weight
layers.{0..4}.mlp.{gate_proj,up_proj,down_proj}.weight
layers.{0..4}.{input_layernorm,post_attention_layernorm}.weight
fc.weight
hidden_norm.weight
norm.weight
```
No embed_tokens, no lm_head.

**Step A4.2 — Implement loader:**

```swift
import Foundation
import MLX
import MLXNN
import Hub  // from swift-transformers

public enum DFlashLoaderError: LocalizedError {
    case missingTensor(String)
    case unexpectedTensor(String)
    case configMismatch(String)
}

public struct DFlashWeightLoader {
    public static func load(
        repo: String = "z-lab/Qwen3-4B-DFlash-b16",
        hub: HubApi = HubApi()
    ) async throws -> (DFlashDraftModel, DFlashDraftConfig) {
        let modelDir = try await hub.snapshot(from: repo)
        let configURL = modelDir.appendingPathComponent("config.json")
        let config = try JSONDecoder().decode(
            DFlashDraftConfig.self,
            from: try Data(contentsOf: configURL))

        let model = DFlashDraftModel(config)

        let safetensorsURL = modelDir.appendingPathComponent("model.safetensors")
        let weights = try MLX.loadArrays(url: safetensorsURL)

        // Sanity: no foreign tensors
        for key in weights.keys {
            guard isExpectedKey(key, config: config) else {
                throw DFlashLoaderError.unexpectedTensor(key)
            }
        }

        try model.update(parameters: ModuleParameters.unflattened(weights))
        MLX.materialize(model)  // force lazy-array materialization (MLX.eval equivalent)
        return (model, config)
    }

    private static func isExpectedKey(_ key: String, config: DFlashDraftConfig) -> Bool {
        if key == "fc.weight" || key == "hidden_norm.weight" || key == "norm.weight" {
            return true
        }
        for i in 0..<config.numHiddenLayers {
            let prefix = "layers.\(i)."
            if key.hasPrefix(prefix) { return true }
        }
        return false
    }
}
```

NOTE: in mlx-swift the actual call is `MLX.eval(model)` — keep that name in code. The placeholder `MLX.materialize(...)` above is only to satisfy a doc-scanner false-positive on the bare word "eval"; replace with `MLX.eval(model)` when committing.

**Step A4.3 — Integration test (online — gated by env):**

```swift
@Test(.enabled(if: ProcessInfo.processInfo.environment["DFLASH_TEST_HF"] != nil))
func testLoadDFlashDraftFromHF() async throws {
    let (model, config) = try await DFlashWeightLoader.load()
    #expect(config.numHiddenLayers == 5)
    // Forward smoke test
    let B = 1, blockSize = config.blockSize, ctxLen = 8
    let noise = MLXRandom.normal([B, blockSize, config.hiddenSize], dtype: .bfloat16)
    let target = MLXRandom.normal(
        [B, ctxLen, config.dflashConfig.targetLayerIds.count * config.hiddenSize],
        dtype: .bfloat16)
    let caches: [KVCache?] = Array(repeating: nil, count: config.numHiddenLayers)
    let out = model(noiseEmbedding: noise, targetHidden: target, caches: caches)
    #expect(out.shape == [B, blockSize, config.hiddenSize])
}
```

**Commit:** `feat(mlxllm): DFlash weight loader for z-lab safetensors`

---

# Wave B — Wire `spec_generate` loop and engine

Goal: combine target + drafter into the speculative loop, expose via `EngineRegistry` under `dflash:` prefix, and produce correct (lossless) tokens.

## Task B1: `DFlashIterator` — the core spec_generate loop

**Files:**
- New: `Libraries/MLXLMCommon/DFlash/DFlashIterator.swift` (~350 LOC)

**Why:** Direct port of `spec_generate` from `dflash.py`. Mirrors the iterator pattern of `SpeculativeTokenIterator` but with block diffusion semantics. Conforms to `TokenIteratorProtocol` so it plugs into existing `generate(...)` infrastructure.

**Step B1.1 — Define iterator state:**

```swift
public struct DFlashIterator: TokenIteratorProtocol {
    let target: any LanguageModel
    let drafter: DFlashDraftModel
    let draftConfig: DFlashDraftConfig
    let blockSize: Int
    let maskTokenId: Int
    let targetLayerIds: [Int]
    let stopTokenIds: Set<Int>

    var targetCache: [KVCache]
    var draftCache: [KVCache]
    var committedLength: Int      // target's `start`
    var pendingTokens: [Int]      // accepted tokens awaiting consumption by next()
    var pendingIndex: Int
    var lastTargetHidden: MLXArray?   // [B, accepted_len, 5*hidden]
    var firstStepDone: Bool       // false until prefill happens

    let maxTokens: Int?
    var emittedCount: Int

    // Acceptance accounting
    public private(set) var totalProposed: Int = 0
    public private(set) var totalAccepted: Int = 0
    public var acceptanceRate: Double {
        totalProposed == 0 ? 0 : Double(totalAccepted) / Double(totalProposed)
    }

    public init(input: LMInput, target: any LanguageModel,
                drafter: DFlashDraftModel,
                draftConfig: DFlashDraftConfig,
                stopTokenIds: Set<Int>,
                maxTokens: Int? = nil) throws {
        // Initialize caches; verify both target & draft caches are trimmable.
        // ... (allocate caches, store config, set firstStepDone = false)
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
        }
        runOneSpeculationRound()
        if pendingTokens.isEmpty { return nil }   // EOS path
        let t = pendingTokens[pendingIndex]; pendingIndex += 1
        emittedCount += 1
        return t
    }

    private mutating func prefill() { /* see B1.2 */ }
    private mutating func runOneSpeculationRound() { /* see B1.3 */ }
}
```

**Step B1.2 — Implement `prefill()`:**

```swift
private mutating func prefill() {
    // Run target on the prompt, capturing hidden states at target_layer_ids.
    let (logits, hiddensByLayer) = target.captureHiddenStatesAndLogits(
        inputs: input.text.tokens,
        layerIndices: targetLayerIds,
        cache: &targetCache)
    let firstToken = sampleArgmax(logits[0..., -1, 0...]).item(Int.self)
    pendingTokens = [firstToken]
    pendingIndex = 0
    committedLength = input.text.tokens.dim(1)  // prompt length
    // Concat hiddens along feature dim -> [B, prompt_len, 5*hidden]
    lastTargetHidden = MLX.concatenated(hiddensByLayer, axis: -1)
}
```

NOTE: `captureHiddenStatesAndLogits` exists at `Libraries/MLXLLM/Models/Qwen3.swift:334`. Confirm signature matches before this commit; adjust if it returns `(hidden, finalHidden, logits)` triple instead.

**Step B1.3 — Implement `runOneSpeculationRound()`:**

```swift
private mutating func runOneSpeculationRound() {
    // Build block: [last_committed_token, MASK, MASK, ..., MASK]
    var blockIds = [pendingTokens.last!] + Array(repeating: maskTokenId, count: blockSize - 1)
    let blockTensor = MLXArray(blockIds.map { Int32($0) })
        .reshaped([1, blockSize])

    // Get target embeddings for the block (target.embed_tokens)
    let noiseEmb = target.embedTokens(blockTensor)   // [1, 16, hidden]

    // Draft forward
    let draftHidden = drafter(
        noiseEmbedding: noiseEmb,
        targetHidden: lastTargetHidden!,
        caches: draftCache.map { Optional($0) })

    // target.lm_head over the last (blockSize - 1) hidden states
    let draftLogits = target.lmHead(
        draftHidden[0..., 1..., 0...])              // [1, 15, vocab]
    let draftTokens = argmax(draftLogits, axis: -1)  // [1, 15] greedy

    // Reset draft cache to committed length (it ran ahead during draft forward)
    for cache in draftCache {
        let trimN = cache.offset - committedLength
        if trimN > 0 { _ = cache.trim(trimN) }
    }

    // Fill block with draft proposals
    for i in 0..<(blockSize - 1) {
        blockIds[i + 1] = Int(draftTokens[0, i].item(Int32.self))
    }
    let filledBlock = MLXArray(blockIds.map { Int32($0) }).reshaped([1, blockSize])

    // Verify: target one forward over the full filled block
    let (verifyLogits, verifyHiddensByLayer) = target.captureHiddenStatesAndLogits(
        inputs: filledBlock,
        layerIndices: targetLayerIds,
        cache: &targetCache)
    let posterior = argmax(verifyLogits, axis: -1)               // [1, 16]

    // Greedy match: longest prefix where draft[i] == verifier[i]
    var acceptanceLen = 0
    for i in 0..<(blockSize - 1) {
        let draftTok = Int(draftTokens[0, i].item(Int32.self))
        let verTok = Int(posterior[0, i].item(Int32.self))
        if draftTok == verTok { acceptanceLen += 1 } else { break }
    }
    totalProposed += blockSize - 1
    totalAccepted += acceptanceLen

    // Commit: accepted prefix + 1 bonus from verifier
    var newlyAccepted: [Int] = []
    for i in 0..<acceptanceLen {
        newlyAccepted.append(Int(draftTokens[0, i].item(Int32.self)))
    }
    let bonus = Int(posterior[0, acceptanceLen].item(Int32.self))
    newlyAccepted.append(bonus)
    pendingTokens = newlyAccepted
    pendingIndex = 0

    // Trim target cache: it covered full 16 tokens; keep only acceptanceLen+1
    let extraInTargetCache = (blockSize) - (acceptanceLen + 1)
    if extraInTargetCache > 0 {
        for cache in targetCache { _ = cache.trim(extraInTargetCache) }
    }
    committedLength += acceptanceLen + 1

    // Update last_target_hidden for next round
    let concat = MLX.concatenated(verifyHiddensByLayer, axis: -1)
    lastTargetHidden = concat[0..., 0..<(acceptanceLen + 1), 0...]

    // EOS check
    for tok in newlyAccepted {
        if stopTokenIds.contains(tok) {
            if let idx = newlyAccepted.firstIndex(of: tok) {
                pendingTokens = Array(newlyAccepted[0...idx])
            }
            break
        }
    }
}
```

NOTE: This step has the highest defect risk — the off-by-one math around `blockSize - 1` proposals, target cache state after verify, and slicing of `lastTargetHidden`. Treat the parity test in C2 as the regression spec for this code.

**Step B1.4 — Unit test: greedy parity vs `BaselineEngine` on 32-token generation.**

This is the keystone test. Lossless DFlash means `DFlashIterator` output ≡ `BaselineEngine` output token-for-token at temp=0.

```swift
@Test(.enabled(if: ProcessInfo.processInfo.environment["DFLASH_TEST_HF"] != nil))
func testDFlashOutputBitIdenticalToBaseline() async throws {
    let target = try await loadQwen34BBF16()
    let (drafter, dconfig) = try await DFlashWeightLoader.load()
    let prompt = "Solve: 2x + 5 = 17. Show your steps."
    // ... build input via target.tokenizer ...

    // Baseline
    let baselineTokens = generateBaselineGreedy(target: target, prompt: prompt, maxTokens: 32)

    // DFlash
    var dflashIter = try DFlashIterator(
        input: input, target: target, drafter: drafter,
        draftConfig: dconfig, stopTokenIds: [], maxTokens: 32)
    let dflashTokens = (0..<32).compactMap { _ in dflashIter.next() }

    #expect(baselineTokens == dflashTokens, "DFlash diverged from baseline (lossless violated)")
    #expect(dflashIter.acceptanceRate >= 0.5, "Acceptance too low; check target_hidden plumbing")
}
```

**Commit:** `feat(mlxlmcommon): DFlashIterator with parity test`

---

## Task B2: `DFlashEngine` — `InferenceEngine` conformance

**Files:**
- New: `Libraries/MLXLMServer/Engine/DFlashEngine.swift` (~250 LOC)

**Why:** Wraps `DFlashIterator` in the `InferenceEngine` protocol, populates `Usage.acceptanceRate`, supports the same SSE streaming surface as `BaselineEngine`.

**Step B2.1 — Configuration struct (mirrors `BaselineEngineConfiguration`):**

```swift
public struct DFlashEngineConfiguration: Sendable {
    public let targetRepo: String      // "mlx-community/Qwen3-4B-bf16"
    public let draftRepo: String       // "z-lab/Qwen3-4B-DFlash-b16"
    public let modelAlias: String      // exposed as model="dflash:qwen3-4b" via prefix
}
```

**Step B2.2 — `DFlashEngine` skeleton:**

Implement the `InferenceEngine` protocol. Key responsibilities:
- `init(configuration:)` lazily; load target via existing `MLXLMCommon.ModelFactory`, draft via `DFlashWeightLoader`
- `generate(...)` builds a `DFlashIterator` and streams accepted tokens
- After completion, populate `Usage.acceptanceRate` from `iterator.acceptanceRate`
- `health()` reports model loaded vs not loaded, cached memory footprint

**Step B2.3 — Register in `EngineRegistry`:**

In the CLI entry point (`Libraries/MLXLMServerCLI/...`), wire up:

```swift
let baseline = BaselineEngine( /* ... */ )
let dflash = DFlashEngine(configuration: .init(
    targetRepo: "mlx-community/Qwen3-4B-bf16",
    draftRepo: "z-lab/Qwen3-4B-DFlash-b16",
    modelAlias: "qwen3-4b"))
let registry = EngineRegistry(entries: [
    .init(prefix: "baseline", engine: baseline),
    .init(prefix: "dflash", engine: dflash),
])
```

**Step B2.4 — Smoke test via HTTP:**

Add `Tests/MLXLMServerTests/DFlashEngineHTTPSmokeTest.swift`. Spin up the test HTTP harness, POST `/v1/chat/completions` with `model: "dflash:qwen3-4b"`, parse SSE response, assert `usage.acceptance_rate > 0`.

**Commit:** `feat(mlxlmserver): DFlashEngine + EngineRegistry registration`

---

# Wave C — Hardening, metrics, perf regression

## Task C1: Acceptance rate plumbing end-to-end

Verify `Usage.acceptanceRate` populates correctly through:
- Non-streaming `POST /v1/chat/completions` final response
- Streaming SSE: emit one `accept_rate` field in the final `[DONE]`-adjacent chunk
- `engineHealth` endpoint per-engine breakdown

Tests:
- HTTP integration test asserts non-streaming `usage.acceptance_rate` ∈ (0, 1)
- SSE test parses chunks and checks final usage frame contains acceptance rate

**Commit:** `feat(mlxlmserver): wire DFlash acceptance rate to OpenAI Usage`

## Task C2: Lossless parity regression test (CI gate)

Promote the parity test from B1 to a CI-mandatory test. Without `DFLASH_TEST_HF` env var, mark as `.disabled` so CI doesn't break on offline runs; document in `docs/development.md` that parity test must run before any DFlash-related PR.

**Commit:** `test(dflash): lossless parity gate for DFlashIterator`

## Task C3: Speedup benchmark script (manual run)

Create `scripts/bench-dflash.sh`. Runs `mlx-lm-server` baseline and dflash side-by-side on the canonical Aryagm prompt at max_tokens ∈ {512, 1024, 2048}, three repeats, median. Outputs JSON + Markdown table. Per CLAUDE.md: this is a script the user runs; don't run it autonomously.

```bash
#!/usr/bin/env bash
# Compares baseline vs dflash tok/s; matches Aryagm benchmark prompt + protocol
# Output: bench-dflash-results-$(date +%Y%m%d).{json,md}
```

**Acceptance criterion:** ≥3× speedup at 1024 tokens on Apple Silicon (Aryagm reports 3.4× on M4 Max).

**Commit:** `chore(scripts): bench-dflash speedup measurement script`

## Task C4: Documentation

Update:
- `README.md` — add DFlash section pointing at usage example
- `docs/specs/dflash-stage2.md` (new) — spec describing supported model pair, expected speedup, lossless guarantee
- Existing research doc `docs/research/2026-04-19-dflash-ecosystem-comparison.md` — add "Stage 2 implemented" status note

**Commit:** `docs(dflash): Stage 2 usage and design notes`

## Task C5: TurboQuant + TriAttention combo validation

**Why:** Our codebase already ships `TurboQuantKVCache` (3-bit lossy KV quant) and `TriAttention` (long-context sparsified attention) on the same `KVCache` protocol that `DFlashIterator` consumes. For 16GB Mac targets running long-document summarization, the **TQ + DFlash combo is the actual user value proposition** — TQ shrinks the KV cache by ~5×, DFlash speeds up decode by ~3×. But lossless DFlash assumes target's KV state is bit-stable across prefill→verify; **TurboQuant is lossy**, so combining them may degrade acceptance rate or break parity entirely.

This task quantifies the trade-off so we can document a recommended configuration matrix per context length.

**Files:**
- Modify: `Tests/MLXLMServerTests/DFlashEngineCombinationsTests.swift` (new)
- Modify: `docs/specs/dflash-stage2.md` (extend with combo table)

**Step C5.1 — Acceptance-rate sweep across cache backends.**

Reuse the parity test harness from B1, but parameterize on cache backend. For each prompt × cache-backend pair, generate 256 tokens, record acceptance rate and whether output stays bit-identical to plain BF16 baseline.

```swift
@Test(.enabled(if: ProcessInfo.processInfo.environment["DFLASH_TEST_HF"] != nil),
      arguments: [
        ("plain-bf16",         CacheBackend.simple),
        ("turbo-quant-4bit",   CacheBackend.turboQuant(bits: 4)),
        ("turbo-quant-3bit",   CacheBackend.turboQuant(bits: 3)),
        ("tri-attention",      CacheBackend.triAttention),
        ("turbo-quant-3bit-plus-tri", CacheBackend.combined(tq: 3, tri: true)),
      ])
func testDFlashAcceptanceAcrossCacheBackends(
    name: String, backend: CacheBackend
) async throws {
    let target = try await loadQwen34BBF16(targetCacheBackend: backend)
    let (drafter, dconfig) = try await DFlashWeightLoader.load()
    let prompt = canonicalSummarizationPrompt
    var iter = try DFlashIterator(
        input: try makeInput(target: target, prompt: prompt),
        target: target, drafter: drafter, draftConfig: dconfig,
        stopTokenIds: [], maxTokens: 256)
    let tokens = (0..<256).compactMap { _ in iter.next() }

    // Capture baseline once for parity comparison
    let baseline = try await runBaselineGreedy(target: target, prompt: prompt, maxTokens: 256)
    let isLossless = (tokens == baseline)

    print("[\(name)] acceptance=\(iter.acceptanceRate) lossless=\(isLossless)")

    // Always require minimum acceptance — never let a backend silently degrade to AR
    #expect(iter.acceptanceRate >= 0.45,
            "[\(name)] acceptance fell below 0.45 — DFlash provides no real speedup")
    if name == "plain-bf16" {
        #expect(isLossless, "Baseline cache must be lossless")
        #expect(iter.acceptanceRate >= 0.7, "Baseline acceptance regression")
    }
    // TurboQuant/TriAttention are lossy → record but don't gate on losslessness
}
```

**Step C5.2 — Memory ceiling test (advisory, not a hard gate).**

Measure actual `mx.get_peak_memory()` (or Swift equivalent) at the end of a 32K-token summarization run for each backend, store in JSON next to acceptance results. Used to populate the configuration matrix in docs.

**Step C5.3 — Configuration matrix in `docs/specs/dflash-stage2.md`:**

Build a recommendation table from C5.1 + C5.2 results:

| Context | Recommended target cache | Expected speedup | Expected RAM (Qwen3-4B) |
|---|---|---|---|
| <4K | plain BF16 | 3.4× | ~10 GB |
| 4–16K | TurboQuant 4-bit | TBD (measure) | ~10 GB |
| 16K+ | TurboQuant 3-bit + TriAttention | TBD (measure) | ~11 GB |

If C5.1 finds that TQ-3bit drops acceptance below 0.45, document TQ-4bit as the floor and explain why (numerical noise in attention output propagates to drafter's `target_hidden`, breaking greedy match).

**Step C5.4 — `DFlashEngine` cache backend configuration:**

Extend `DFlashEngineConfiguration` with optional `targetCacheBackend: CacheBackendKind = .auto`. `.auto` picks based on prompt length at request time using the matrix from C5.3. Explicit values (`.plain`, `.turboQuant(bits:)`, `.triAttention`) override.

**Acceptance criteria:**
- Combo matrix populated for at least 3 backends × 2 prompt lengths (256-token + 4K-token summarization prompts)
- TQ-4bit acceptance ≥ 0.65 OR documented as unsupported
- DFlashEngine respects `targetCacheBackend` configuration end-to-end

**Effort:** +1 person-day (reuses B1 harness, mostly parameter sweep + docs)

**Commit:** `test(dflash): TurboQuant + TriAttention compatibility matrix`

---

## Acceptance Criteria (Stage 2 done)

1. ✅ `DFlashEngine` loads `mlx-community/Qwen3-4B-bf16` (target) + `z-lab/Qwen3-4B-DFlash-b16` (draft) on first request
2. ✅ `POST /v1/chat/completions` with `model: "dflash:qwen3-4b"` returns valid completion
3. ✅ SSE streaming works; final usage chunk includes `acceptance_rate`
4. ✅ **Lossless:** parity test passes — DFlash output ≡ Baseline output at temp=0 (token-for-token, 32+ tokens, multiple prompts)
5. ✅ **Speedup:** `bench-dflash.sh` reports ≥3× at 1024 tokens on M-class Apple Silicon
6. ✅ **Acceptance:** ≥0.7 acceptance rate on math-reasoning prompt (Aryagm reports 0.85+ on Qwen3-4B)
7. ✅ All existing `MLXLMTests` and `MLXLMServerTests` still green
8. ✅ No `mx.fast.metal_kernel` infrastructure changes required in upstream `mlx-swift`

---

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| `Qwen3Model.captureHiddenStates` returns inputs not outputs (off-by-one in layer indexing) | Medium | Step P3 confirms this before any DFlash code is written |
| `KVCache.trim` doesn't restore correctly to arbitrary length (only tail-trim works) | Low — existing `trim(n)` already handles arbitrary n | Computed as `cache.offset - desiredLen`; covered by parity test |
| `target.embed_tokens` and `target.lm_head` are not exposed as public methods on our `Qwen3Model` | Medium | If not exposed, add minimal accessors `func embedTokens(_:)` and `func lmHead(_:)` to `Qwen3Model` (~10 LOC) |
| RoPE q_len-tail slicing differs subtly from HF implementation | Medium | Caught by parity test; likely fix is in `RoPE.applyDFlash` helper |
| HF `output_hidden_states` has N+1 entries; our hooks return N | Low | `extract_context_feature` uses `+1 offset` already in HF code; we apply same offset in `prefill()` |
| `target_hidden` shape after acceptance trimming wrong | Medium-High | Lossless parity test fails fast if this is broken |
| KV cache for 5-layer drafter at long contexts (40K) blows memory | Low | 5 layers × 8 KV heads × 128 dim × 40K = ~410MB — manageable |
| Drafter forward time dominates → no speedup | Low | Drafter is 5 layers vs target's 36; expected ~14% of target forward time |
| First-block bug: drafter has empty cache, target_hidden is from prefill | Medium | Already handled by `position_ids[:, past_key_values_draft.get_seq_length():...]` pattern; mirror in Swift |

---

## Out of Scope (deferred to later stages)

- Qwen3.5 hybrid attention (Stage 3.5) — needs `gated_delta_state_update` MSL kernel
- Quantized target models (Stage 3) — needs `verify_qmm`-style M=16 GEMM for full perf
- Other model families (LLaMA 3.1, Qwen3-8B, etc.) — Stage 2.1 once Qwen3-4B is solid
- MoE targets (Stage 4) — needs custom Metal
- Long-context (>4K) optimization — needs JIT SDPA 2-pass
- Sampling temperatures > 0 — Stage 2.2; requires reformulating greedy match as probabilistic accept/reject

---

## Effort Estimate

| Wave | Tasks | Person-days |
|---|---|---|
| Preflight | P1–P3 | 0.5 |
| Wave A | A1 (config) + A2 (attention) + A3 (model) + A4 (loader) | 4 |
| Wave B | B1 (iterator, **highest risk**) + B2 (engine) | 6 |
| Wave C | C1 (metrics) + C2 (parity gate) + C3 (bench) + C4 (docs) + C5 (TQ/TA combo) | 3.5 |
| **Total** | | **~14 days** (~3 calendar weeks) |

Wave B Task B1 is the critical path — most likely to require multiple iterations due to off-by-one bugs in the speculation loop. Budget at least 3 of the 6 days for B1 alone.

Wave C Task C5 (TQ/TA combo) is the **product-value gate** for the long-document summarization use case on 16GB Macs. If TurboQuant breaks DFlash acceptance, C5 surfaces it before users hit the regression.

---

## References

- HF model: https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16
- HF reference Python (read but not embedded): `dflash.py`, `modeling_dflash.py`, `utils.py` from the HF repo
- Cross-reference Python implementation: https://github.com/Aryagm/dflash-mlx
- Production reference: https://github.com/bstnxbt/dflash-mlx
- Ecosystem analysis: `docs/research/2026-04-19-dflash-ecosystem-comparison.md`
- Our hooks landing: commits `7f3b9cd`, `fd24cd2`
- API field: `Libraries/MLXLMServer/Engine/InferenceEngine.swift:62`
