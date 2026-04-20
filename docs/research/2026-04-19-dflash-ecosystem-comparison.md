# DFlash Ecosystem Comparison & Implementation Roadmap

**Date:** 2026-04-19
**Branch:** `feat/mlxlm-server`
**Scope:** Compare third-party DFlash implementations (VoiceInk PR #14, bstnxbt/dflash-mlx, Aryagm/dflash-mlx) to inform our native Swift implementation strategy.

> **Status note (2026-04-20):** Stage 2 is now implemented in `mlx-swift-lm`
> for the BF16 `mlx-community/Qwen3-4B-bf16` target paired with
> `z-lab/Qwen3-4B-DFlash-b16`. The OpenAI-compatible server route
> `dflash:qwen3-4b` is smoke-tested end-to-end, and final usage payloads now
> include `acceptance_rate` metrics.

---

## TL;DR

- **VoiceInk PR #14** doesn't implement DFlash — it spawns the official Python `dflash-serve` process and proxies via OpenAI-compatible HTTP. ~600 lines of Swift glue code.
- **bstnxbt/dflash-mlx** (511★, Python) — production-grade with custom Metal kernels (`verify_qmm`, JIT SDPA 2-pass). 86–91% acceptance, 1.34–4.37× speedup. Uses `mx.fast.metal_kernel`.
- **Aryagm/dflash-mlx** (323★, Python) — native MLX port. **3.1–4.6× speedup on Qwen3-4B BF16** (M4 Max, 4028 tokens) using **zero custom Metal on the hot dense-attention path**. Has one custom kernel (`gated_delta_state_update`, ~100 lines MSL) only for Qwen3.5 hybrid attention, with pure-Python fallback. Limited to Qwen3-4B / Qwen3.5-4B model families. Quantized 4-bit hits only 1.4× without `verify_qmm`.
- **Our `mlx-swift-lm`** — has hooks (`captureHiddenStates`, `captureHiddenStatesAndLogits`), engine routing infrastructure, baseline engine, but **no DFlash engine yet**. Existing `SpeculativeTokenIterator` is classic Leviathan-style spec decoding (port of mlx-lm), not DFlash.

---

## 1. VoiceInk PR #14 (charliemeyer2000)

**URL:** https://github.com/charliemeyer2000/VoiceInk/pull/14
**State:** MERGED
**Strategy:** Process supervision + HTTP proxy

### What it is
- 600 LOC Swift across 9 files
- `DFlashServerManager.swift` (192 lines) — `Process()` lifecycle, health timer, stdio drain
- `DFlashModelRegistry.swift` (164 lines) — model catalog + `huggingface-cli download` shell-out
- `DFlashSettingsView.swift` (151 lines) — SwiftUI controls
- `AIService.swift` — adds `.dflash` enum case alongside Groq/OpenAI/Ollama; same OpenAI-compatible code path

### What it doesn't do
- No DFlash algorithm in Swift
- No MLX integration
- No model loading
- No verify/rollback logic

### Takeaway
**Validates the proxy strategy as a real shipping option.** A production app shipped DFlash to users by treating `dflash-serve` as a subprocess. Acceptance rate, speedup, and correctness are inherited from the upstream Python implementation.

---

## 2. bstnxbt/dflash-mlx (Python, 511★)

**URL:** https://github.com/bstnxbt/dflash-mlx
**Strategy:** Native MLX implementation with custom Metal kernels

### Architecture
```
dflash_mlx/
├── engine.py            # _BaseEngine: arm_rollback / verify / rollback
│                        # FullAttentionEngine, HybridGDNEngine
├── runtime.py           # Verify-loop orchestration
├── draft_backend.py     # Draft model interface
├── recurrent_rollback_cache.py  # Tape-replay rollback for GatedDeltaNet
├── kernels.py           # Custom Metal: gated_delta_tape (innovation tape)
├── verify_qmm.py        # Custom Metal: M=16 int4 GEMM (mma2big, mma2big_pipe)
├── verify_linear.py     # Custom Metal: M=16 fp GEMM
├── adapter.py           # Per-architecture hooks
├── model.py             # Model loading
├── serve.py             # OpenAI-compatible server (wraps mlx_lm.server)
└── generate.py          # CLI entry point
```

### Custom Metal kernels (the hard part)

| Kernel | Purpose | Why custom |
|---|---|---|
| `verify_qmm` (`mma2big`) | M=16 int4 quantized matmul | Stock `mx.quantized_matmul` is M=1 optimized; M=16 needs simdgroup MMA tiles |
| `verify_qmm` (`mma2big_pipe`) | Same + K-split with double-buffered staging | For very large K dimensions |
| `verify_linear` | M=16 non-quantized GEMM | Same shape mismatch as verify_qmm |
| `gated_delta_tape` | Innovation tape for recurrent linear-attention rollback | GatedDeltaNet state restoration |
| JIT SDPA 2-pass | Long-context verify (N≥1024) | Numerical alignment with stock MLX attention |

All kernels written via `mx.fast.metal_kernel(name, source, ...)` — runtime MSL compilation.

### Auto-enable heuristic
- `verify_qmm` auto-enabled on MoE targets and dense models with ≥40 layers
- Opt-in override via `DFLASH_VERIFY_LINEAR={0,1}` env var
- Variant selection: `K >= 8192 or N <= 8192 → mma2big_pipe`, else `mma2big`

### Benchmarks (Apple M5 Max, 64GB)

| Model | Tokens | Baseline (tok/s) | DFlash (tok/s) | Speedup | Acceptance |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B | 1024 | 53.80 | 182.87 | 3.40× | 86.43% |
| Qwen3.5-4B | 2048 | 53.90 | 188.70 | 3.49× | 87.70% |
| Qwen3.5-4B | 4096 | 53.49 | 195.84 | 3.66× | 88.35% |
| Qwen3.5-4B | 8192 | 53.28 | 160.51 | 3.02× | 87.30% |
| Qwen3.5-9B | 1024 | 30.95 | 135.34 | 4.37× | 89.55% |
| Qwen3.5-9B | 2048 | 30.70 | 113.00 | 3.65× | 89.16% |
| Qwen3.5-9B | 4096 | 30.56 | 94.59 | 3.06× | 88.31% |
| Qwen3.5-9B | 8192 | 29.43 | 66.94 | 2.22× | 86.67% |
| Qwen3.5-27B-4bit | 1024 | 33.55 | 79.02 | 2.37× | 90.04% |
| Qwen3.5-27B-4bit | 2048 | 33.10 | 70.21 | 2.12× | 89.60% |
| Qwen3.5-27B-4bit | 4096 | 31.47 | 55.68 | 1.77× | 88.38% |
| Qwen3.5-27B-4bit | 8192 | 33.88 | 45.29 | 1.34× | 85.97% |
| Qwen3.5-35B-A3B-4bit | 1024 | 143.03 | 248.85 | 1.76× | 89.26% |
| Qwen3.6-35B-A3B-4bit | 1024 | 138.26 | 300.33 | 2.20× | 91.02% |

**Methodology:** stock `mlx_lm.stream_generate` (baseline) vs DFlash runtime, sequential, 3 repeats, median, 60s cooldown. Math reasoning prompt with chat templates.

---

## 3. Aryagm/dflash-mlx (Python, 323★)

**URL:** https://github.com/Aryagm/dflash-mlx
**Strategy:** Native MLX port with **one** custom Metal kernel (only for Qwen3.5 hybrid path; has Python fallback)

### Architecture
```
dflash_mlx/
├── runtime.py                   # Verify orchestration
├── draft.py                     # DraftArgs + block-diffusion forward
├── adapters.py                  # Per-architecture hooks; contains gated_delta_state_update Metal kernel
├── custom_qwen35_model.py       # Qwen3.5 hybrid attention support (duplicates the kernel)
├── model_prep.py                # Weight loading
├── api.py                       # Python API (DFlashGenerator)
├── cli.py / chat_cli.py / benchmark_cli.py / inspect_cli.py
└── history.py                   # Benchmark history tracking
```

### Custom Metal usage (correction from earlier draft)
- **One** `mx.fast.metal_kernel` definition: `gated_delta_state_update` in `adapters.py:82`
- **Only invoked** for GatedDeltaNet (recurrent linear attention) — i.e. Qwen3.5 / Qwen3-Next family hybrid stack
- **Has pure-Python fallback** when Metal is unavailable
- **NOT used** on dense Qwen3 path — so Aryagm's headline 4.6× on Qwen3-4B BF16 is achieved with **zero custom Metal on the hot path**

### Headline benchmarks (M4 Max 36GB, Qwen3-4B BF16, target `mlx-community/Qwen3-4B-bf16`, draft `z-lab/Qwen3-4B-DFlash-b16`)

| Max tokens | MLX-LM baseline | dflash-mlx | Speedup | Avg accepted (of 16) |
|---:|---:|---:|---:|---:|
| 512 | 42.3 | 133.1 | **3.1×** | 8.81 |
| 1024 | 42.0 | 144.6 | **3.4×** | 9.66 |
| 2048 | 41.3 | 174.4 | **4.2×** | 11.97 |
| 4028 | 40.6 | 186.4 | **4.6×** | 13.55 |

### Quantized comparison

| Runtime | tok/s | vs MLX baseline |
|---|---:|---:|
| MLX-LM 4-bit baseline | 110.5 | 1.0× |
| dflash-mlx 4-bit (no `verify_qmm`) | 159.2 | **1.4×** |

### Qwen3.5 archived results (worse — hybrid attention is harder)

| Draft mask | Max tokens | Baseline | dflash-mlx | Speedup | Avg accepted |
|---|---:|---:|---:|---:|---:|
| causal | 512 | 39.15 | 84.75 | 2.17× | 6.10 |
| none | 1024 | 39.80 | 76.40 | 1.92× | 5.49 |
| none | 2048 | 39.74 | 73.98 | 1.86× | 5.37 |

Aryagm explicitly notes Qwen3.5 path is "supported but archived". Verifier forward dominates (21.6s of 27.7s on 2048-token run); rollback is only 0.11s.

### Key design points
- **Block size hardcoded to 16** in `DraftArgs(block_size: int = 16)`
- Reuses `mlx_lm.models.qwen3.MLP` and `mlx_lm.models.rope_utils.initialize_rope`
- Uses `mlx_lm.utils.quantize_model` for quantized variants
- Per-layer KV cache rollback (extends stock MLX cache)
- `ADDING_MODELS.md` documents adapter pattern for new architectures

### Limitations vs bstnxbt
- Only Qwen3-4B and Qwen3.5-4B (Qwen3.5 marked "incomplete")
- No custom Metal kernels → loses on quantized + MoE models
- No published headline benchmark table in README (only methodology files)

### Why this is the better reference for our Swift port
- **Smaller surface area** — easier to translate file-by-file
- **No custom Metal** — matches what's feasible in mlx-swift today
- **Adapter pattern** — maps cleanly to our `EngineRegistry` design
- **Documented model-addition flow** — `ADDING_MODELS.md` is a port roadmap

---

## 4. Our `mlx-swift-lm` State Audit

### What's ready

| Component | Location | Status |
|---|---|---|
| Hidden state extraction hooks | `MLXLLM/Models/Qwen3.swift:187,231,313,334` | ✅ landed (commits `7f3b9cd`, `fd24cd2`) |
| `BaselineEngine` (autoregressive) | `MLXLMServer/Engine/BaselineEngine.swift` | ✅ working |
| `EngineRegistry` with prefix routing | `EngineRegistry.swift` (handles `dflash:foo` → dflash) | ✅ ready to accept dflash engine |
| `Usage.acceptanceRate` API field | `InferenceEngine.swift:62` | ✅ field only, not computed |
| Per-engine health breakdown | `EngineRegistry.swift:115` | ✅ |
| `SpeculativeTokenIterator` | `MLXLMCommon/Evaluate.swift:744` | ⚠️ classic Leviathan-style spec decoding (mlx-lm port), **not DFlash** |
| OpenAI-compatible server | `MLXLMServer/HTTP/` | ✅ chat completions, SSE streaming |

### What's missing for DFlash

- ❌ `DFlashEngine` class (no file in `MLXLMServer/Engine/`)
- ❌ Loader for `z-lab/*-DFlash-b16` draft weights
- ❌ Block-diffusion drafter forward pass (denoising loop, 16 tokens parallel)
- ❌ Verify+rollback loop with per-layer KV cache restore
- ❌ Acceptance-rate accounting that populates `Usage.acceptanceRate`
- ❌ Benchmark CLI for tokens/sec measurement
- ❌ `mx.fast.metal_kernel` equivalent in mlx-swift (blocker for any custom kernel work)

---

## 5. Why `verify_qmm` Custom Metal Kernel Is The Hardest Piece

Ranked by severity:

### 5.1 The Python file is a Metal Shading Language template
The "Python" code in `verify_qmm.py` is just an f-string that builds MSL source and passes it to `mx.fast.metal_kernel(source=...)`. The actual algorithm lives in MSL: `simdgroup_matrix<T,8,8>`, `simdgroup_load`, `simdgroup_multiply_accumulate`, threadgroup memory layout, two variants `mma2big` vs `mma2big_pipe` with K-split + double-buffered staging.

**Porting** = rewriting the same MSL + finding the binding API in mlx-swift.

### 5.2 `mx.fast.metal_kernel` is not first-class in mlx-swift
- Python MLX: stable public API since 2024
- mlx-swift: equivalent is either missing, exposed only through private `MLX.Cmlx` C-bindings, or requires raw `MTLDevice.makeLibrary(source:)` bypassing MLX

**Before any kernel work**, infrastructure for runtime MSL compilation needs to land in mlx-swift itself (1–2 weeks upstream PR work).

### 5.3 Five custom kernels, not one
| Kernel | Purpose | Difficulty |
|---|---|---|
| `verify_qmm` `mma2big` | M=16 int4 GEMM | High |
| `verify_qmm` `mma2big_pipe` | Same + K-split + double-buffered | Very high |
| `verify_linear` | M=16 fp GEMM | Medium |
| `gated_delta_tape` | Innovation tape for recurrent rollback | Very high (Qwen3.5 only) |
| JIT SDPA 2-pass | Long-context numerical parity | Very high |

Each needs its own parity test against stock MLX.

### 5.4 Stock MLX is M=1-optimized
`mx.quantized_matmul` was designed for M=1 (single-token AR). At M=16 (speculative batch):
- Dequantization cost amortizes badly
- Memory pattern wrong for speculative tile shape
- Can't use Apple's simdgroup MMA (needs M≥8)

bstnxbt's kernel hardcodes `constexpr int BM = 16` — locked to speculative decoding with `block_size=16`.

### 5.5 Hardware-specific microarchitecture
The MSL code makes assumptions about:
- 32 threads per simdgroup, 4 simdgroups per threadgroup
- 8×8 MMA tiles (Apple Silicon GPU "tensor cores")
- Threadgroup shared memory layout for dequantized weights
- `threadgroup_barrier` synchronization between dequant and compute

These need re-tuning per hardware (M5 Max ≠ M4 Pro: different ALU per simdgroup, different bandwidth-to-compute ratio).

### 5.6 Numerical parity is a first-class problem
bstnxbt has `tests/test_verify_qmm_parity.py` and `tests/test_verify_linear_parity.py` — formal tests that custom kernel output is bit-equivalent (within bf16 tolerance) to `mx.quantized_matmul`. Without parity, speculative decoder desyncs from target → acceptance rate drops to ~0.

### 5.7 ROI is bounded to specific model classes
Custom kernels matter for:
- 4-bit quantized models (Qwen3.5-27B-4bit, MoE 35B-A3B-4bit)
- Dense models with ≥40 layers

For Qwen3-4B BF16 (likely our most common case), stock MLX path already gives bstnxbt 3.4× speedup **without** custom kernels.

### Realistic effort estimate

| Stage | Person-weeks | Blocker |
|---|---|---|
| Land `mx.fast.metal_kernel` API in mlx-swift | 1–2 | Upstream PR likely required |
| Port `verify_linear` (no quantization) | 1 | Simplest kernel, good entry point |
| Port `verify_qmm` `mma2big` | 2 | Hardware benchmarks needed |
| Port `verify_qmm` `mma2big_pipe` (double-buffered) | 1–2 | Only if first variant insufficient |
| Port `gated_delta_tape` | 2 | Only for Qwen3.5 hybrid models |
| Parity tests for all kernels | 1 | Cannot ship without |
| Calibrate auto-enable thresholds | 1 | Per-model × per-context benchmarks |
| **Total** | **9–12** | — |

For comparison: native DFlash **without** custom kernels (just speculative loop on stock mlx-swift) is ~2–3 weeks. Custom kernels are 3–4× the entire native implementation effort.

---

## 6. Model Format Clarification

### Target weights — no conversion needed
`mlx-community/Qwen3-4B-bf16`, `Qwen/Qwen3.5-9B`, etc. are standard MLX-converted weights. Loaded by our existing `Qwen3Model` without modification.

### Draft weights — separate trained model, not a conversion
`z-lab/Qwen3-4B-DFlash-b16` is **not** a converted Qwen. It's:
- A separate ~1B parameter model trained from scratch by z-lab
- Architecture: Qwen-like backbone + block-diffusion denoising head
- Trained via distillation: predicts tokens conditioned on target's hidden states
- `b16` suffix = block size 16 (block-diffusion proposal length), **not** quantization bits

Evidence from Aryagm's `dflash_mlx/draft.py`:
```python
@dataclass
class DraftArgs:
    block_size: int = 16          # ← origin of "b16"
    dflash_config: dict | None = None
```

### What this means for Swift
| Artifact | Action | Status |
|---|---|---|
| Target `Qwen3-4B-bf16` weights | Load via `MLXLMCommon.ModelFactory` | ✅ works today |
| Target forward + hidden state extraction | `Qwen3Model.captureHiddenStatesAndLogits` | ✅ landed |
| Draft `z-lab/*-DFlash-b16` safetensors | Load standard format | ⚠️ loads, but no model class |
| Draft forward pass (block-diffusion denoising) | Implement in Swift — **not** Qwen forward | ❌ missing |
| Verify + rollback engine | New `DFlashEngine` next to `BaselineEngine` | ❌ missing |
| Per-layer KV cache rollback | Extend `KVCache` with `restore(to:)` | ❌ missing |

**Summary:** weights don't need conversion (safetensors format works), but the **architecture class** to consume those weights doesn't exist in Swift yet.

---

## 7. Recommended Path Forward

### Stage 1 — Proxy engine (quick win, ~1 week)
Add `DFlashProxyEngine` to `MLXLMServer/Engine/`. Spawns `dflash-serve` Python subprocess (mirroring VoiceInk's pattern), proxies HTTP requests. Ships production-grade DFlash today by inheriting bstnxbt's perf. Decouples API surface from native implementation timeline.

### Stage 2 — Native MVP (~2–3 weeks)
Reference `Aryagm/dflash-mlx`. BF16 dense models only (Qwen3-4B). **Expected 3.1–4.6× speedup** (Aryagm empirically measures exactly this on M4 Max with zero custom Metal on the hot path). Components:
- `DFlashEngine` in `MLXLMServer/Engine/`
- Block-diffusion drafter Swift class (`MLXLLM/Models/DFlashDraft.swift`)
- Loader for `z-lab/*-DFlash-b16` safetensors
- Verify + per-layer rollback loop
- Wire `Usage.acceptanceRate` accounting

### Stage 3 — Quantized model support (~1–2 weeks)
Use stock `mx.quantized_matmul`. Aryagm's empirical ceiling without `verify_qmm`: **~1.4× on Qwen3-4B 4-bit** (110.5 → 159.2 tok/s). Validates int4 serving works end-to-end; leaves room for Stage 4 if we want to match bstnxbt's 2.37× on 27B-4bit.

### Stage 3.5 — Qwen3.5 hybrid attention (~1 week)
Port Aryagm's single `gated_delta_state_update` Metal kernel (~100 lines MSL, has pure-Python fallback → pure-Swift fallback is straightforward). Enables Qwen3.5 family but with the "archived" acceptance degradation Aryagm documents (~1.9× on 2048 tokens).

### Stage 4 — bstnxbt-grade Metal kernels (only on demand, ~9–12 weeks)
Only pays off for:
- Qwen3.5-27B-4bit (`verify_qmm` lifts 1.4× → 2.37×)
- MoE 35B-A3B-4bit (same)
- Very long contexts (`JIT SDPA 2-pass` for N≥1024)

Skip unless real users need these. Buy-vs-build option: keep Stage 1 proxy engine for heavy models permanently, native engine for BF16 + Qwen3.5 dense. Hybrid is often the pragmatic answer.

---

## 7.1 Combo with TurboQuant + TriAttention (our existing memory optimizations)

Our codebase already ships two orthogonal KV-cache optimizations on the same `KVCache` protocol:

- **TurboQuant** (`Libraries/MLXLMCommon/KVCache.swift:187`, branch `review-fixes-2026-04-16` already shipped) — 3-bit lossy quantization of K/V tensors. Reduces KV cache footprint to ~3/16 of BF16.
- **TriAttention** (`Libraries/MLXLMCommon/TriAttention.swift`) — sparsified attention scoring for long contexts (≥4K). Drops low-importance attention computations.

These were built independent of DFlash but are **complementary** for the canonical use case (long-document summarization on consumer Macs):

### Memory ceiling: Qwen3-4B target on 16GB Mac (e.g. Air)

| Context | Plain BF16 KV | TurboQuant 3-bit KV | Total RAM (TQ) |
|---:|---:|---:|---:|
| 4K | 600 MB | 115 MB | ~10 GB ✅ |
| 16K | 2.4 GB | 460 MB | ~10.5 GB ✅ |
| 32K | 4.8 GB | 900 MB | ~11 GB ✅ |
| 32K (no TQ) | 4.8 GB | — | ~14.8 GB ⚠️ swap |

For long-document summarization, TQ is the difference between "comfortable on 16GB Air" and "swapping to disk". For users, **TQ matters more than DFlash** here — TQ enables the workload at all; DFlash makes it 3× faster once it fits.

### Compatibility risk: TurboQuant + DFlash lossless property

DFlash relies on `target.hidden_states[i+offset]` being **bit-stable** between prefill and verify, because:
- Prefill captures `target_hidden` from the target's hidden states at layers `[1,9,17,25,33]`
- Drafter consumes that `target_hidden` to propose 16 tokens
- Verify recomputes `target_hidden` from the same target after attending to the new block
- Greedy match `draft[i] == verifier[i]` succeeds only if these computations stay coherent across the rejection point

TurboQuant introduces numerical noise into the KV cache. If the noise is amplified through attention output → hidden states → drafter input, **acceptance rate may degrade**. Worst case: drops from 0.85 (plain) to 0.5 (TQ-3bit), reducing speedup from 3× to ~1.7×. Still a win, but not as dramatic.

Validation plan: Stage 2 plan adds **Task C5** — sweep acceptance rate across `{plain, TQ-4bit, TQ-3bit, TriAttn, combined}` × `{short prompt, 4K prompt}`, populate a configuration recommendation matrix in `docs/specs/dflash-stage2.md`. Recommended defaults will be:

| Context | Recommended target cache backend |
|---|---|
| <4K | plain BF16 (no TQ) |
| 4–16K | TurboQuant 4-bit (safer noise floor than 3-bit) |
| 16K+ | TurboQuant 3-bit + TriAttention |

If TQ-3bit empirically holds acceptance ≥0.7, defaults move to TQ-3bit at all context lengths.

### Impact on app-embedding decision

For the canonical use case "embed in native macOS summarization app":
- The user-facing value of TQ + DFlash combo > sum of either alone
- 16GB Air (most common consumer config) becomes a viable target for long-doc summarization
- Stack: `MLXLLM` (target) + `MLXLMCommon` (KVCache backends + DFlashIterator + TriAttention) — embeddable as Swift Package, no `MLXLMServer` HTTP overhead needed

---

## 8. Validation Plan (Before Writing Any DFlash Code)

Two benchmarks worth running first to establish baseline + ground truth:

1. **Baseline `mlx-lm-server` tok/s** — measure our `BaselineEngine` on Qwen3.5-4B / 9B at contexts 1024/2048/4096/8192. Confirms our baseline matches bstnxbt's baseline (~53 tok/s for Qwen3.5-4B). Validates methodology.

2. **bstnxbt's `dflash-benchmark` on our hardware** — `uv pip install dflash-mlx`, run their benchmark script. Gives:
   - Real DFlash numbers on our specific machine (M5 Max in their README may not match our box)
   - Acceptance rate ground truth
   - Target numbers our native engine should approach

Output format: extend the bstnxbt benchmark table with two columns:

| Model | Tokens | Our baseline | bstnxbt baseline | bstnxbt DFlash | Our DFlash |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B | 1024 | TBD | 53.80 | 182.87 | N/A (not impl) |

Scripts to prepare (user runs manually, per project rules):
- `scripts/bench-baseline.sh` — wraps `mlx-lm-server` + math reasoning prompt
- `scripts/bench-bstnxbt-reference.sh` — installs and runs bstnxbt's benchmark CLI

---

## References

- [VoiceInk PR #14](https://github.com/charliemeyer2000/VoiceInk/pull/14)
- [bstnxbt/dflash-mlx](https://github.com/bstnxbt/dflash-mlx)
- [Aryagm/dflash-mlx](https://github.com/Aryagm/dflash-mlx)
- [DFlash paper (arXiv:2602.06036)](https://arxiv.org/abs/2602.06036)
- [z-lab DFlash collection on HuggingFace](https://huggingface.co/collections/z-lab/dflash)
- Our hidden state hooks: `Libraries/MLXLLM/Models/Qwen3.swift:187,231,313,334`
- Our engine routing: `Libraries/MLXLMServer/Engine/EngineRegistry.swift`
- Our existing spec decoding (Leviathan-style, not DFlash): `Libraries/MLXLMCommon/Evaluate.swift:744`
