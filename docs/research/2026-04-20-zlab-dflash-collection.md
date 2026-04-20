# z-lab DFlash Collection — Model Inventory & Our Coverage Roadmap

**Date:** 2026-04-20
**Source:** https://huggingface.co/collections/z-lab/dflash
**Context:** Stage 2 (Qwen3-4B BF16) is functionally complete on branch `feat/dflash-stage2-bf16`. This inventory maps the remaining 12 models in z-lab's official DFlash collection to our roadmap stages and clarifies what is required to extend `mlx-swift-lm` to each.

---

## Full collection (13 drafts, 2026-04-20 snapshot)

Sorted by update recency. All drafts are text-generation models trained to predict 16-token blocks conditioned on target hidden states.

| Draft repo | Draft size | Target | Target arch | Updated | Our runtime status |
|---|---:|---|---|---|---|
| z-lab/Qwen3.6-35B-A3B-DFlash | 0.5B | Qwen3.6-35B-A3B | **MoE** (A3B ≈ 3B active) | 18h ago | ❌ Stage 4 (MoE + `verify_qmm`) |
| z-lab/Kimi-K2.5-DFlash | 3B | Kimi-K2.5 | exotic | 3d ago | ❌ Stage 5 (no target in MLXLLM) |
| z-lab/Qwen3.5-4B-DFlash | 0.5B | Qwen3.5-4B | dense + hybrid linear attn | 13d ago | ❌ Stage 3.5 (GatedDeltaNet MSL kernel) |
| z-lab/Qwen3.5-9B-DFlash | 1B | Qwen3.5-9B | dense + hybrid linear attn | 13d ago | ❌ Stage 3.5 |
| z-lab/Qwen3.5-35B-A3B-DFlash | 0.5B | Qwen3.5-35B-A3B-4bit | MoE 4-bit | 13d ago | ❌ Stage 4 |
| z-lab/Qwen3.5-27B-DFlash | 2B | Qwen3.5-27B | dense | 13d ago | ❌ Stage 3 (Qwen3.5 dense model) |
| z-lab/Qwen3-Coder-Next-DFlash | 0.5B | Qwen3-Coder-Next | Qwen3 coder variant | 13d ago | ❌ needs Qwen3-Coder target |
| z-lab/Qwen3-Coder-30B-A3B-DFlash | 0.5B | Qwen3-Coder-30B-A3B | coding MoE | 13d ago | ❌ Stage 4 |
| **z-lab/Qwen3-4B-DFlash-b16** | **0.5B** | **Qwen3-4B** | **dense BF16** | 13d ago | **✅ Stage 2 (current)** |
| z-lab/Qwen3-8B-DFlash-b16 | 1B | Qwen3-8B | dense BF16 | 13d ago | ⚠️ same arch, config swap only |
| z-lab/gpt-oss-20b-DFlash | 0.8B | gpt-oss-20b | GPT-OSS dense | 13d ago | ❌ needs GPT-OSS target in MLXLLM |
| z-lab/gpt-oss-120b-DFlash | 0.8B | gpt-oss-120b | GPT-OSS dense | Mar 17 | ❌ same, + 120B memory |
| z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat | 1B | Meta-Llama-3.1-8B-Instruct | LLaMA dense | 13d ago | ⚠️ LLaMA3Model exists; needs `DFlashTargetModel` conformance + draft arch adapter |

Total: 13 drafts, covering 4 model families: Qwen3/3.5/3.6 (+ Coder variants), LLaMA 3.1, GPT-OSS, Kimi-K2.5.

---

## Naming conventions

- **`-b16` suffix** — only on older Qwen3 family (4B, 8B). Indicates bfloat16 training run using a specific block_size=16 configuration that predates Qwen3.5. New drafts (Qwen3.5+) drop the suffix.
- **`-UltraChat` suffix** — LLaMA 3.1 draft was trained on the UltraChat dataset specifically. Other drafts use the default z-lab training corpus.
- **`A3B` suffix** (on Qwen3.5/3.6 targets) — "Active 3B" MoE routing; only 3B parameters active per forward pass out of the total.

---

## Compression ratios

| Pair | Target params | Draft params | Ratio |
|---|---:|---:|---:|
| Qwen3-4B | 4B | 0.5B | 8× |
| LLaMA 3.1 8B | 8B | 1B | 8× |
| Qwen3.5-27B | 27B | 2B | 13.5× |
| Qwen3.5-9B | 9B | 1B | 9× |
| gpt-oss-20b | 20B | 0.8B | 25× |
| gpt-oss-120b | 120B | 0.8B | **150×** |
| Qwen3.6-35B-A3B | 35B | 0.5B | 70× |
| Qwen3.5-35B-A3B | 35B | 0.5B | 70× |
| Kimi-K2.5 | (large) | 3B | unknown |

Pattern: MoE targets get extremely compact drafts (0.5B), because only active 3B of the target is on the critical path. Dense targets need proportionally larger drafts (1B for 8–9B, 2B for 27B).

---

## Draft architecture family

All Qwen3/3.5/3.6 drafts appear to use the same architecture we already implemented: 5-layer Qwen3-like decoder with one modified attention (K/V concat of target hidden + noise) + `fc` projection + 2 RMSNorms. Hyperparameters differ per size:

| Draft | `hidden_size` | `num_hidden_layers` | `num_target_layers` |
|---|---:|---:|---:|
| Qwen3-4B-DFlash-b16 | 2560 | 5 | 36 |
| Qwen3-8B-DFlash-b16 | (TBD, probably ~3072) | (TBD) | (TBD) |
| Qwen3.5-4B-DFlash | (TBD) | (TBD) | (TBD) |

Our current Swift `DFlashDraftModel` takes `DFlashDraftConfig` which already abstracts these; loading a different draft is a JSON config swap away.

**Open questions for non-Qwen drafts:**
- LLaMA 3.1 draft probably uses LLaMA backbone (different RoPE, different MLP activation). May need a `LLaMADFlashDraftModel` variant.
- GPT-OSS draft arch is not publicly documented. May be Qwen-shaped or LLaMA-shaped — needs inspection of `dflash.py` shipped in the repo (z-lab publishes custom `.py` alongside each draft).
- Kimi-K2.5 draft at 3B is an outlier — larger drafter suggests more complex architecture.

---

## Our extension roadmap

Given Stage 2 (Qwen3-4B BF16) complete on `feat/dflash-stage2-bf16`, the cheapest-to-richest extensions:

### Stage 2.1 — Qwen3-8B BF16 (~1 day, trivial)
- Same `Qwen3Model` target via `LLMModelFactory.loadContainer(configuration: .init(id: "mlx-community/Qwen3-8B-bf16"))`
- Same `DFlashDraftModel` class; load `z-lab/Qwen3-8B-DFlash-b16` — config has different hidden sizes but same structure
- CLI gets `--dflash-target-8b / --dflash-draft-8b / --dflash-alias-8b` or a new registry entry
- Covers 32GB+ Macs (weights ~16GB + draft ~2GB)

### Stage 3 — Qwen3.5 dense (4–27B) (~1–2 weeks)
Requires a `Qwen3_5Model: DFlashTargetModel` in `MLXLLM/Models/`. The existing mlx-swift Qwen3 code may already cover Qwen3.5 dense paths (attention layer types vary per layer).

### Stage 3.5 — Qwen3.5 hybrid attention (~1 week additional)
- Port Aryagm's `gated_delta_state_update` MSL kernel (~100 LOC) + pure-Swift fallback
- Needed because Qwen3.5/Qwen3-Next mixes full attention + sliding + recurrent linear-attention state; the linear state needs per-layer rollback using an innovation-tape kernel
- Enables Qwen3.5-4B and 9B

### Stage 4 — MoE + quantized (~3–4 weeks)
Unlocks: Qwen3.5-27B, Qwen3.5-35B-A3B-4bit, Qwen3.6-35B-A3B, Qwen3-Coder-30B-A3B
- MoE router in target (for A3B variants)
- `verify_qmm` Metal kernel for M=16 int4 quantized GEMM (the kernel bstnxbt wrote, 300+ MSL LOC). Without it, Aryagm's stock-MLX impl caps at ~1.4× speedup on 4-bit vs 2.37× with `verify_qmm`
- Stage 4 gives flagship models on Apple Silicon

### Stage 4b — LLaMA 3.1 (~1 week)
- Make existing `LLaMA3Model` (in `MLXLLM/Models/`) conform to `DFlashTargetModel`
- Possibly write a `LLaMADFlashDraftModel` variant if the draft repo's `dflash.py` diverges from Qwen3 structure
- Extends coverage to Meta model family — important for apps targeting broad ecosystem

### Stage 5 — GPT-OSS + Kimi-K2.5 (~4+ weeks)
- Need full target model implementations added to `MLXLLM/Models/`
- Kimi-K2.5 is likely the most complex; not seen in any public MLX port
- Probably lowest priority until market demand

---

## Implications for native Mac app use cases

| Target device | Recommended model | Required stage |
|---|---|---|
| 16GB Mac Air (summarization, chat) | Qwen3-4B BF16 + DFlash | **Stage 2 ✅ today** |
| 32GB Mac Pro/Studio | Qwen3-8B BF16 + DFlash | **Stage 2.1 (~1 day)** |
| 64GB Mac Studio | Qwen3.5-27B dense + DFlash, or Qwen3.5-35B-A3B-4bit | Stage 3 or Stage 4 |
| 128GB+ Mac Studio | Qwen3.6-35B-A3B + DFlash (flagship) | Stage 4 |
| iPhone 15 Pro 8GB | Qwen3-4B 4-bit (not BF16) + DFlash | Stage 3 (quantized path) |
| iPad Pro M4 16GB | Qwen3-4B BF16 + DFlash | Stage 2 ✅ today (if metallib bundles on iOS) |

**Key point:** Stage 2 already covers the most common consumer Mac config (16GB Air + 8GB iPhone needs quantization). Stage 2.1 is a trivial 1-day effort that doubles our device coverage to 32GB+ machines.

---

## Upstream velocity

z-lab's collection got its newest addition **18 hours before this snapshot** (Qwen3.6-35B-A3B). The project is in active development. Our implementation choices should favor:

1. **Narrow abstractions** — `DFlashTargetModel` / `DFlashDraftConfiguration` / `DFlashDraftingModel` protocols already let us plug in new target/draft pairs without touching the iterator.
2. **Config-driven** — our `DFlashDraftConfig` already reads `config.json` verbatim from HF. New drafts should load as config swaps.
3. **Architecture adapter pattern** — keeping draft class agnostic of specific target arch helps when Qwen3.6 adds features we don't yet support.

If `mlx-swift-lm` ships an open-source DFlash implementation for MLX Swift, we'd be the first. That positions us to be **the** Swift implementation cited in benchmarks, especially for flagship Qwen3.5/3.6 MoE models on Apple Silicon.

---

## References

- HF collection: https://huggingface.co/collections/z-lab/dflash
- DFlash paper: arXiv:2602.06036 (Chen et al., 2026)
- Our Stage 2 plan: `docs/plans/2026-04-19-dflash-stage2-bf16-mvp.md`
- Our Wave B decomposition: `docs/plans/2026-04-20-dflash-stage2-b1-decomposition.md`
- Ecosystem review (comparison to Python impls): `docs/research/2026-04-19-dflash-ecosystem-comparison.md`
