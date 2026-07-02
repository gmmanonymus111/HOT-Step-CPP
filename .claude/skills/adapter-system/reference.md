# Adapter system — deep reference

Companion to [SKILL.md](SKILL.md). Line references verified against the repo at commit
`168dcb5` (2026-07-02). If line numbers have drifted, grep for the named symbol.

## 1. Merge mode internals (`engine/src/adapter-merge.h`)

- Entry: `adapter_merge(wctx, ws, path, scale, backend, promote_f32, rebase_source, rebase_beta)` (:1332). Accepts a PEFT directory (`adapter_model.safetensors` + `adapter_config.json`) or a flat LyCORIS file; auto-detects LoKr via `.lokr_w1/.lokr_w2` tensor names (`adapter_detect_lokr`, :360; reverse name map `lokr_build_reverse_map`, :328).
- Alpha resolution order: per-tensor baked alpha (ComfyUI style) → `adapter_config.json` `lora_alpha`/`alpha` → fallback `alpha = rank` (scaling 1.0) with a loud warning when the config is missing.
- Unified GPU merge graph per tensor (`adapter_merge_on_backend`, :445): upload base in native quant from mmap → `ggml_cast` dequant to F32 on the backend → add the BF16-rounded delta (mirrors PEFT/LyCORIS numerics) → optional DoRA per-row rescale → encode back to native quant when the backend supports it (`adapter_backend_can_encode`, :304), else host-side `ggml_quantize_chunk`. NVFP4/MXFP4 lack a GPU cast → a split path promotes to BF16 instead of requantizing (host FP4 requant is ~60 s per model).
- `promote_f32 = true` (the default "merge"/merge_hq behavior): merged weights stay F32 to avoid catastrophic BF16 cancellation. It **skips non-layer tensors whose group scale is 0** (`adapter_hq_should_skip`, :709 — time_embed/cond_embed default to 0 in HQ mode). FP4 base models auto-disable promotion (`dit.h:549-558` — saves ~13 GB VRAM).
- **Multi-adapter accumulation:** each merge sources the base from the tensor's *current pending copy by name*, so a stack merges as `W ← W + s1·Δ1 + s2·Δ2 + …` sequentially (see comment at `dit.h:561-563`). Pre-permuted Conv1d (proj_in 3D→2D) is handled.
- Conv1d shape expansion: adapter delta `[out, in_ch]` vs base `[in_ch*P, out]` → delta tiled with `ggml_repeat` across the patch dimension (both merge and runtime paths).
- **Basin re-base** (`adapter_rebase_fetch`, :78; applied inside the merge before the delta add): `base ← base + β·(S − base)`, where S is the adapter's home-base safetensors (absolute path in `g_hotstep_params.rebase_source`). Applied to the FIRST stack adapter only (`dit.h:576-586`). Merge mode only; shape-preserving, so useless for cross-architecture transfer. Fatal failure signature if misused: at β=1, per-adapter application replaces the base each merge — only the last adapter survives.
- DoRA (`dora_scale` tensors) is **merge-only**: runtime mode cannot express a multiplicative per-row rescale as an additive delta. The runtime loader detects it and warns loudly (`adapter-runtime.h:394-405`); the UI warns too.

## 2. Runtime mode internals (`engine/src/adapter-runtime.h`)

- Deltas are precomputed on GPU per projection (`adapter_compute_delta`, shared with merge mode), staged host-side as F32, then finalized in one buffer alloc with parallel quantize + upload (`adapter_runtime_finalize`, :667).
- Storage type (`adapter_runtime_storage_type`, :136): `"q4_0"` (and legacy `"q4_k"`, aliased after `47fedfe`) → Q4_0; `"q8_0"` → Q8_0; both require `ne0 % 32 == 0`, otherwise BF16 fallback. Q4_0 ≈ ¼ the VRAM (~2.2 GB per XL adapter unquantized).
- Slot model (`dit_lora_slot`, :82): 32 layers × 11 projections (self-attn q/k/v/o, cross-attn q/k/v/o, mlp gate/up/down) + proj_in, cond_emb, and 6 time_embed tensors. Tensors without a slot are merge-only and skipped with an INFO log.
- **Stacking = delta sum:** `adapter_stage_delta` (:150) sums a later adapter's F32 delta into the already-staged tensor for the same projection. Per-step compute cost and VRAM are flat regardless of stack depth on this (non-section) path. Entry for stacks: `adapter_load_runtime_stack` (:748).
- Vulkan: the precision-rounding cast uses F16 instead of BF16 (no bf16→f32 shader on that backend).
- Cancel: the precompute loops check `adapter_cancel_requested()` between every delta (<100 ms cancel latency). The worker points `g_adapter_cancel` at the job's cancel flag around `ace_synth_load` with a scope guard (`hot-step-server.cpp`, ~:1214-1228). Job phase `ADAPTER_PRECOMPUTE` (= 7, `hot-step-server.cpp:263`) is reported during the ~17 s cold-start precompute.
- Graph application: `dit_ggml_linear_lora` (`dit-graph.h:45`) adds `delta@x` beside `W@x`. `mul_mat` dequantizes quantized deltas transparently; nothing else may touch delta bytes assuming BF16 (`46603bf`).

## 3. Cache-key recipe (exact)

`ModelKey` DiT extras (`model-store.h:90-102`), built in `pipeline-synth.cpp:168-199`:

- `adapter_path` (primary adapter), `adapter_scale`
- all six `adapter_group_scales` (self_attn, cross_attn, mlp, cond_embed, time_embed, proj_in — the last two were the `168dcb5` gap)
- `rebase_source` / `rebase_beta` — **merge mode only**; explicitly cleared in runtime mode
- `adapter_stack` string = `hotstep_adapter_stack_sig(...)` (`hot-step-params.h:211` — each path + `@%08x` float-bit-pattern of its scale), then:
  - `+ "|sect"` when `adapter_sections` is non-empty (per-section loads N separate deltas — distinct DiT; the section weights themselves live in per-frame masks, not the model, so they are NOT keyed)
  - `+ "|q:<quant>"` when runtime mode and `adapter_runtime_quant != "bf16"`
  - `+ "|mode:runtime"` when runtime mode with any adapter — a merge↔runtime toggle must never reuse the cached DiT

Hash: `model-store.cpp:48-69`. Equality: `model-store.cpp:84-94`. **Adding a key field requires touching all three sites.**

Load-failure rule: a runtime adapter load failure FAILS `dit_ggml_load` (`dit.h:648-658`), freeing the already-GPU-resident wctx (`dit_ggml_free`) because the store's failure path only does `delete m`. Per-section requires ALL stack adapters to load (`dit.h:624-626`) — a partial stack would silently generate with adapters missing and be cached that way.

## 4. Per-section masking — P1/P2 pipeline detail

- `AceRequest.adapter_sections` = `[{weights: number[], size: number}]`; `weights` indexed to stack order, `size` = section character count (frame-allocation hint). Built by `parseAdapterSections` (`server/src/services/generation/adapterSections.ts:113`).
- **P1** (`hot-step-sampler.h:306-323`): partition the S latent frames proportionally by section `size`; build per-adapter `[S]` float masks with a triangular crossfade of `xf = S/300` (~0.5 s; shortened from a longer fade in `ce57675` to reduce transition garble). `rebuild_section_masks` (:274) writes `lora_mask_host` and uploads.
- **P2** (`run_p2_alignment`, :338; invoked at :791 when `current_model_step == p2_align_step`): fires at `adapter_section_align_at` × num_steps (default 0.55; exposed in the UI as "Alignment Timing"; useful tuning range 0.3-0.4 per the lead). Estimates `x0 = xt − t·vt`, runs `dit_alignment_extract` on a **private scheduler** (never `dit->sched` — `4e48176`), obtains cross-attn `[enc_S, S]`, maps each frame to its dominant token, token → section via `g_hotstep_params.adapter_section_token_map`, median-smooths, then `rebuild_section_masks(..., "alignment")`. Any failure keeps the P1 proportional map and logs why.
- **Token→section map** (`pipeline-synth-ops.cpp:1419-1495`): scans lyric token text for `[...]` headers. Anchored when header count H == n_sections or H+1 == n_sections (server may emit a preamble section before the first header); otherwise falls back to char-proportional. Log line: `[Adapter-RT] P2: token→section map (... header-anchored|char-proportional ...)`.
- Masks must be forced to the backend in the scheduler; in NOMASK mode they must not be created at all (`34dce60`).
- Known limitation: cross-attn k/v and cond_emb use the scalar mean section weight (`dit-graph.h:430-433`), so text conditioning is a constant blend across the whole song — open issue.

## 5. Reverted: regional self-attention isolation

- `0f3bf6d` added a cross-section self-attn logit penalty (penalty = isolation × 8) to fight first-adapter bias. `ee041e1` reverted it: multi-second silence gaps between sections, degenerate later sections. Self-attention carries both adapter identity and musical coherence.
- `adapter_section_isolation` remains plumbed but dormant: `hot-step-params.h:137`, `ServerFields` parse `hot-step-server.cpp:573/:656`, `translateParams.ts:116` — the engine reads it into `g_hotstep_params` and then ignores it.
- Structurally safer untried idea on record: **section-aware CROSS-attention** — a token-axis mask on ca k/v + cond_emb using the existing token→section map. Cross-attn does not carry musical continuity, so it should not reproduce the silence-gap failure. Uncertain payoff: first-adapter bias may already be near the ceiling for weight-masking on a coherence-trained model. Try free knobs first: Alignment Timing 0.3-0.4, and raising the intended adapter's per-section directive weight to 1.2-1.4 in the lyric directive itself (e.g. `[Chorus]{myadapter=1.3}` — there is no separate "dominant section weight" param).

## 6. Cross-base / cross-arch conversion research history

Source: `docs/plans/cross-arch-adapter-conversion.md` (LOCAL-ONLY, gitignored — may be absent). Distilled here; items marked (unverified) were cited from that doc without opening the artifacts.

### Cross-base (same architecture, different base weights)

- **Ground truth (departing lead):** LoKR cross-base conversion to non-turbo XL bases FAILS despite ~99% weight-identity. Root cause: **basin-sensitivity** — the adapter delta lands in a different loss basin — NOT weight drift. Candidate fix: β·(S−T) basin nudge, designed but **NOT validated**.
- The nudge IS implemented and shipped (SKILL.md Golden rule 3; `W_merged = T + β·(S−T) + scale·ΔW`). The local doc claims the XL→XL sibling-base nudge was "validated by ear" via `bake_basin_nudge.py` — this **conflicts** with the lead's handoff. Treat as unvalidated; verify by ear. 2B→2B re-base: expected to work, untested.
- Re-base is shape-preserving → does nothing cross-architecture.

### Cross-arch (XL 32 layers/2560 hidden ↔ 2B 24 layers/2048 hidden) — UNSOLVED

- Six approaches failed. **Decisive finding:** even with excellent per-layer alignment (CCA on real inputs: CKA 0.90-0.99, fit residual 0.120), the baked model is garbled — independent per-layer linear transforms compound across 24 layers (~0.88²⁴ ≈ 5% end-to-end). **Any per-layer linear weight surgery (crop/pad, ridge alignment, CCA, Procrustes) is ruled out — do not retry.**
- Shared spaces that matter for future work: 64-channel output latent (both feed the same VAE), text-encoder 2048-dim, timestep embed 256, KV heads 8 × head_dim 128 (k/v output matches at 1024), `condition_embedder` / `time_embed.linear_1` as a "Rosetta Stone".
- **Distillation is unblocked:** the v1 non-convergence was an optimization bug (LR 2e-3 + single-sample SGD), not bf16 precision (measured bf16 noise = 4.5% of teacher delta). fp32 + per-sample-normalized loss + AdamW warmup/cosine reached rel error 0.09 by step 150 on an 8-tuple overfit. **Real, on-manifold conditioning inputs are mandatory** — synthetic Gaussian inputs are useless.
- Status (in progress, unverified artifacts): capture 384 real tuples (`capture_dit_inputs_v2.py`), rank-128 training (`distill_v3_train.py`), validate on held-out prompts. Scripts in `docs/plans/cross-arch-scripts/` (local-only, unverified). Real-input capture uses the Python sister project `D:\Ace-Step-Latest\hot-step-9000` (`ACESTEP_INIT_LLM=false`, `PYTHONIOENCODING=utf-8`) (unverified). Distilled students export as normal PEFT adapters — the engine already loads them.
- Ranked next directions: (A) learned paired converter from (XL, 2B) ground-truth adapter pairs; (B) distillation as a real training job; (C) conversion-as-init + light finetune. Unpursued: cross-model output-space guidance (run 2B + XL_adapted + XL_base each step, add `(XL_adapted − XL_base)` — zero training but 3 forward passes/step with both models resident).

## 7. Status ledger

**DONE / shipped:** merge + runtime modes (LoRA, LoKr, DoRA-in-merge); multi-adapter stacking (merge accumulation, runtime delta-sum, Sum/Blend UI, all trigger words injected); basin re-base (once-per-stack, merge-only); per-section masking P1 (proportional) + P2 (alignment, header-anchored token map); runtime delta quantization Q4_0/Q8_0 with parallel finalize; cooperative cancel; cache-key hardening + no-failure-caching (`168dcb5`); LM-echo rebuild fix (`8ea519b`).

**REVERTED:** regional self-attn isolation (`0f3bf6d` → `ee041e1`); `adapter_section_isolation` param dormant.

**PLANNED / open:** first-adapter bias mitigation via section-aware cross-attention (untried); on-the-fly low-rank runtime apply for 8+ adapter stacks (~40× VRAM saving, large engine change); 2B→2B re-base confirmation; cross-arch distillation v3 training run; per-adapter group scales / per-adapter re-base; multi-adapter presets.

**Commit archaeology (all hashes verified in git log):**

| Commit | What |
|---|---|
| `168dcb5` | Cache-key gaps (time_embed/proj_in group scales, runtime-mode marker), load-failure caching, sampler fixes |
| `8ea519b` | LM-echo adapter param plumbing (rebuild from `{...aceReq}`, no whitelist) |
| `ee041e1` / `0f3bf6d` | Revert / original self-attn isolation |
| `46603bf` | Removed BF16-assuming delta diagnostic (crashed on quantized deltas) |
| `a6db135` | Re-upload per-section masks every step (the "nil adapter" bug) |
| `2052bc3` | Re-upload per-section masks after CFG-cutoff graph rebuild |
| `4e48176` | Private scheduler for mid-sampling alignment (P2 CUDA crash) |
| `e09d6a3` | Scale scheduler hash-set with adapter count |
| `ce57675` | ~0.5 s per-section crossfade (transition garble) |
| `47fedfe` | Fast Q4_0 + parallel quantization for runtime delta VRAM saver |
| `34dce60` | NOMASK debug crash on unused mask input tensors |
