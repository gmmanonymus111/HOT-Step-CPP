#pragma once
// lm-train-run.h — `ace-train train-lm` stage driver (plan §3.5.3, §3.5.4).
//
// extract -> train -> export, each with its own teardown. The training loop is
// the hand-rolled one from §3.5: our own persistent grad accumulators driven by
// ggml_build_backward_expand, and our own merged norm+clip+AdamW graph.
//
// Emission contract (§2.2): `start` is the first JSONL line, `done` is the
// last; a `fatal` line replaces `done` and precedes a non-zero exit.

#include "backend.h"
#include "bpe.h"
#include "train/lm-ckpt.h"
#include "train/lm-common.h"
#include "train/lm-data.h"
#include "train/lm-export.h"
#include "train/lm-extract.h"
#include "train/lm-graph.h"
#include "train/lm-optim.h"
#include "train/lm-selftest.h"
#include "train/lm-vram.h"
#include "version.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

struct LmTrainArgs {
    std::vector<std::string> stages{ "extract", "train", "export" };

    std::string tensors_dir, codes_path, out_dir, models_dir;
    std::string dit_path, lm_path, lm_name, lm_size;  // already resolved by cmd_train_lm

    int   rank = 16, alpha = 32;
    float lr = 1e-4f;
    int   epochs = 75, grad_accum = 2;   // GA2 = Side-Step parity (GA4 halves optimizer steps/epoch)
    float warmup_ratio = 0.05f, grad_clip = 1.0f, weight_decay = 0.01f;
    int   seed         = 42;
    float target_loss  = 4.0f;   // parity: Side-Step's own electriccallboy run used 4.0 and stopped at epoch 29
    std::string order  = "shuffle";

    int  max_len         = 0;
    int  vram_reserve_mb = 1024;
    bool loss_on_cot     = true;

    // 4B low-VRAM path (2026-07-28 plan §2.1)
    std::string low_vram        = "auto";  // auto|on|off
    int         attn_head_block = -1;      // -1 = engine picks (lm_ckpt_default_head_block)
    int         chunk           = 128;

    // Speed levers (2026-07-28 plan §2.1). Both DEFAULT TO THE SHIPPED
    // BEHAVIOUR — §6.0 requires a flags-off run to be byte-identical.
    std::string weights = "f32-window";  // f32-window|bf16  (Lever A)
    std::string batch   = "1";           // 1..8|auto        (Lever B)

    // MUL_MAT activation-gradient formulation (engine/patches/mm-backward.patch):
    //   "outprod" = ggml upstream, out_prod(src0, transpose(grad)) — F32-only on CUDA
    //   "mm"      = mul_mat(cont(transpose(src0)), grad) — dtype-agnostic, BF16
    //               tensor cores. ~1.7-1.8x per layer per step on an RTX 5090.
    // ENGINE DEFAULT IS "outprod" so a bare ace-train invocation is unchanged; the
    // server passes --bwd mm. Selected by setting GGML_BACKWARD_MM before any
    // backward graph is built (ace-train.cpp).
    std::string bwd = "outprod";

    // Optimizer (2026-07-30, ported from the DiT trainer — lm-optim.h is shared,
    // so the rule split, batched Newton-Schulz and shape bucketing were already
    // sitting underneath this trainer unused). "adamw" is the default and the
    // shipped path. "muon" puts every 2-D parameter whose SHORT side is >=
    // muon_min_dim on orthogonalized-momentum updates.
    //
    // NOTE FOR LoRA: A is [H, r], so the SHORT SIDE IS THE RANK. At the default
    // rank 16 that is exactly muon_min_dim, and a rank-8 adapter would fall
    // entirely through to AdamW — lm_optim_init logs the split, so check it.
    std::string optimizer     = "adamw";
    float       muon_lr_scale = 1.0f;
    float       muon_momentum = 0.95f;
    int         muon_ns_steps = 5;
    bool        muon_nesterov = true;
    int         muon_min_dim  = 16;
    int         muon_bucket   = 16;

    // Trigger word embedded in the exported adapter. Empty = fall back to the
    // variant's preprocess_meta.json custom_tag/tag_position; still empty after
    // that = no trigger keys written and the adapter is byte-identical to a
    // pre-trigger build. docs/plans/2026-07-28-adapter-trigger-embedding.md T5
    std::string trigger, trigger_position;

    float milestone_step = 1.0f;
    int   milestone_keep = 6;

    bool overwrite = false;
    int  limit     = 0;
    bool self_test = false;
};

// Resolve the trigger to embed: explicit CLI flags win, else the variant's
// preprocess_meta.json, else nothing. `replace` positions are dropped with a
// warn (the tag was never applied to those captions — T4).
static void lm_resolve_trigger(const std::string & tensors_dir, std::string * trigger, std::string * position) {
    if (trigger->empty() && !tensors_dir.empty()) {
        lm_read_variant_tag(tensors_dir, trigger, position);
    }
    std::string why;
    if (!lm_trigger_normalize(trigger, position, &why) && !why.empty()) {
        jl("{\"type\":\"log\",\"level\":\"warn\",\"message\":\"%s\"}", lm_json_escape(why).c_str());
        fprintf(stderr, "[train-lm] %s\n", why.c_str());
    }
}

static inline bool lm_has_stage(const LmTrainArgs & a, const char * s) {
    for (size_t i = 0; i < a.stages.size(); i++) {
        if (a.stages[i] == s) {
            return true;
        }
    }
    return false;
}

// ─── the training stage ─────────────────────────────────────────────────────

struct LmTrainOutcome {
    int       epochs_run       = 0;
    double    final_loss       = -1.0;
    double    best_loss        = -1.0;
    int       best_epoch       = 0;
    bool      stopped_on_target = false;
    int       samples          = 0;
    int       skipped_long     = 0;
    bool      exported         = false;
    int       export_tensors   = 0;
    long long ms               = 0;
};

static int lm_train_stage(const LmTrainArgs & a, LmExportMeta * meta, LmTrainOutcome * out) {
    const int64_t t_stage0 = ggml_time_ms();
    jl("{\"type\":\"stage\",\"stage\":\"train\",\"state\":\"begin\",\"total\":%d}", a.epochs);

    // ── D11: the hard `hidden_size >= 2560 || n_layers >= 36` refusal is GONE.
    // Refusal is now purely the generic VRAM refusal against whichever footprint
    // model the selected mode uses; `fatal reason:"unsupported-size"` is retired.

    // ── load the LM, unfused (L18) ───────────────────────────────────────
    const int64_t t_lm0 = ggml_time_ms();
    g_qwen3_load_no_fuse = true;
    Qwen3LM    lm;
    const bool loaded    = qw3lm_load(&lm, a.lm_path.c_str(), /*max_seq_len=*/64, /*n_kv_sets=*/1);
    g_qwen3_load_no_fuse = false;
    if (!loaded) {
        lm_fatal("model-load", "cannot load the LM base " + a.lm_path);
        return 1;
    }
    const Qwen3LMConfig & c = lm.cfg;
    const int             H = c.hidden_size, V = c.vocab_size;
    jl("{\"type\":\"model\",\"stage\":\"lm\",\"path\":\"%s\",\"ms\":%lld,\"layers\":%d,\"hidden\":%d,\"vocab\":%d}",
       lm_json_escape(a.lm_path).c_str(), (long long) (ggml_time_ms() - t_lm0), c.n_layers, H, V);

    // ── mode selection (§3.2) ────────────────────────────────────────────
    //
    // DEVIATION vs §3.2's step order, with justification. The plan evaluates the
    // naive fit BEFORE the mirror exists, relying on the lemma that the mirror
    // term cancels out of the budget. That lemma is arithmetically right but its
    // inputs are not stable: `free` is a live CUDA reading that moves with every
    // other process on the card, and allocator padding makes
    // free_after_mirror + mirror_bytes only approximately free0 + base_bytes.
    // Gate (e) demands the naive maxLen INTEGER be unchanged, so instead of
    // re-deriving it we simply run the shipped code path untouched whenever the
    // mode could still be naive, and only switch afterwards. 4B (and
    // --low-vram on) skip the mirror entirely and never enter that path.
    const std::string size_label = lm_size_label_from_config(c);
    const bool        auto_mode  = (a.low_vram != "on" && a.low_vram != "off");
    LmVramMode        mode       = lm_vram_pick_mode(a.low_vram, size_label, /*naive_max_len=*/0);

    // ── Lever A gating (§3.4), before the mirror decision ────────────────
    //
    // bf16 requires BOTH the CUDA backend and a BF16-native base. Neither
    // condition used to have a graceful path (a fatal exit-1, or on CPU an
    // outright GGML_ABORT deep in the backward pass — lm-graph.h §property
    // 5 — since ggml_out_prod is F32-only on CUDA and aborts for BF16 on
    // CPU). Now that the server defaults `weights` to 'bf16' (2026-07-29)
    // and CPU/Vulkan release builds exist, both checks warn and fall back to
    // 'f32-window' instead — mirroring the DiT trainer's --mirror bf16 CUDA
    // gate (dit-train-run.h). `weights_used` (not `a.weights`) drives every
    // downstream report (vram JSONL, dit/lm_train_log.json) so a fallback is
    // never mislabelled as the bf16 run it wasn't.
    bool        weights_bf16 = (a.weights == "bf16");
    std::string weights_used = a.weights;
    if (weights_bf16) {
        std::string fallback_reason;
        if (strncmp(ggml_backend_name(lm.backend), "CUDA", 4) != 0) {
            // Gate BEFORE the graph is ever built: only ggml-cuda's out_prod
            // carries the BF16 patch (engine/patches/bf16-out-prod.patch);
            // CPU/Vulkan would GGML_ABORT mid-backward-pass instead of
            // failing cleanly.
            char b[192];
            snprintf(b, sizeof(b), "BF16 weights require CUDA — falling back to f32-window (this run picked '%s')",
                      ggml_backend_name(lm.backend));
            fallback_reason = b;
        } else if (!lm_bf16_base_is_bf16(lm)) {
            char b[256];
            snprintf(b, sizeof(b),
                     "BF16 weights require a BF16-native LM base — falling back to f32-window (%s loads its "
                     "projections as %s)",
                     a.lm_name.c_str(), lm_bf16_base_proj_type_name(lm));
            fallback_reason = b;
        }
        if (!fallback_reason.empty()) {
            lm_log("warn", fallback_reason);
            weights_bf16 = false;
            weights_used = "f32-window";
        }
    }
    if (weights_bf16) {
        // bf16 is meaningless on the naive path: it mirrors every weight to F32
        // and releases the BF16 buffer, so there is nothing left to run a BF16
        // GEMM on. --low-vram off + --weights bf16 was already rejected at exit
        // 2 in cmd_train_lm, so forcing the mode here can never contradict the
        // user. Only reached when bf16 survived both fallback checks above.
        mode = LM_VRAM_LOWVRAM;
    }

    size_t base_bytes = lm_base_weight_bytes(lm);

    // ── F32 mirror; a quantized base is refused here ─────────────────────
    LmF32Mirror mirror;
    if (mode == LM_VRAM_NAIVE) {
        std::string err;
        if (!lm_build_f32_mirror(&lm, &mirror, &err)) {
            lm_fatal("model-load", err);
            lm_mirror_free(&mirror);
            qw3lm_free(&lm);
            return 1;
        }
    } else {
        std::string err;
        if (!lm_ckpt_check_base(&lm, &err)) {
            lm_fatal("model-load", err);
            qw3lm_free(&lm);
            return 1;
        }
        fprintf(stderr, "[train-lm] low-vram mode: base stays %s resident (%.1f MB), no F32 mirror\n",
                ggml_type_name(lm.embed_tokens->type), base_bytes / 1048576.0);
    }

    // ── tokenizer ────────────────────────────────────────────────────────
    BPETokenizer bpe;
    if (!load_bpe_from_gguf(&bpe, a.lm_path.c_str())) {
        lm_fatal("model-load", "no BPE tokenizer in " + a.lm_path);
        lm_mirror_free(&mirror);
        qw3lm_free(&lm);
        return 1;
    }

    // ── VRAM auto-fit (L8) ───────────────────────────────────────────────
    LmVramModel vm;
    vm.n_layers     = c.n_layers;
    vm.hidden       = H;
    vm.ffn          = c.intermediate_size;
    vm.n_heads      = c.n_heads;
    vm.vocab        = V;
    vm.mirror_bytes = mirror.bytes;
    {
        size_t np = 0;
        for (int s = 0; s < QW_LORA_NSLOTS; s++) {
            int in_dim = 0, out_dim = 0;
            lm_slot_dims(c, s, &in_dim, &out_dim);
            np += (size_t) in_dim * (size_t) a.rank + (size_t) a.rank * (size_t) out_dim;
        }
        vm.lora_params = np * (size_t) c.n_layers;
    }

    int head_block = (a.attn_head_block >= 0) ? a.attn_head_block : lm_ckpt_default_head_block(c);
    if (!lm_ckpt_head_block_ok(c, head_block)) {
        char b[192];
        snprintf(b, sizeof(b), "--attn-head-block %d is not valid for n_heads %d / n_kv_heads %d — falling back to %d",
                 head_block, c.n_heads, c.n_kv_heads, lm_ckpt_default_head_block(c));
        lm_log("warn", b);
        head_block = lm_ckpt_default_head_block(c);
    }
    LmVramLowCfg lc;
    lc.attn_head_block = head_block;
    lc.chunk           = a.chunk;
    lc.head_dim        = c.head_dim;
    lc.base_bytes      = base_bytes;
    lc.emb_t_bytes     = (size_t) V * (size_t) H * ggml_type_size(lm.embed_tokens->type);
    lc.layer_w_bytes   = lm_layer_weight_bytes(c);
    lc.layer_wt_bytes  = lm_layer_proj_bytes(c, GGML_TYPE_BF16);  // §3.5
    lc.weights_bf16    = weights_bf16;

    LmVramFit fit;
    if (mode == LM_VRAM_NAIVE) {
        fit = lm_vram_fit(vm, lm.backend, a.vram_reserve_mb, a.max_len);
        // §3.2 step 6, second clause: auto turns low-vram ON for smaller bases
        // when the naive path would have to skip full-song samples. On this
        // machine the shipped fit yields ~3200 (0.6B) and ~2515 (1.7B), so this
        // branch is unreachable there and Rob's behaviour is unchanged.
        //
        // `a.max_len <= 0` is load-bearing: D5 words the rule as "the shipped
        // naive AUTO-FIT yields max_len < 2048", but lm_vram_fit() returns
        // `user_len` VERBATIM when the user pinned --max-len (lm-vram.h:97-102).
        // Without this guard any user value in [512, 2047] — a plain
        // Training-Studio field, always emitted by buildTrainLmArgs — silently
        // flipped 0.6B/1.7B onto the checkpointed path, violating D5's "Do
        // 0.6B/1.7B change? No." A pinned length is a user statement about
        // sequence length, not about VRAM.
        if (auto_mode && a.max_len <= 0 && fit.max_len < 2048) {
            char b[192];
            snprintf(b, sizeof(b), "naive auto-fit yields max_len %d (< 2048) — switching to low-VRAM mode",
                     fit.max_len);
            lm_log("info", b);
            lm_mirror_free(&mirror);
            qw3lm_free(&lm);
            g_qwen3_load_no_fuse = true;
            const bool re        = qw3lm_load(&lm, a.lm_path.c_str(), /*max_seq_len=*/64, /*n_kv_sets=*/1);
            g_qwen3_load_no_fuse = false;
            if (!re) {
                lm_fatal("model-load", "cannot re-load the LM base for low-VRAM mode " + a.lm_path);
                return 1;
            }
            std::string err;
            if (!lm_ckpt_check_base(&lm, &err)) {
                lm_fatal("model-load", err);
                qw3lm_free(&lm);
                return 1;
            }
            base_bytes      = lm_base_weight_bytes(lm);
            lc.base_bytes   = base_bytes;
            vm.mirror_bytes = 0;
            mode            = LM_VRAM_LOWVRAM;
        }
    }
    if (mode == LM_VRAM_LOWVRAM) {
        fit = lm_vram_fit_lowvram(vm, lc, lm.backend, a.vram_reserve_mb, a.max_len);
    }

    if (!fit.ok) {
        char extra[224];
        snprintf(extra, sizeof(extra), ",\"needMb\":%lld,\"freeMb\":%lld,\"lmSize\":\"%s\",\"mode\":\"%s\"",
                 (long long) (fit.est_bytes / 1048576.0), (long long) fit.free_mb, a.lm_size.c_str(),
                 mode == LM_VRAM_LOWVRAM ? "lowvram" : "naive");
        lm_fatal("vram",
                 "not enough free VRAM for LM training: need ~" + std::to_string((long long) (fit.est_bytes / 1048576.0)) +
                     " MB, " + std::to_string((long long) fit.free_mb) + " MB free",
                 extra);
        lm_mirror_free(&mirror);
        qw3lm_free(&lm);
        return 1;
    }
    const int max_len = fit.max_len;

    // ── samples ──────────────────────────────────────────────────────────
    std::vector<LmCodeRow> rows;
    {
        std::string err;
        if (!lm_load_codes(a.codes_path.c_str(), &rows, &err)) {
            lm_fatal("no-samples", err);
            lm_mirror_free(&mirror);
            qw3lm_free(&lm);
            return 1;
        }
    }
    if (a.limit > 0 && (int) rows.size() > a.limit) {
        rows.resize((size_t) a.limit);
    }

    std::vector<LmSample> samples;
    int                   skipped_long = 0;  // §2.2: OVER-LENGTH only (the L5 skip)
    int                   skipped_bad  = 0;  // structural rejections — malformed rows
    int                   min_len = 0, max_seq = 0;
    long long             trained_tokens = 0;
    for (size_t i = 0; i < rows.size(); i++) {
        LmSample    s;
        std::string why;
        bool        over = false;
        if (!lm_build_sequence(bpe, rows[i], a.loss_on_cot, max_len, &s, &why, &over)) {
            // Counting a malformed row as "too long" would point the user at
            // --max-len instead of at the data.
            if (over) {
                skipped_long++;
            } else {
                skipped_bad++;
            }
            lm_log("warn", "SKIP " + rows[i].file + ": " + why);
            continue;
        }
        const int S = (int) s.tokens.size();
        if (samples.empty() || S < min_len) {
            min_len = S;
        }
        if (S > max_seq) {
            max_seq = S;
        }
        trained_tokens += s.s_tr;
        samples.push_back(s);
    }
    if (samples.empty()) {
        lm_fatal("no-samples", "no usable training samples (every song was skipped; max_len " +
                                   std::to_string(max_len) + ")");
        lm_mirror_free(&mirror);
        qw3lm_free(&lm);
        return 1;
    }

    int s_tr_max = 0;
    for (size_t i = 0; i < samples.size(); i++) {
        s_tr_max = std::max(s_tr_max, samples[i].s_tr);
    }

    const int n            = (int) samples.size();
    const int steps_per_ep = (n + a.grad_accum - 1) / a.grad_accum;
    const int total_steps  = std::max(1, steps_per_ep * a.epochs);

    // lm_lr_lambda(0, ...) is exactly 0 by design (L6b), so a warmup that covers
    // every optimizer step trains NOTHING: AdamW updates only m/v, B stays 0 and
    // the exported adapter is an identity. The plan's `max(1, ...)` does exactly
    // that whenever total_steps == 1 (`--epochs 1` with n <= grad_accum, the
    // natural quick-test invocation) and also makes `--warmup-ratio 0`
    // inoperable. Hence the half-run cap below, and ratio 0 == "no warmup".
    //
    // The 50-step FLOOR mirrors dit-train-run.h (kept in lockstep deliberately):
    // percentage-only warmup is a hyperparameter-porting hazard, because the
    // reference recipes these learning rates come from (Side-Step) warm up over
    // a fixed ~50 of ~400 optimizer steps. With our smaller effective batch, 5 %
    // compressed to ~7 steps on a short run and the LR reached full scale before
    // the adapter had settled — reproducibly blowing up three DiT LoKR runs at
    // the end of the ramp (2026-07-29).
    int warmup_steps = 0;
    if (a.warmup_ratio > 0.0f) {
        warmup_steps = std::max(50, (int) ((double) total_steps * (double) a.warmup_ratio));
        warmup_steps = std::min(warmup_steps, total_steps / 2);
        if (warmup_steps < 1) {
            warmup_steps = 0;  // only reachable at total_steps == 1
        }
    }

    // ── vram + data events ───────────────────────────────────────────────
    // estMb is the predicted PEAK, i.e. the footprint at the LONGEST ACCEPTED
    // sequence — not at maxLen, which is only the skip threshold. That is the
    // quantity G4 compares against the observed peak.
    const bool   low  = (mode == LM_VRAM_LOWVRAM);
    const double est_bytes =
        low ? lm_vram_bytes_lowvram(vm, lc, max_seq, s_tr_max) : lm_vram_bytes(vm, max_seq, s_tr_max);
    const double ckpt_bytes = low ? (double) c.n_layers * (double) H * (double) max_seq * 4.0 : 0.0;
    const double seg_bytes  = low ? lm_vram_lowvram_transient(vm, lc, max_seq) : 0.0;

    // §2.2: five additive fields. In "naive" mode they read "naive", the mirror
    // size and three zeros, so the event stays byte-compatible with [P] §2.9.
    jl("{\"type\":\"vram\",\"freeMb\":%lld,\"totalMb\":%lld,\"reserveMb\":%d,\"mirrorMb\":%lld,\"maxLen\":%d,"
       "\"estMb\":%lld,\"source\":\"%s\",\"mode\":\"%s\",\"baseMb\":%lld,\"ckptMb\":%lld,\"segPeakMb\":%lld,"
       "\"attnHeadBlock\":%d,\"chunk\":%d,\"weights\":\"%s\",\"batch\":%d,\"batchSource\":\"%s\"}",
       (long long) fit.free_mb, (long long) fit.total_mb, a.vram_reserve_mb,
       (long long) (low ? 0 : mirror.bytes / 1048576), max_len, (long long) (est_bytes / 1048576.0),
       a.max_len > 0 ? "user" : "auto", low ? "lowvram" : "naive",
       (long long) ((low ? (double) base_bytes : (double) mirror.bytes) / 1048576.0),
       (long long) (ckpt_bytes / 1048576.0), (long long) (seg_bytes / 1048576.0), low ? lc.attn_head_block : 0,
       low ? lc.chunk : 0,
       // §2.2: three additive fields. In a default run they read
       // "f32-window", 1, "user", so the event stays byte-compatible in meaning
       // and [P] §2.9's SSE mapping needs no change. weights_used (not
       // a.weights) so a CUDA/base-dtype fallback reports "f32-window", the
       // mode actually run, not the "bf16" the caller requested.
       weights_used.c_str(), 1, "user");

    // `skippedBad` is additive (§2.2: consumers ignore unknown fields) and keeps
    // `skippedLong` meaning what its name says.
    jl("{\"type\":\"data\",\"samples\":%d,\"skippedLong\":%d,\"skippedBad\":%d,\"minLen\":%d,\"maxLen\":%d,"
       "\"maxLenCap\":%d,\"trainedTokens\":%lld,\"stepsPerEpoch\":%d,\"totalSteps\":%d,\"warmupSteps\":%d,"
       "\"loraParams\":%lld,\"batches\":%d,\"padTokens\":%lld,\"padPct\":%.1f}",
       n, skipped_long, skipped_bad, min_len, max_seq, max_len, trained_tokens, steps_per_ep, total_steps,
       warmup_steps, (long long) vm.lora_params,
       // §2.2: at --batch 1 there is one batch per sample and no padding at all,
       // so these read `samples`, 0 and 0.0 — the honest cost side of Lever B's
       // ledger, reported even when the lever is off.
       n, (long long) 0, 0.0);
    if (skipped_bad > 0) {
        char sb[128];
        snprintf(sb, sizeof(sb), "%d song(s) rejected for malformed prompt/codes — not a length problem", skipped_bad);
        lm_log("warn", sb);
    }

    // ── persistent allocations (§3.5.1) ──────────────────────────────────
    ggml_context * ctx_static = nullptr;
    {
        ggml_init_params p = { 32 * ggml_tensor_overhead(), nullptr, true };
        ctx_static         = ggml_init(p);
    }
    ggml_tensor * t_tok = ggml_new_tensor_1d(ctx_static, GGML_TYPE_I32, max_seq);
    ggml_tensor * t_pos = ggml_new_tensor_1d(ctx_static, GGML_TYPE_I32, max_seq);
    ggml_tensor * t_msk = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, (int64_t) max_seq * max_seq);
    // The [V, s_tr_max] label buffer is a naive-path structure: 1,184 MiB at
    // 4B / s_tr 1556. Low-vram sparse-writes a [V, chunk] buffer instead (D4).
    ggml_tensor * t_lab      = low ? nullptr : ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, V, s_tr_max);
    ggml_tensor * t_adamw    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 7);
    ggml_tensor * t_lossgrad = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_clip     = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_eps      = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_gnorm2   = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_gs       = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_one      = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_set_input(t_tok);
    ggml_set_input(t_pos);
    ggml_set_input(t_msk);
    if (t_lab) {
        ggml_set_input(t_lab);
    }

    ggml_backend_buffer_t buf_static = ggml_backend_alloc_ctx_tensors(ctx_static, lm.backend);
    if (!buf_static) {
        lm_fatal("vram", "static input buffer allocation failed");
        lm_mirror_free(&mirror);
        qw3lm_free(&lm);
        return 1;
    }
    ggml_backend_buffer_clear(buf_static, 0);

    {  // constants uploaded once
        const float lg = 1.0f / (float) a.grad_accum;  // == Side-Step's loss/(n_tok*grad_accum)
        const float cl = a.grad_clip;
        const float ep = 1e-6f;
        // D9: the low-vram trunk surrogate's loss gradient is EXACTLY 1.0 —
        // 1/grad_accum is already folded into the per-chunk `gs`. Using
        // t_lossgrad here would scale every gradient by 1/grad_accum^2.
        const float on = 1.0f;
        ggml_backend_tensor_set(t_lossgrad, &lg, 0, sizeof(float));
        ggml_backend_tensor_set(t_clip, &cl, 0, sizeof(float));
        ggml_backend_tensor_set(t_eps, &ep, 0, sizeof(float));
        ggml_backend_tensor_set(t_one, &on, 0, sizeof(float));
    }

    LmLora lora;
    {
        std::string err;
        if (!lm_lora_init(&lora, &lm, 0, c.n_layers, a.rank, (float) a.alpha, (uint64_t) a.seed, /*b_sigma=*/0.0f,
                          &err)) {
            lm_fatal("vram", err);
            return 1;
        }
    }
    if (lora.n_params != vm.lora_params) {
        char b[192];
        snprintf(b, sizeof(b), "LoRA parameter count %zu != predicted %zu", lora.n_params, vm.lora_params);
        lm_log("warn", b);
    }

    LmOptim opt;
    {
        std::string err;
        // BEFORE init: the rule split and the optimizer-state allocation are both
        // decided there (a Muon parameter gets no second momentum buffer).
        opt.optimizer     = a.optimizer;
        opt.muon.lr_scale = a.muon_lr_scale;
        opt.muon.momentum = a.muon_momentum;
        opt.muon.ns_steps = a.muon_ns_steps;
        opt.muon.nesterov = a.muon_nesterov;
        opt.muon.min_dim  = a.muon_min_dim;
        opt.muon.bucket   = a.muon_bucket;
        if (!lm_optim_init(&opt, lora.params, lm.backend, &err)) {
            lm_fatal("vram", err);
            return 1;
        }
    }
    opt.t_adamw      = t_adamw;
    opt.t_lossgrad   = t_lossgrad;
    opt.t_clip       = t_clip;
    opt.t_eps        = t_eps;
    opt.t_gnorm2     = t_gnorm2;
    opt.base_lr      = a.lr;
    opt.weight_decay = a.weight_decay;
    opt.grad_clip    = a.grad_clip;
    opt.total_steps  = total_steps;
    opt.warmup_steps = warmup_steps;

    // ── low-vram persistent state (§3.3) ─────────────────────────────────
    LmCkptState ckpt;
    LmCkptRun   run;
    if (low) {
        LmCkptCfg cc;
        cc.chunk           = lc.chunk;
        cc.attn_head_block = lc.attn_head_block;
        cc.s_max           = max_seq;
        cc.layer_lo        = 0;
        cc.layer_hi        = c.n_layers;
        cc.weights_bf16    = weights_bf16;  // Lever A
        std::string err;
        if (!lm_ckpt_alloc(&ckpt, &lm, cc, &err) || !lm_ckpt_build_embed_t(&ckpt, &err)) {
            lm_fatal("vram", err.empty() ? std::string("low-vram allocation failed") : err);
            return 1;
        }
        run.lm         = &lm;
        run.opt        = &opt;
        run.st         = &ckpt;
        run.t_tok      = t_tok;
        run.t_pos      = t_pos;
        run.t_msk      = t_msk;
        run.t_gs       = t_gs;
        run.t_one      = t_one;
        run.grad_accum = a.grad_accum;
    }

    // ── graph arena + scheduler sized from a real node count ─────────────
    std::vector<uint8_t> arena((size_t) 128 << 20);

    int  last_mask_S = 0;
    auto upload_mask = [&](int S) {
        if (S == last_mask_S) {
            return;
        }
        std::vector<float> m;
        lm_causal_mask(S, &m);
        ggml_backend_tensor_set(t_msk, m.data(), 0, m.size() * sizeof(float));
        last_mask_S = S;
    };

    // Build (but do not run) the largest graph to size the scheduler.
    int graph_nodes = 0, graph_leafs = 0;
    if (low) {
        // The worst low-vram graph is one backward segment at S = max_seq: the
        // trunk is never built whole, so sizing from it would over-allocate the
        // sched by ~L x. Reuse of a single sched across ~110 graph computes per
        // micro-step is what keeps the allocator arena stable (T13).
        upload_mask(max_seq);
        run.sched   = nullptr;
        graph_nodes = lm_ckpt_probe_segment_nodes(run, max_seq);
        graph_leafs = 0;
    } else {
        ggml_init_params ip  = { arena.size(), arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 65536, /*grads=*/true);

        const LmSample & s      = samples[0];
        ggml_tensor *    hidden = lm_build_trunk(ctx, &lm, t_tok, t_pos, t_msk, max_seq);
        ggml_tensor *    hd     = ggml_cont(
            ctx, ggml_view_2d(ctx, hidden, H, s.s_tr, hidden->nb[1], (size_t) (s.n_masked - 1) * hidden->nb[1]));
        ggml_tensor * logits = ggml_mul_mat(ctx, lm.embed_tokens, hd);
        ggml_tensor * labv   = ggml_view_2d(ctx, t_lab, V, s.s_tr, t_lab->nb[1], 0);
        ggml_tensor * loss   = ggml_cross_entropy_loss(ctx, logits, labv);
        ggml_set_loss(loss);
        ggml_set_output(loss);
        ggml_build_forward_expand(gf, loss);
        std::vector<ggml_tensor *> gacc;
        lm_optim_fill_gacc(&opt, gf, &gacc);
        ggml_build_backward_expand(ctx, gf, gacc.data());
        graph_nodes = ggml_graph_n_nodes(gf);
        graph_leafs = 0;  // ggml_cgraph is opaque; size the sched from nodes + headroom
        ggml_free(ctx);
    }
    fprintf(stderr, "[train-lm] %s graph: %d nodes\n", low ? "worst segment fwd+bwd" : "fwd+bwd", graph_nodes);

    BackendPair bp;
    bp.backend     = lm.backend;
    bp.cpu_backend = lm.cpu_backend;
    bp.has_gpu     = lm.backend != lm.cpu_backend;
    ggml_backend_sched_t sched = backend_sched_new(bp, std::max(8192, graph_nodes + graph_nodes / 2 + 2048));
    if (low) {
        run.sched = sched;
    }

    // ── one micro-step ───────────────────────────────────────────────────
    LmVramTracker tracker;
    // The shipped naive lambda, MOVED but not edited.
    auto micro_step_naive = [&](const LmSample & s, bool count_loss, double * ce_out) -> bool {
        const int S    = (int) s.tokens.size();
        const int s_tr = s.s_tr;

        upload_mask(S);
        ggml_backend_tensor_set(t_tok, s.tokens.data(), 0, (size_t) S * 4);
        {
            std::vector<int32_t> ip((size_t) S);
            for (int i = 0; i < S; i++) {
                ip[(size_t) i] = i;
            }
            ggml_backend_tensor_set(t_pos, ip.data(), 0, (size_t) S * 4);
        }

        ggml_init_params gip = { arena.size(), arena.data(), true };
        ggml_context *   ctx = ggml_init(gip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 65536, /*grads=*/true);

        ggml_tensor * hidden = lm_build_trunk(ctx, &lm, t_tok, t_pos, t_msk, S);
        ggml_tensor * hd =
            ggml_cont(ctx, ggml_view_2d(ctx, hidden, H, s_tr, hidden->nb[1], (size_t) (s.n_masked - 1) * hidden->nb[1]));
        ggml_tensor * logits = ggml_mul_mat(ctx, lm.embed_tokens, hd);  // [V, s_tr] (tied head)
        ggml_tensor * labv   = ggml_view_2d(ctx, t_lab, V, s_tr, t_lab->nb[1], 0);
        ggml_tensor * loss   = ggml_cross_entropy_loss(ctx, logits, labv);
        ggml_set_loss(loss);
        ggml_set_output(loss);
        ggml_build_forward_expand(gf, loss);

        std::vector<ggml_tensor *> gacc;
        lm_optim_fill_gacc(&opt, gf, &gacc);
        ggml_build_backward_expand(ctx, gf, gacc.data());

        bool ok = false;
        {
            LmLabelGuard guard(t_lab, s.targets.data(), s_tr, V);
            ggml_backend_sched_reset(sched);
            ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
            if (ok && count_loss) {
                float ce = 0.0f;
                ggml_backend_tensor_get(loss, &ce, 0, sizeof(float));
                *ce_out = (double) ce;
            }
        }
        ggml_free(ctx);
        return ok;
    };

    auto micro_step = [&](const LmSample & s, bool count_loss, double * ce_out) -> bool {
        return low ? lm_ckpt_micro_step(run, s, count_loss, ce_out) : micro_step_naive(s, count_loss, ce_out);
    };

    // ── high-water probe (§3.7) ──────────────────────────────────────────
    //
    // The probe must bound the run's PEAK allocation, not merely its longest
    // sequence. The two biggest transients — `logits` and its gradient, each
    // V*s_tr*4 (1.66 MB per trained token at 0.6B) — scale with s_tr, and
    // argmax(S) != argmax(s_tr) in general because n_masked (the caption+lyrics
    // prompt length) varies independently of the code count. Probing only the
    // longest sequence lets ggml_gallocr grow mid-epoch, which transiently holds
    // the old and new arenas at once — exactly the "40 songs in, OOM" §3.7 says
    // the probe exists to prevent — and it invalidates the L10 leak baseline.
    //
    // §3.7 already sanctions synthetic tokens, so probe the JOINT worst case
    // (max_seq, s_tr_max) even if no single real sample exhibits it.
    {
        size_t longest = 0;
        for (size_t i = 1; i < samples.size(); i++) {
            if (samples[i].tokens.size() > samples[longest].tokens.size()) {
                longest = i;
            }
        }
        LmSample probe = samples[longest];            // real token ids, longest length
        probe.n_masked = max_seq - s_tr_max;          // >= 1: every sample has s_tr <= S-1
        probe.s_tr     = s_tr_max;
        GGML_ASSERT(probe.n_masked >= 1 && (int) probe.tokens.size() == max_seq);
        probe.targets.assign(probe.tokens.begin() + probe.n_masked, probe.tokens.end());
        GGML_ASSERT((int) probe.targets.size() == s_tr_max);

        double dummy = 0.0;
        if (!micro_step(probe, false, &dummy)) {
            lm_fatal("vram", "the high-water probe failed to run — not enough VRAM for the worst-case sequence");
            return 1;
        }
        lm_optim_zero_grad(&opt);

        // Trainer-owned footprint: the buffers we allocated plus the scheduler's
        // own compute arena. Device-wide (total - free) would fold in every other
        // process on the card and is not comparable with estMb.
        size_t fixed = low ? (base_bytes + ckpt.fixed_bytes())
                           : (mirror.buf ? ggml_backend_buffer_get_size(mirror.buf) : mirror.bytes);
        fixed += ggml_backend_buffer_get_size(buf_static);
        if (lora.buf) {
            fixed += ggml_backend_buffer_get_size(lora.buf);
        }
        if (opt.buf_grad) {
            fixed += ggml_backend_buffer_get_size(opt.buf_grad);
        }
        if (opt.buf_mom) {
            fixed += ggml_backend_buffer_get_size(opt.buf_mom);
        }
        tracker.probe_baseline(lm.backend, sched, fixed);
        fprintf(stderr,
                "[train-lm] high-water probe at S=%d s_tr=%d: trainer VRAM %zu MB (est %lld MB, device %zu MB)\n",
                max_seq, s_tr_max, tracker.base_mb, (long long) (est_bytes / 1048576.0),
                lm_vram_used_mb(lm.backend));
    }

    // A previous run into the same <out> leaves its milestone dirs behind; they
    // are absent from this run's log and would be offered by the UI picker.
    lm_milestone_reset(a.out_dir);

    // ── epoch loop (§3.5.4) ──────────────────────────────────────────────
    meta->max_len        = max_len;
    meta->max_len_source = a.max_len > 0 ? "user" : "auto";
    meta->samples        = n;
    meta->skipped_long   = skipped_long;
    meta->vram_free_mb   = fit.free_mb;
    meta->vram_total_mb  = fit.total_mb;
    meta->vram_mirror_mb = mirror.bytes / 1048576;
    meta->vram_est_mb    = (size_t) (est_bytes / 1048576.0);

    meta->low_vram        = low;
    meta->attn_head_block = low ? lc.attn_head_block : 0;
    meta->chunk           = low ? lc.chunk : 0;
    meta->weights         = weights_used;   // §2.3 — recorded so a resume can refuse (S6); ACTUAL mode, not requested
    meta->batch           = 1;
    meta->vram_mode       = low ? "lowvram" : "naive";
    meta->vram_base_mb    = (size_t) ((low ? (double) base_bytes : (double) mirror.bytes) / 1048576.0);
    meta->vram_ckpt_mb    = (size_t) (ckpt_bytes / 1048576.0);
    meta->vram_seg_peak_mb = (size_t) (seg_bytes / 1048576.0);

    double    ladder      = 0.0;
    bool      ladder_seed = false;
    long long ep_ms_sum   = 0;
    int       global_step = 0;
    std::string export_err;
    int       rc = 0;

    for (int epoch = 0; epoch < a.epochs; epoch++) {
        const int64_t t_ep0 = ggml_time_ms();
        std::vector<int> order;
        lm_epoch_order(&order, n, a.order != "fixed", (uint64_t) a.seed, epoch);

        double running = 0.0;
        int    n_micro = 0;
        lm_optim_zero_grad(&opt);
        LmStepStats last_stats;

        // §2.2 defines `step.micro` as the micro-batches folded into THIS
        // optimizer step and `step.ms` as that step's own wall time. Both need
        // window-local state: the epoch accumulators would report a cumulative
        // count and a smoothed epoch mean, and timing from inside the sample loop
        // measures only the final micro-batch (~1/grad_accum of the truth).
        double  win_loss  = 0.0;
        int     win_micro = 0;
        int64_t t_win0    = ggml_time_ms();

        for (int j = 0; j < n; j++) {
            double ce = 0.0;
            if (!micro_step(samples[(size_t) order[(size_t) j]], true, &ce)) {
                lm_fatal("vram", "graph compute failed mid-epoch");
                rc = 1;
                break;
            }
            running += ce;
            n_micro++;
            win_loss += ce;
            win_micro++;

            if ((j + 1) % a.grad_accum == 0 || (j + 1) == n) {
                if (!lm_optim_step(&opt, sched, &last_stats)) {
                    lm_fatal("vram", "optimizer step failed");
                    rc = 1;
                    break;
                }
                global_step++;
                const size_t vram_mb = tracker.sample();
                jl("{\"type\":\"step\",\"epoch\":%d,\"step\":%d,\"totalSteps\":%d,\"micro\":%d,\"loss\":%.6f,"
                   "\"lr\":%.9g,\"gradNorm\":%.6f,\"clipScale\":%.6f,\"ms\":%lld,\"vramMb\":%lld,"
                   "\"samples\":%d}",
                   epoch + 1, global_step, total_steps, win_micro, win_loss / std::max(1, win_micro),
                   (double) last_stats.lr, (double) last_stats.grad_norm, (double) last_stats.clip,
                   (long long) (ggml_time_ms() - t_win0), (long long) vram_mb,
                   // §4.8: samples == micro * B_cur summed over the window. At
                   // B == 1 that is exactly `micro`.
                   win_micro);
                win_loss  = 0.0;
                win_micro = 0;
                t_win0    = ggml_time_ms();
            }
        }
        if (rc != 0) {
            break;
        }

        const double avg   = running / std::max(1, n_micro);
        const long long ems = (long long) (ggml_time_ms() - t_ep0);
        ep_ms_sum += ems;

        const bool best = (out->best_loss < 0.0) || (avg < out->best_loss);
        if (best) {
            out->best_loss  = avg;
            out->best_epoch = epoch + 1;
        }
        out->final_loss = avg;
        out->epochs_run = epoch + 1;

        LmEpochRec rec;
        rec.epoch     = epoch + 1;
        rec.loss      = avg;
        rec.lr        = last_stats.lr;
        rec.grad_norm = last_stats.grad_norm;
        rec.ms        = ems;
        meta->epoch_log.push_back(rec);

        // Export EVERY epoch, BEFORE the stop test — the checkpoint that
        // triggered the stop is already on disk (train_lm.py:325,333), and a
        // cancelled/crashed run still leaves a usable adapter.
        meta->epochs_run = out->epochs_run;
        meta->final_loss = out->final_loss;
        meta->best_loss  = out->best_loss;
        meta->best_epoch = out->best_epoch;
        {
            LmExportResult xr;
            if (!lm_export_peft(lora, c, *meta, a.out_dir, &xr, &export_err)) {
                lm_fatal("export", export_err);
                rc = 1;
                break;
            }
            out->exported       = true;
            out->export_tensors = xr.tensors;
        }
        // milestones (L20)
        if (a.milestone_step > 0.0f) {
            if (!ladder_seed) {
                ladder      = floor(avg / (double) a.milestone_step) * (double) a.milestone_step;
                ladder_seed = true;
            }
            while (avg <= ladder + 1e-12) {
                const std::string label = lm_fmt1(ladder);          // "7.3"
                const double      lval  = atof(label.c_str());       // exactly what the dir says
                const std::string rel   = "milestones/loss_" + label;
                const std::string mdir  = lm_join(a.out_dir, rel);
                LmExportResult    mr;
                std::string       merr;
                if (lm_export_peft(lora, c, *meta, mdir, &mr, &merr)) {
                    LmMilestoneRec ms;
                    ms.loss  = lval;
                    ms.epoch = epoch + 1;
                    ms.path  = rel;
                    meta->milestones.push_back(ms);
                    jl("{\"type\":\"milestone\",\"loss\":%.4g,\"epoch\":%d,\"path\":\"%s\",\"bytes\":%lld}", lval,
                       epoch + 1, lm_json_escape(mdir).c_str(), mr.bytes);
                    lm_milestone_prune(a.out_dir, &meta->milestones, a.milestone_keep);
                } else {
                    lm_log("warn", "milestone snapshot failed: " + merr);
                }
                ladder -= (double) a.milestone_step;
            }
        }

        // §3.5.4 exports every epoch so a cancelled/crashed run leaves a usable
        // adapter — but the server reads lm_train_log.json out of <out> (§4.3),
        // so the log has to land alongside it, not only at the end of the export
        // stage. Written after the milestone block so it lists this epoch's
        // snapshots. `total_ms` is the stage elapsed here; the export stage
        // overwrites it with the whole-run figure.
        meta->vram_peak_mb = tracker.peak_mb;
        meta->total_ms     = (long long) (ggml_time_ms() - t_stage0);
        if (!lm_write_train_log(a.out_dir, *meta)) {
            lm_log("warn", "cannot write lm_train_log.json in " + a.out_dir);
        }

        const long long eta = (long long) ((double) ep_ms_sum / (double) (epoch + 1) * (double) (a.epochs - epoch - 1));
        jl("{\"type\":\"epoch\",\"epoch\":%d,\"epochs\":%d,\"loss\":%.6f,\"lr\":%.9g,\"gradNorm\":%.6f,\"ms\":%lld,"
           "\"etaMs\":%lld,\"best\":%s}",
           epoch + 1, a.epochs, avg, (double) last_stats.lr, (double) last_stats.grad_norm, ems, eta,
           best ? "true" : "false");
        jl("{\"type\":\"progress\",\"completed\":%d,\"total\":%d,\"phase\":\"train\"}", epoch + 1, a.epochs);
        fprintf(stderr, "[train-lm] epoch %d/%d loss=%.6f lr=%.3e gnorm=%.3f (%lld ms)\n", epoch + 1, a.epochs, avg,
                (double) last_stats.lr, (double) last_stats.grad_norm, ems);

        if (a.target_loss > 0.0f && avg <= (double) a.target_loss) {
            out->stopped_on_target  = true;
            meta->target_stop       = true;
            meta->target_stop_epoch = epoch + 1;
            meta->target_stop_loss  = avg;
            jl("{\"type\":\"target_stop\",\"epoch\":%d,\"loss\":%.6f,\"targetLoss\":%.6g}", epoch + 1, avg,
               (double) a.target_loss);
            break;
        }
    }

    meta->vram_peak_mb = tracker.peak_mb;
    out->samples       = n;
    out->skipped_long  = skipped_long;
    out->ms            = (long long) (ggml_time_ms() - t_stage0);

    char b[224];
    snprintf(b, sizeof(b), "leak counter: baseline %zu MB, peak %zu MB, max delta %lld MB over %d optimizer steps",
             tracker.base_mb, tracker.peak_mb, tracker.max_delta, global_step);
    fprintf(stderr, "[train-lm] %s\n", b);
    jl("{\"type\":\"leak\",\"baselineMb\":%lld,\"peakMb\":%lld,\"deltaMb\":%lld,\"steps\":%d}",
       (long long) tracker.base_mb, (long long) tracker.peak_mb, tracker.max_delta, global_step);

    if (rc == 0) {
        jl("{\"type\":\"stage\",\"stage\":\"train\",\"state\":\"end\",\"epochsRun\":%d,\"finalLoss\":%.6f,"
           "\"bestLoss\":%.6f,\"stoppedOnTarget\":%s,\"ms\":%lld}",
           out->epochs_run, out->final_loss, out->best_loss, out->stopped_on_target ? "true" : "false", out->ms);
    }

    // teardown (free the GPU before the export stage does its file writes)
    ggml_backend_sched_free(sched);
    lm_optim_free(&opt);
    lm_lora_detach(&lora, &lm);
    lm_lora_free(&lora);
    lm_ckpt_free(&ckpt);
    ggml_backend_buffer_free(buf_static);
    ggml_free(ctx_static);
    lm_mirror_free(&mirror);
    qw3lm_free(&lm);
    return rc;
}

// ─── main entry ─────────────────────────────────────────────────────────────

static int lm_train_main(const LmTrainArgs & a) {
    if (a.self_test) {
        return lm_self_test(a.lm_path, a.codes_path, (uint64_t) a.seed);
    }

    const int64_t t_run0 = ggml_time_ms();

    std::string stage_csv;
    for (size_t i = 0; i < a.stages.size(); i++) {
        if (i) {
            stage_csv += ",";
        }
        stage_csv += "\"" + a.stages[i] + "\"";
    }

    jl("{\"type\":\"start\",\"stages\":[%s],\"tensors\":\"%s\",\"codes\":\"%s\",\"out\":\"%s\",\"lm\":\"%s\","
       "\"lmSize\":\"%s\",\"rank\":%d,\"alpha\":%d,\"lr\":%.9g,\"epochs\":%d,\"gradAccum\":%d,\"targetLoss\":%.9g,"
       "\"gradClip\":%.9g,\"seed\":%d,\"lossOnCot\":%s,\"bwd\":\"%s\"}",
       stage_csv.c_str(), lm_json_escape(a.tensors_dir).c_str(), lm_json_escape(a.codes_path).c_str(),
       lm_json_escape(a.out_dir).c_str(), lm_json_escape(a.lm_name).c_str(), a.lm_size.c_str(), a.rank, a.alpha,
       (double) a.lr, a.epochs, a.grad_accum, (double) a.target_loss, (double) a.grad_clip, a.seed,
       a.loss_on_cot ? "true" : "false", a.bwd.c_str());

    // ── extract ──────────────────────────────────────────────────────────
    if (lm_has_stage(a, "extract")) {
        LmExtractOpts eo;
        eo.tensors_dir = a.tensors_dir;
        eo.codes_path  = a.codes_path;
        eo.dit_path    = a.dit_path;
        eo.overwrite   = a.overwrite;
        eo.limit       = a.limit;
        std::string err;
        if (!lm_extract_run(eo, &err)) {
            lm_fatal("model-load", err);
            return 1;
        }
    }

    LmExportMeta   meta;
    LmTrainOutcome out;
    meta.producer       = std::string("ace-train ") + ACE_VERSION;
    meta.created_at     = pm_iso8601_utc_now();
    meta.lm_path        = a.lm_path;
    meta.lm_size        = a.lm_size;
    meta.codes_path     = a.codes_path;
    meta.tensors_dir    = a.tensors_dir;
    meta.order          = a.order;
    meta.rank           = a.rank;
    meta.alpha          = a.alpha;
    meta.lr             = a.lr;
    meta.grad_clip      = a.grad_clip;
    meta.weight_decay   = a.weight_decay;
    meta.warmup_ratio   = a.warmup_ratio;
    meta.target_loss    = a.target_loss;
    meta.epochs         = a.epochs;
    meta.grad_accum     = a.grad_accum;
    meta.seed           = a.seed;
    meta.loss_on_cot    = a.loss_on_cot;
    meta.bwd            = a.bwd;

    // Trigger word (T5): CLI flags win, else the variant's preprocess_meta.json.
    // `--codes` always sits in the variant dir, so its parent is the fallback
    // when `--tensors` was not passed (the train-only stage does not need it).
    {
        meta.trigger          = a.trigger;
        meta.trigger_position = a.trigger_position;
        std::string vdir      = a.tensors_dir.empty() ? lm_dirname(a.codes_path) : a.tensors_dir;
        lm_resolve_trigger(vdir, &meta.trigger, &meta.trigger_position);
        if (!meta.trigger.empty()) {
            fprintf(stderr, "[train-lm] trigger \"%s\" (%s) will be embedded in the adapter\n",
                    meta.trigger.c_str(), meta.trigger_position.c_str());
        }
    }

    // ── train ────────────────────────────────────────────────────────────
    if (lm_has_stage(a, "train")) {
        const int rc = lm_train_stage(a, &meta, &out);
        if (rc != 0) {
            return rc;
        }
    }

    // ── export ───────────────────────────────────────────────────────────
    if (lm_has_stage(a, "export")) {
        jl("{\"type\":\"stage\",\"stage\":\"export\",\"state\":\"begin\"}");
        const int64_t t_x0 = ggml_time_ms();
        if (!out.exported) {
            lm_log("warn", "nothing to export — the train stage did not run or produced no epoch");
            jl("{\"type\":\"stage\",\"stage\":\"export\",\"state\":\"end\",\"ms\":%lld}",
               (long long) (ggml_time_ms() - t_x0));
        } else {
            meta.total_ms = (long long) (ggml_time_ms() - t_run0);
            if (!lm_write_train_log(a.out_dir, meta)) {
                lm_fatal("export", "cannot write lm_train_log.json in " + a.out_dir);
                return 1;
            }
            long long bytes = 0;
            pm_stat_file(lm_join(a.out_dir, "adapter_model.safetensors"), &bytes, NULL);
            jl("{\"type\":\"export\",\"path\":\"%s\",\"tensors\":%d,\"bytes\":%lld,\"ms\":%lld}",
               lm_json_escape(a.out_dir).c_str(), out.export_tensors, bytes,
               (long long) (ggml_time_ms() - t_x0));
            jl("{\"type\":\"stage\",\"stage\":\"export\",\"state\":\"end\",\"ms\":%lld}",
               (long long) (ggml_time_ms() - t_x0));
        }
    }

    const long long run_ms = (long long) (ggml_time_ms() - t_run0);
    jl("{\"type\":\"done\",\"stages\":[%s],\"epochsRun\":%d,\"finalLoss\":%.6f,\"stoppedOnTarget\":%s,"
       "\"adapter\":\"%s\",\"samples\":%d,\"skippedLong\":%d,\"ms\":%lld}",
       stage_csv.c_str(), out.epochs_run, out.final_loss, out.stopped_on_target ? "true" : "false",
       lm_json_escape(a.out_dir).c_str(), out.samples, out.skipped_long, run_ms);

    fprintf(stderr, "[train-lm] done in %.1f s\n", (double) run_ms / 1000.0);
    return 0;
}
