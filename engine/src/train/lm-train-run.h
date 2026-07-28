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
    int   epochs = 16, grad_accum = 4;
    float warmup_ratio = 0.05f, grad_clip = 1.0f, weight_decay = 0.01f;
    int   seed         = 42;
    float target_loss  = 0.4f;
    std::string order  = "shuffle";

    int  max_len         = 0;
    int  vram_reserve_mb = 1024;
    bool loss_on_cot     = true;

    float milestone_step = 0.1f;
    int   milestone_keep = 6;

    bool overwrite = false;
    int  limit     = 0;
    bool self_test = false;
};

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

    // ── 4B refusal BEFORE loading any weights (L7 / §3.7) ────────────────
    const bool          is_st = !ends_with_gguf(a.lm_path.c_str());
    const Qwen3LMConfig probe_cfg = qw3lm_load_config(a.lm_path.c_str(), is_st);
    if (probe_cfg.hidden_size >= 2560 || probe_cfg.n_layers >= 36) {
        lm_fatal("unsupported-size",
                 "4B LM training needs ~54 GB; this build supports 0.6B and 1.7B. See "
                 "docs/plans/2026-07-27-lm-trainer-implementation.md §3.7.",
                 ",\"lmSize\":\"4B\"");
        return 1;
    }

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

    // ── F32 mirror; a quantized base is refused here ─────────────────────
    LmF32Mirror mirror;
    {
        std::string err;
        if (!lm_build_f32_mirror(&lm, &mirror, &err)) {
            lm_fatal("model-load", err);
            lm_mirror_free(&mirror);
            qw3lm_free(&lm);
            return 1;
        }
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

    const LmVramFit fit = lm_vram_fit(vm, lm.backend, a.vram_reserve_mb, a.max_len);
    if (!fit.ok) {
        char extra[192];
        snprintf(extra, sizeof(extra), ",\"needMb\":%lld,\"freeMb\":%lld,\"lmSize\":\"%s\"",
                 (long long) (fit.est_bytes / 1048576.0), (long long) fit.free_mb, a.lm_size.c_str());
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
    // inoperable. Clamp to total_steps - 1, and honour ratio 0 as "no warmup".
    int warmup_steps = (a.warmup_ratio > 0.0f)
                           ? std::max(1, (int) ((double) total_steps * (double) a.warmup_ratio))
                           : 0;
    if (warmup_steps > total_steps - 1) {
        warmup_steps = std::max(0, total_steps - 1);
    }

    // ── vram + data events ───────────────────────────────────────────────
    // estMb is the predicted PEAK, i.e. the footprint at the LONGEST ACCEPTED
    // sequence — not at maxLen, which is only the skip threshold. That is the
    // quantity G4 compares against the observed peak.
    const double est_bytes = lm_vram_bytes(vm, max_seq, s_tr_max);
    jl("{\"type\":\"vram\",\"freeMb\":%lld,\"totalMb\":%lld,\"reserveMb\":%d,\"mirrorMb\":%lld,\"maxLen\":%d,"
       "\"estMb\":%lld,\"source\":\"%s\"}",
       (long long) fit.free_mb, (long long) fit.total_mb, a.vram_reserve_mb,
       (long long) (mirror.bytes / 1048576), max_len, (long long) (est_bytes / 1048576.0),
       a.max_len > 0 ? "user" : "auto");

    // `skippedBad` is additive (§2.2: consumers ignore unknown fields) and keeps
    // `skippedLong` meaning what its name says.
    jl("{\"type\":\"data\",\"samples\":%d,\"skippedLong\":%d,\"skippedBad\":%d,\"minLen\":%d,\"maxLen\":%d,"
       "\"maxLenCap\":%d,\"trainedTokens\":%lld,\"stepsPerEpoch\":%d,\"totalSteps\":%d,\"warmupSteps\":%d,"
       "\"loraParams\":%lld}",
       n, skipped_long, skipped_bad, min_len, max_seq, max_len, trained_tokens, steps_per_ep, total_steps,
       warmup_steps, (long long) vm.lora_params);
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
    ggml_tensor * t_tok      = ggml_new_tensor_1d(ctx_static, GGML_TYPE_I32, max_seq);
    ggml_tensor * t_pos      = ggml_new_tensor_1d(ctx_static, GGML_TYPE_I32, max_seq);
    ggml_tensor * t_msk      = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, (int64_t) max_seq * max_seq);
    ggml_tensor * t_lab      = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, V, s_tr_max);
    ggml_tensor * t_adamw    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 7);
    ggml_tensor * t_lossgrad = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_clip     = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_eps      = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_gnorm2   = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_set_input(t_tok);
    ggml_set_input(t_pos);
    ggml_set_input(t_msk);
    ggml_set_input(t_lab);

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
        ggml_backend_tensor_set(t_lossgrad, &lg, 0, sizeof(float));
        ggml_backend_tensor_set(t_clip, &cl, 0, sizeof(float));
        ggml_backend_tensor_set(t_eps, &ep, 0, sizeof(float));
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
    {
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
    fprintf(stderr, "[train-lm] fwd+bwd graph: %d nodes\n", graph_nodes);

    BackendPair bp;
    bp.backend     = lm.backend;
    bp.cpu_backend = lm.cpu_backend;
    bp.has_gpu     = lm.backend != lm.cpu_backend;
    ggml_backend_sched_t sched = backend_sched_new(bp, std::max(8192, graph_nodes + graph_nodes / 2 + 2048));

    // ── one micro-step ───────────────────────────────────────────────────
    LmVramTracker tracker;
    auto micro_step = [&](const LmSample & s, bool count_loss, double * ce_out) -> bool {
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
        size_t fixed = mirror.buf ? ggml_backend_buffer_get_size(mirror.buf) : mirror.bytes;
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
                   "\"lr\":%.9g,\"gradNorm\":%.6f,\"clipScale\":%.6f,\"ms\":%lld,\"vramMb\":%lld}",
                   epoch + 1, global_step, total_steps, win_micro, win_loss / std::max(1, win_micro),
                   (double) last_stats.lr, (double) last_stats.grad_norm, (double) last_stats.clip,
                   (long long) (ggml_time_ms() - t_win0), (long long) vram_mb);
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
       "\"gradClip\":%.9g,\"seed\":%d,\"lossOnCot\":%s}",
       stage_csv.c_str(), lm_json_escape(a.tensors_dir).c_str(), lm_json_escape(a.codes_path).c_str(),
       lm_json_escape(a.out_dir).c_str(), lm_json_escape(a.lm_name).c_str(), a.lm_size.c_str(), a.rank, a.alpha,
       (double) a.lr, a.epochs, a.grad_accum, (double) a.target_loss, (double) a.grad_clip, a.seed,
       a.loss_on_cot ? "true" : "false");

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
