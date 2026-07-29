#pragma once
// dit-train-run.h — `ace-train train-dit` stage driver (plan §3.5.1-§3.5.3).
//
// train -> export. The loop is lm-optim.h, unchanged and unmodified (D1): our own
// persistent grad accumulators driven by ggml_build_backward_expand, and our own
// merged norm+clip+AdamW graph. The DiT trainer is the LM trainer with a
// different graph and a different loss.
//
// Emission contract (§2.2): `start` is the first JSONL line, `done` is the last;
// a `fatal` line replaces `done` and precedes a non-zero exit.

#include "train/dit-data.h"
#include "train/dit-export.h"
#include "train/dit-selftest.h"
#include "train/dit-train-graph.h"
#include "train/dit-vram.h"
#include "train/lm-optim.h"
#include "version.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

// ─── args (§2.1) ────────────────────────────────────────────────────────────

struct DitTrainArgs {
    std::vector<std::string> stages{ "train", "export" };

    std::string tensors_dir, out_dir, models_dir, dit_path, dit_name;

    std::string adapter_type = "lora";
    int         rank = 128, alpha = 256;
    // LoKR (--adapter-type lokr); Uber-LoKR-4 defaults, K2.
    int   lokr_dim            = 512;
    float lokr_alpha          = 512.0f;
    int   lokr_factor         = 6;
    bool  lokr_decompose_both = true;
    // ON by default: an attention-only LoRA leaves the MLP projections — where
    // most of the timbre lives — frozen. --no-target-mlp turns it back off.
    bool        target_mlp = true;
    int         layers     = 0;  // 0 = auto (top-K depth)

    std::string loss_weighting = "flow_snr";
    float       snr_gamma = 5.0f, t_bias = 0.5f;
    bool        channel_balance = true;
    float       timestep_mu = -0.4f, timestep_sigma = 1.0f, t_min = 0.0f, t_max = 1.0f, cfg_ratio = 0.15f;
    int         genre_ratio = 30;

    float       lr = 5e-4f;
    int         epochs = 400, grad_accum = 4;
    float       warmup_ratio = 0.05f, grad_clip = 1.0f, weight_decay = 0.01f;
    int         seed        = 42;
    float       target_loss = 0.4f;
    std::string order       = "shuffle";

    int   crop = 0, crop_min = 375, crop_max = 1250;
    int   vram_reserve_mb = 2048;
    float vram_safety     = 0.05f;

    // Trigger word embedded in the exported adapter. Empty = fall back to the
    // variant's preprocess_meta.json custom_tag/tag_position; still empty after
    // that = no trigger keys written and the adapter is byte-identical to a
    // pre-trigger build. docs/plans/2026-07-28-adapter-trigger-embedding.md T5
    std::string trigger, trigger_position;

    float milestone_step = 0.1f;
    int   milestone_keep = 6;

    bool overwrite = false;
    int  limit     = 0;
    bool self_test = false;
};

// Resolve the trigger to embed: explicit CLI flags win, else the variant's
// preprocess_meta.json, else nothing. `replace` positions are dropped with a
// warn (the tag was never applied to those captions — T4).
static void dit_resolve_trigger(const std::string & tensors_dir, std::string * trigger, std::string * position) {
    if (trigger->empty() && !tensors_dir.empty()) {
        lm_read_variant_tag(tensors_dir, trigger, position);
    }
    std::string why;
    if (!lm_trigger_normalize(trigger, position, &why) && !why.empty()) {
        jl("{\"type\":\"log\",\"level\":\"warn\",\"message\":\"%s\"}", lm_json_escape(why).c_str());
        fprintf(stderr, "[train-dit] %s\n", why.c_str());
    }
}

static inline bool dit_has_stage(const DitTrainArgs & a, const char * s) {
    for (size_t i = 0; i < a.stages.size(); i++) {
        if (a.stages[i] == s) {
            return true;
        }
    }
    return false;
}

struct DitTrainOutcome {
    int       epochs_run        = 0;
    double    final_loss        = -1.0;
    double    best_loss         = -1.0;
    int       best_epoch        = 0;
    bool      stopped_on_target = false;
    int       samples           = 0;
    int       crop              = 0;
    int       layers            = 0;
    bool      exported          = false;
    int       export_tensors    = 0;
    long long ms                = 0;
};

// Milestone directory label. %.1f is right for the 0.1 default and wrong for
// anything finer, so the number of decimals tracks --milestone-step.
static std::string dit_milestone_label(double v, float step) {
    int dec = 1;
    for (int d = 1; d <= 4; d++) {
        dec                 = d;
        const double scaled = (double) step * pow(10.0, d);
        if (fabs(scaled - floor(scaled + 0.5)) < 1e-6) {
            break;  // `step` is representable with d decimals
        }
    }
    char b[48];
    snprintf(b, sizeof(b), "%.*f", dec, v);
    return std::string(b);
}

// ─── the training stage ─────────────────────────────────────────────────────

static int dit_train_stage(const DitTrainArgs & a, DitTrainLog * log, DitTrainOutcome * out) {
    const int64_t t_stage0 = ggml_time_ms();
    const bool    is_lokr  = (a.adapter_type == "lokr");
    jl("{\"type\":\"stage\",\"stage\":\"train\",\"state\":\"begin\",\"total\":%d}", a.epochs);

    // ── samples (host-resident for the whole run) ────────────────────────
    std::vector<DitSample> samples;
    {
        std::string err;
        if (!dit_scan_samples(a.tensors_dir.c_str(), a.limit, &samples, &err)) {
            lm_fatal("no-samples", err);
            return 1;
        }
    }
    DitChannelStats cstats;
    const bool      have_cstats = dit_load_channel_stats(a.tensors_dir.c_str(), &cstats);
    const bool      chan_bal    = a.channel_balance && have_cstats;
    if (a.channel_balance && !have_cstats) {
        lm_log("info", "no channel_stats.json in the variant — channel balancing disabled (Side-Step does the same)");
    }

    // Pad every encoder state to one common enc_S so the graph shape is constant.
    //
    // DEVIATION (E4, documented): §3.1 leaves enc_S per-song. A varying enc_S
    // changes the graph shape every micro-step, which makes ggml_gallocr grow
    // mid-epoch (the exact "40 songs in, OOM" the high-water probe exists to
    // prevent) and makes the footprint model unfalsifiable. Padding to enc_S_max
    // and zeroing the padded positions in the encoder-padding mask is NUMERICALLY
    // EXACT — soft_max_ext gives masked positions exactly zero weight and exactly
    // zero gradient (§3.4 item 1) — and costs only cross-attention FLOPs on the
    // shorter prompts.
    int enc_S = 0, enc_H = 0, max_T = 0, min_T = 0;
    int genre_samples = 0;
    for (size_t i = 0; i < samples.size(); i++) {
        enc_S = std::max(enc_S, std::max(samples[i].enc_S, samples[i].enc_S_genre));
        enc_H = std::max(enc_H, samples[i].enc_H);
        max_T = std::max(max_T, samples[i].T);
        min_T = (i == 0) ? samples[i].T : std::min(min_T, samples[i].T);
        if (!samples[i].enc_genre.empty()) {
            genre_samples++;
        }
    }
    for (size_t i = 0; i < samples.size(); i++) {
        DitSample & s = samples[i];
        s.enc.resize((size_t) enc_S * (size_t) enc_H, 0.0f);
        s.enc_mask.resize((size_t) enc_S, 0.0f);
        s.enc_S = enc_S;
        if (!s.enc_genre.empty()) {
            s.enc_genre.resize((size_t) enc_S * (size_t) enc_H, 0.0f);
            s.enc_mask_genre.resize((size_t) enc_S, 0.0f);
            s.enc_S_genre = enc_S;
        }
    }
    const int n = (int) samples.size();

    // ── load the base on the CPU backend (D3) ────────────────────────────
    DitTrainModel M;
    {
        std::string err;
        if (!dit_train_load(&M, a.dit_path.c_str(), /*lora_lo=*/0, &err)) {
            if (err == "convrot") {
                lm_fatal("convrot-unsupported",
                         "this DiT base carries acestep.convrot_map — ConvRot bases are not trainable in v1 "
                         "(adapter-merge.h already refuses merge mode on them)");
            } else {
                lm_fatal("model-load", err);
            }
            dit_train_free(&M);
            return 1;
        }
    }
    const DiTGGMLConfig & c   = M.m.cfg;
    const int             H   = c.hidden_size, Oc = c.out_channels, P = c.patch_size, L = c.n_layers;
    jl("{\"type\":\"model\",\"stage\":\"dit\",\"path\":\"%s\",\"ms\":%lld,\"layers\":%d,\"hidden\":%d,\"heads\":%d,"
       "\"kvHeads\":%d,\"headDim\":%d,\"inCh\":%d,\"outCh\":%d,\"patch\":%d,\"slidingWindow\":%d,\"convrot\":false}",
       lm_json_escape(a.dit_path).c_str(), M.load_ms, L, H, c.n_heads, c.n_kv_heads, c.head_dim, c.in_channels, Oc, P,
       c.sliding_window);

    if (enc_H != (int) M.m.cond_emb_w->ne[0]) {
        char b[192];
        snprintf(b, sizeof(b), "cached encoder states are [.., %d] but this DiT's condition_embedder wants %d — the "
                               "variant was preprocessed against a different base",
                 enc_H, (int) M.m.cond_emb_w->ne[0]);
        lm_fatal("model-load", b);
        dit_train_free(&M);
        return 1;
    }
    if (samples[0].Oc != Oc || samples[0].Cc + Oc != c.in_channels) {
        char b[192];
        snprintf(b, sizeof(b), "cached latents are [%d target / %d context] but this DiT wants %d/%d", samples[0].Oc,
                 samples[0].Cc, Oc, c.in_channels - Oc);
        lm_fatal("model-load", b);
        dit_train_free(&M);
        return 1;
    }

    // ── the trainer's own compute backend ────────────────────────────────
    {
        std::string err;
        if (!dit_train_backend_init(&M, &err)) {
            lm_fatal("vram", err);
            dit_train_free(&M);
            return 1;
        }
    }

    // ── card support gate (D9) ───────────────────────────────────────────
    const size_t total_mb = lm_vram_total_mb(M.backend);
    if (!dit_vram_card_supported(total_mb)) {
        char extra[128];
        snprintf(extra, sizeof(extra), ",\"needMb\":%d,\"freeMb\":%lld", DIT_MIN_VRAM_MB,
                 (long long) lm_vram_free_mb(M.backend));
        char b[320];
        snprintf(b, sizeof(b),
                 "DiT LoRA training needs a 16 GB card or larger; this one reports %lld MB. Even the 2-layer floor "
                 "(~8.3 GB of F32 mirror) does not fit a 12 GB budget — see the plan's D9 / R1.",
                 (long long) total_mb);
        lm_fatal("vram", b, extra);
        dit_train_free(&M);
        return 1;
    }

    // ── VRAM auto-fit (D10/D11) ──────────────────────────────────────────
    DitVramModel vm;
    vm.m           = &M.m;
    vm.n_layers    = L;
    vm.patch       = P;
    vm.rank        = a.rank;
    vm.target_mlp  = a.target_mlp;
    vm.is_lokr     = is_lokr;
    vm.lokr_dim    = a.lokr_dim;
    vm.lokr_factor = a.lokr_factor;
    vm.in_ch       = c.in_channels;
    vm.out_ch      = Oc;
    vm.hidden      = H;
    vm.enc_H       = enc_H;
    vm.enc_S       = enc_S;

    const DitVramFit fit =
        dit_vram_fit(vm, M.backend, a.vram_reserve_mb, a.vram_safety, a.crop, a.layers, a.crop_min, a.crop_max, max_T);
    if (!fit.ok) {
        char extra[160];
        snprintf(extra, sizeof(extra), ",\"needMb\":%lld,\"freeMb\":%lld",
                 (long long) (fit.est_bytes / 1048576.0), (long long) fit.free_mb);
        char b[320];
        snprintf(b, sizeof(b),
                 "not enough free VRAM: the smallest configuration that would run needs ~%lld MB, %lld MB free "
                 "(reserve %d MB, safety %.0f%%)",
                 (long long) (fit.est_bytes / 1048576.0), (long long) fit.free_mb, a.vram_reserve_mb,
                 (double) a.vram_safety * 100.0);
        lm_fatal("vram", b, extra);
        dit_train_free(&M);
        return 1;
    }
    const int K        = fit.layers;
    const int crop_len = fit.crop;
    const int lora_lo  = L - K;
    const int S_max    = crop_len / P;
    out->crop          = crop_len;
    out->layers        = K;

    // ── the streamed F32 mirror, built ONCE at the chosen depth ──────────
    {
        std::string err;
        if (!dit_build_mirror(&M, lora_lo, &err)) {
            lm_fatal(err.find("quantized") != std::string::npos ? "model-load" : "vram", err);
            dit_train_free(&M);
            return 1;
        }
    }
    fprintf(stderr, "[train-dit] mirror: %d promoted to F32 (%.1f MB) + %d kept native, %.1f MB total; device-wide %zu MB\n",
            M.mirror.n_f32, M.mirror.bytes_f32 / 1048576.0, M.mirror.n_keep, M.mirror.bytes / 1048576.0,
            lm_vram_used_mb(M.backend));

    // ── scheduler ────────────────────────────────────────────────────────
    {
        BackendPair bp;
        bp.backend     = M.backend;
        bp.cpu_backend = M.cpu;
        bp.has_gpu     = true;
        M.sched        = backend_sched_new(bp, 65536);
        if (!M.sched) {
            lm_fatal("vram", "cannot create the training scheduler");
            dit_train_free(&M);
            return 1;
        }
    }

    // ── static input buffers (1-D bases; every graph tensor is a CONTIGUOUS
    //    view at offset 0, because soft_max_ext will not take a strided mask) ─
    ggml_context * ctx_static;
    {
        ggml_init_params p = { 32 * ggml_tensor_overhead(), nullptr, true };
        ctx_static         = ggml_init(p);
    }
    ggml_tensor * b_input = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, (int64_t) c.in_channels * crop_len);
    ggml_tensor * b_enc   = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, (int64_t) enc_H * enc_S);
    ggml_tensor * b_pos   = ggml_new_tensor_1d(ctx_static, GGML_TYPE_I32, S_max);
    ggml_tensor * t_temb  = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, H);
    ggml_tensor * t_tproj = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 6 * H);
    ggml_tensor * b_sa    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F16, (int64_t) S_max * S_max);
    ggml_tensor * b_ca    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F16, (int64_t) enc_S * S_max);
    ggml_tensor * b_vtgt  = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, (int64_t) Oc * crop_len);
    ggml_tensor * t_cw    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, Oc);
    ggml_tensor * t_adamw    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 7);
    ggml_tensor * t_lossgrad = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_clip     = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_eps      = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_gnorm2   = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    for (ggml_tensor * t : { b_input, b_enc, b_pos, t_temb, t_tproj, b_sa, b_ca, b_vtgt, t_cw }) {
        ggml_set_input(t);
    }
    ggml_backend_buffer_t buf_static = ggml_backend_alloc_ctx_tensors(ctx_static, M.backend);
    if (!buf_static) {
        lm_fatal("vram", "static input allocation failed");
        ggml_free(ctx_static);
        dit_train_free(&M);
        return 1;
    }
    {
        std::vector<float> cw((size_t) Oc, 1.0f);
        for (int i = 0; i < Oc && i < (int) cstats.weight.size(); i++) {
            cw[(size_t) i] = cstats.weight[(size_t) i];
        }
        ggml_backend_tensor_set(t_cw, cw.data(), 0, cw.size() * sizeof(float));
        const float epsv = 1e-6f, clipv = a.grad_clip, lg = 1.0f;
        ggml_backend_tensor_set(t_eps, &epsv, 0, 4);
        ggml_backend_tensor_set(t_clip, &clipv, 0, 4);
        ggml_backend_tensor_set(t_lossgrad, &lg, 0, 4);
    }

    // ── adapter + optimizer ──────────────────────────────────────────────
    DitAdapterLora lora;
    DitAdapterLoKr lokr;
    DitAdapter *   adapter = is_lokr ? (DitAdapter *) &lokr : (DitAdapter *) &lora;
    {
        DitAdapterCfg cfg;
        cfg.rank                = a.rank;
        cfg.alpha               = (float) a.alpha;
        cfg.target_mlp          = a.target_mlp;
        cfg.seed                = (uint64_t) a.seed;
        cfg.lokr_dim            = a.lokr_dim;
        cfg.lokr_alpha          = a.lokr_alpha;
        cfg.lokr_factor         = a.lokr_factor;
        cfg.lokr_decompose_both = a.lokr_decompose_both;
        std::string err;
        if (!adapter->init(&M.m, M.backend, lora_lo, L, cfg, &err)) {
            lm_fatal("unsupported-adapter", err);
            ggml_backend_buffer_free(buf_static);
            ggml_free(ctx_static);
            dit_train_free(&M);
            return 1;
        }
        const size_t expect =
            is_lokr ? dit_lokr_expected_params(c, lora_lo, L, a.lokr_dim, a.lokr_factor, a.target_mlp)
                    : dit_lora_expected_params(c, lora_lo, L, a.rank, a.target_mlp);
        if (adapter->nParams() != expect) {
            char b[192];
            snprintf(b, sizeof(b), "%s parameter count %zu != expected %zu", adapter->typeName(), adapter->nParams(),
                     expect);
            lm_fatal("unsupported-adapter", b);
            ggml_backend_buffer_free(buf_static);
            ggml_free(ctx_static);
            dit_train_free(&M);
            return 1;
        }
    }
    const DitAdapter * ad = adapter;

    LmOptim opt;
    {
        std::string err;
        if (!lm_optim_init(&opt, adapter->params(), M.backend, &err)) {
            lm_fatal("vram", err);
            adapter->free();
            ggml_backend_buffer_free(buf_static);
            ggml_free(ctx_static);
            dit_train_free(&M);
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

    const int steps_per_ep = (n + a.grad_accum - 1) / a.grad_accum;
    const int total_steps  = std::max(1, steps_per_ep * a.epochs);
    int       warmup_steps = (a.warmup_ratio > 0.0f)
                                 ? std::max(1, (int) ((double) total_steps * (double) a.warmup_ratio))
                                 : 0;
    if (warmup_steps > total_steps - 1) {
        warmup_steps = std::max(0, total_steps - 1);  // lm_lr_lambda(0,...) == 0 by design
    }
    opt.total_steps  = total_steps;
    opt.warmup_steps = warmup_steps;

    // ── events (§2.2 order: model, vram, data, then steps) ───────────────
    // `deviceWideMb` is (total - free) for the WHOLE card, unlike every other
    // number here: it is the only figure that can see Windows quietly backing an
    // over-subscribed allocation with shared system memory. Sampled here, after
    // the mirror / adapter / optimizer are resident but before the probe.
    jl("{\"type\":\"vram\",\"freeMb\":%lld,\"totalMb\":%lld,\"reserveMb\":%d,\"mirrorMb\":%lld,\"crop\":%d,"
       "\"layers\":%d,\"loraParams\":%lld,\"estMb\":%lld,\"deviceWideMb\":%lld,\"source\":\"%s\","
       "\"cropSource\":\"%s\",\"layersSource\":\"%s\"}",
       (long long) fit.free_mb, (long long) fit.total_mb, a.vram_reserve_mb, (long long) (M.mirror.bytes / 1048576),
       crop_len, K, (long long) adapter->nParams(), (long long) (fit.est_bytes / 1048576.0),
       (long long) lm_vram_used_mb(M.backend), (fit.crop_user && fit.layers_user) ? "user" : "auto",
       fit.crop_user ? "user" : "auto", fit.layers_user ? "user" : "auto");
    if (K < L) {
        // §2.8 gates the UI banner on layersSource === 'auto', so this line must
        // say which it was; and the number that matters is the fitted BUDGET, not
        // the card's total VRAM (which the auto-fit never sees).
        const double budget_mb =
            ((double) fit.free_mb - (double) a.vram_reserve_mb) * (double) (1.0f - a.vram_safety);
        char b[288];
        snprintf(b, sizeof(b),
                 "%s trained depth is %d of %d layers (fitted against a %lld MB budget: %lld MB free - %d MB reserve, "
                 "%.0f%% safety) — adapter quality at partial depth is unvalidated",
                 fit.layers_user ? "Requested" : "Auto-fit reduced:", K, L, (long long) budget_mb,
                 (long long) fit.free_mb, a.vram_reserve_mb, (double) a.vram_safety * 100.0);
        lm_log("warn", b);
    }

    // D13: "If null_condition_emb is absent from the base, CFG dropout is
    // disabled with a `warn` and the run continues." Disabling it by short-circuit
    // alone left dit_train_log.json claiming cfg_ratio 0.15 on a base that never
    // applied it. Decide once, warn once, and use the same flag everywhere so a
    // `step` event can never report cfgDrop:1 without a replacement having happened.
    const bool cfg_ok = !M.null_cond.empty() && (int) M.null_cond.size() == enc_H;
    if (a.cfg_ratio > 0.0f && !cfg_ok) {
        char cb[256];
        snprintf(cb, sizeof(cb),
                 "CFG dropout disabled: this base has %s null_condition_emb, so the requested --cfg-ratio %.3g is not "
                 "applied (D13)",
                 M.null_cond.empty() ? "no" : "a wrongly-sized", (double) a.cfg_ratio);
        lm_log("warn", cb);
    }

    LmRng rng_order, rng_t, rng_noise, rng_crop, rng_cond;
    lm_rng_seed(&rng_order, (uint64_t) a.seed);
    lm_rng_seed(&rng_t, (uint64_t) a.seed ^ 0x517cc1b727220a95ull);
    lm_rng_seed(&rng_noise, (uint64_t) a.seed ^ 0x9e3779b97f4a7c15ull);
    lm_rng_seed(&rng_crop, (uint64_t) a.seed ^ 0xbf58476d1ce4e5b9ull);
    lm_rng_seed(&rng_cond, (uint64_t) a.seed ^ 0xd1b54a32d192ed03ull);

    // ── the micro-step ───────────────────────────────────────────────────
    std::vector<uint8_t> arena((size_t) 512 << 20);
    std::vector<float>   input_buf((size_t) c.in_channels * crop_len, 0.0f);
    std::vector<float>   flow_xt, flow_v;
    std::vector<int32_t> pos_buf((size_t) S_max);
    std::vector<uint16_t> sa_buf, ca_buf;
    std::vector<float>    enc_buf((size_t) enc_H * enc_S, 0.0f);
    int                   graph_nodes = 0;
    int                   last_cfg_drop = 0, last_crop_start = 0;

    auto micro_step = [&](const DitSample & s, float t, const std::vector<float> & temb_h,
                          const std::vector<float> & tproj_h, bool use_genre, bool cfg_drop, bool backward,
                          double * raw_out) -> bool {
        const DitCrop cr  = dit_sample_crop(&rng_crop, s.T, crop_len, P);
        const int     len = cr.len;
        const int     S   = len / P;
        last_crop_start   = cr.start;
        last_cfg_drop     = cfg_drop ? 1 : 0;

        // rectified flow (D15): one rng stream, index order
        dit_flow_target(s.lat.data() + (size_t) cr.start * (size_t) s.Oc, (size_t) len * (size_t) s.Oc, t, &rng_noise,
                        &flow_xt, &flow_v);

        // pack [context(Cc) | xt(Oc)] per frame — hot-step-sampler.h:546/728
        for (int f = 0; f < len; f++) {
            float * dst = &input_buf[(size_t) f * (size_t) c.in_channels];
            memcpy(dst, &s.ctxl[(size_t) (cr.start + f) * (size_t) s.Cc], (size_t) s.Cc * sizeof(float));
            memcpy(dst + s.Cc, &flow_xt[(size_t) f * (size_t) s.Oc], (size_t) s.Oc * sizeof(float));
        }

        // encoder conditioning: genre variant | null (CFG dropout) | caption
        const std::vector<float> * src_enc  = &s.enc;
        const std::vector<float> * src_mask = &s.enc_mask;
        if (use_genre && !s.enc_genre.empty()) {
            src_enc  = &s.enc_genre;
            src_mask = &s.enc_mask_genre;
        }
        if (cfg_drop) {
            for (int i = 0; i < enc_S; i++) {
                memcpy(&enc_buf[(size_t) i * (size_t) enc_H], M.null_cond.data(), (size_t) enc_H * sizeof(float));
            }
        } else {
            memcpy(enc_buf.data(), src_enc->data(), enc_buf.size() * sizeof(float));
        }

        for (int i = 0; i < S; i++) {
            pos_buf[(size_t) i] = i;
        }
        dit_sa_mask(S, c.sliding_window, &sa_buf);
        dit_ca_mask(enc_S, S, *src_mask, &ca_buf);

        // §3.0: EVERY input is re-uploaded, unconditionally. GGML clobbers input
        // buffers; "only upload if changed" is the bug class the per-section
        // adapter masking work hit.
        ggml_backend_tensor_set(b_input, input_buf.data(), 0, (size_t) c.in_channels * (size_t) len * sizeof(float));
        ggml_backend_tensor_set(b_vtgt, flow_v.data(), 0, flow_v.size() * sizeof(float));
        ggml_backend_tensor_set(b_enc, enc_buf.data(), 0, enc_buf.size() * sizeof(float));
        ggml_backend_tensor_set(b_pos, pos_buf.data(), 0, (size_t) S * sizeof(int32_t));
        ggml_backend_tensor_set(b_sa, sa_buf.data(), 0, sa_buf.size() * sizeof(uint16_t));
        ggml_backend_tensor_set(b_ca, ca_buf.data(), 0, ca_buf.size() * sizeof(uint16_t));
        ggml_backend_tensor_set(t_temb, temb_h.data(), 0, temb_h.size() * sizeof(float));
        ggml_backend_tensor_set(t_tproj, tproj_h.data(), 0, tproj_h.size() * sizeof(float));

        ggml_init_params ip  = { arena.size(), arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 65536, /*grads=*/backward);

        DitInputs in;
        in.t_input = ggml_view_2d(ctx, b_input, c.in_channels, len, (size_t) c.in_channels * sizeof(float), 0);
        in.t_enc   = ggml_view_2d(ctx, b_enc, enc_H, enc_S, (size_t) enc_H * sizeof(float), 0);
        in.t_pos   = ggml_view_1d(ctx, b_pos, S, 0);
        in.t_temb  = t_temb;
        in.t_tproj = t_tproj;
        in.t_sa    = ggml_view_2d(ctx, b_sa, S, S, (size_t) S * sizeof(ggml_fp16_t), 0);
        in.t_ca    = ggml_view_2d(ctx, b_ca, enc_S, S, (size_t) enc_S * sizeof(ggml_fp16_t), 0);
        in.t_vtgt  = ggml_view_2d(ctx, b_vtgt, Oc, len, (size_t) Oc * sizeof(float), 0);
        in.t_cw    = t_cw;

        ggml_tensor * vpred = dit_train_forward(ctx, &M, ad, in, len, enc_S);
        ggml_tensor * loss  = dit_train_loss(ctx, vpred, in, Oc, len, chan_bal);
        if (backward) {
            ggml_set_loss(loss);
        }
        ggml_build_forward_expand(gf, loss);
        if (backward) {
            std::vector<ggml_tensor *> gacc;
            lm_optim_fill_gacc(&opt, gf, &gacc);
            ggml_build_backward_expand(ctx, gf, gacc.data());
        }
        graph_nodes = ggml_graph_n_nodes(gf);
        GGML_ASSERT(graph_nodes < 65536);

        ggml_backend_sched_reset(M.sched);
        const bool ok = ggml_backend_sched_graph_compute(M.sched, gf) == GGML_STATUS_SUCCESS;
        if (ok && raw_out) {
            float lv = 0.0f;
            ggml_backend_tensor_get(loss, &lv, 0, sizeof(float));
            *raw_out = (double) lv;
        }
        ggml_free(ctx);
        return ok;
    };

    // ── teardown helper (used by every failure path from here on) ────────
    auto teardown = [&]() {
        lm_optim_free(&opt);
        adapter->free();
        ggml_backend_buffer_free(buf_static);
        ggml_free(ctx_static);
        dit_train_free(&M);
    };

    // ── high-water probe (mandatory, §3.7) ───────────────────────────────
    //
    // One throwaway micro-step at the chosen (crop, K) on the LONGEST sample,
    // loss discarded. This forces ggml_gallocr to its peak up front, so a
    // 40-songs-in OOM becomes an immediate honest failure, and it establishes
    // the leak counter's baseline.
    LmVramTracker tracker;
    {
        size_t longest = 0;
        for (size_t i = 1; i < samples.size(); i++) {
            if (samples[i].T > samples[longest].T) {
                longest = i;
            }
        }
        std::vector<float>              tv(1, 0.5f);
        std::vector<std::vector<float>> tb, tp;
        double                          lv  = 0.0;
        const int64_t                   tp0 = ggml_time_ms();
        if (!dit_train_temb(&M, tv, &tb, &tp)) {
            lm_fatal("vram", "the timestep-embedding precompute graph failed");
            teardown();
            return 1;
        }
        if (!micro_step(samples[longest], 0.5f, tb[0], tp[0], false, false, true, &lv)) {
            lm_fatal("vram", "the high-water probe failed — not enough VRAM for this crop/depth");
            teardown();
            return 1;
        }
        lm_optim_zero_grad(&opt);
        size_t fixed = M.mirror.buf ? ggml_backend_buffer_get_size(M.mirror.buf) : M.mirror.bytes;
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
        tracker.probe_baseline(M.backend, M.sched, fixed);
        const size_t device_wide_mb = lm_vram_used_mb(M.backend);
        fprintf(stderr,
                "[train-dit] graph %d nodes (fwd+bwd); high-water probe loss=%.6f in %lld ms; trainer-owned %zu MB "
                "(est %lld MB), device-wide %zu MB\n",
                graph_nodes, lv, (long long) (ggml_time_ms() - tp0), tracker.base_mb,
                (long long) (fit.est_bytes / 1048576.0), device_wide_mb);
        // The probe succeeding does NOT mean it fitted: on Windows an
        // over-subscribed CUDA allocation is silently backed by shared system
        // memory instead of failing, and the run then trains at a crawl. Only the
        // device-wide figure can see it — the trainer-owned one counts the same
        // bytes whether they landed in VRAM or in host RAM. Warn, never fail: the
        // spill is a performance cliff, not an error, and the reserve/safety
        // margins already own the refusal path. (Written as `+ 512 >` so a card
        // reporting under 512 MB total cannot underflow the size_t.)
        if (device_wide_mb + 512 > total_mb) {
            char b[288];
            snprintf(b, sizeof(b),
                     "VRAM overcommitted after probe (%lld of %lld MB device-wide) — Windows is likely spilling into "
                     "shared GPU memory; reduce crop, layers, or lokr dim, or close other GPU apps",
                     (long long) device_wide_mb, (long long) total_mb);
            lm_log("warn", b);
        }
    }

    jl("{\"type\":\"data\",\"samples\":%d,\"skipped\":0,\"minFrames\":%d,\"maxFrames\":%d,\"encSMax\":%d,"
       "\"channelStats\":%s,\"genreSamples\":%d,\"stepsPerEpoch\":%d,\"totalSteps\":%d,\"warmupSteps\":%d,"
       "\"graphNodes\":%d}",
       n, min_T, max_T, enc_S, have_cstats ? "true" : "false", genre_samples, steps_per_ep, total_steps, warmup_steps,
       graph_nodes);

    dit_milestone_reset(a.out_dir);

    // ── log config ───────────────────────────────────────────────────────
    log->layers         = K;
    log->n_layers_total = L;
    log->layers_source  = fit.layers_user ? "user" : "auto";
    log->crop           = crop_len;
    log->crop_source    = fit.crop_user ? "user" : "auto";
    log->samples        = n;
    log->partial_depth  = K < L;
    log->channel_balance = chan_bal;
    log->vram_free_mb   = fit.free_mb;
    log->vram_total_mb  = fit.total_mb;
    log->vram_mirror_mb = M.mirror.bytes / 1048576;
    log->vram_est_mb    = (size_t) (fit.est_bytes / 1048576.0);

    // ── epoch loop (§3.5.3) ──────────────────────────────────────────────
    {
        DitExportMeta xmeta;
        xmeta.base_model_path = a.dit_path;
        xmeta.producer        = std::string("ace-train ") + ACE_VERSION;
        xmeta.trigger          = log->trigger;
        xmeta.trigger_position = log->trigger_position;

        std::vector<double> ep_hist;
        double              ladder      = 0.0;
        bool                ladder_seed = false;
        long long           ep_ms_sum   = 0;
        int                 global_step = 0;
        int                 rc          = 0;

        for (int epoch = 0; epoch < a.epochs && rc == 0; epoch++) {
            const int64_t    t_ep0 = ggml_time_ms();
            std::vector<int> order((size_t) n);
            for (int i = 0; i < n; i++) {
                order[(size_t) i] = i;
            }
            if (a.order != "fixed") {
                LmRng er;
                lm_rng_seed(&er, (uint64_t) a.seed + (uint64_t) epoch * 7919ull);
                lm_rng_shuffle(&er, order);
            }

            double      running = 0.0, running_raw = 0.0;
            int         n_micro = 0;
            LmStepStats last_stats;
            lm_optim_zero_grad(&opt);

            for (int w0 = 0; w0 < n && rc == 0; w0 += a.grad_accum) {
                const int wlen = std::min(a.grad_accum, n - w0);

                // Sample the WHOLE window's timesteps up front: the flow_snr mean
                // needs them, and so does the temb precompute (§3.5's batch-of-1 trap).
                std::vector<float> ts((size_t) wlen), wgt((size_t) wlen);
                double             wsum = 0.0;
                for (int i = 0; i < wlen; i++) {
                    ts[(size_t) i]  = dit_sample_t(&rng_t, a.timestep_mu, a.timestep_sigma, a.t_min, a.t_max);
                    wgt[(size_t) i] = (a.loss_weighting == "flow_snr")
                                          ? dit_flow_snr_w(ts[(size_t) i], a.t_bias, a.snr_gamma)
                                          : 1.0f;
                    wsum += (double) wgt[(size_t) i];
                }
                const double wbar = wsum / std::max(1, wlen);

                std::vector<std::vector<float>> temb_h, tproj_h;
                if (!dit_train_temb(&M, ts, &temb_h, &tproj_h)) {
                    lm_fatal("vram", "the timestep-embedding precompute graph failed mid-epoch");
                    rc = 1;
                    break;
                }

                double        win_loss = 0.0, win_raw = 0.0;
                const int64_t t_win0 = ggml_time_ms();
                for (int i = 0; i < wlen; i++) {
                    const DitSample & s = samples[(size_t) order[(size_t) (w0 + i)]];
                    const float       wnorm = (float) ((double) wgt[(size_t) i] / (wbar > 0.0 ? wbar : 1.0));
                    const float       lg    = wnorm / (float) wlen;
                    ggml_backend_tensor_set(t_lossgrad, &lg, 0, sizeof(float));

                    const bool use_genre = a.genre_ratio > 0 && !s.enc_genre.empty() &&
                                           (lm_rng_uniform(&rng_cond) * 100.0f < (float) a.genre_ratio);
                    const bool cfg_drop =
                        a.cfg_ratio > 0.0f && cfg_ok && (lm_rng_uniform(&rng_cond) < a.cfg_ratio);

                    double raw = 0.0;
                    if (!micro_step(s, ts[(size_t) i], temb_h[(size_t) i], tproj_h[(size_t) i], use_genre, cfg_drop,
                                    true, &raw)) {
                        lm_fatal("vram", "graph compute failed mid-epoch");
                        rc = 1;
                        break;
                    }
                    win_raw += raw;
                    win_loss += (double) wnorm * raw;
                    running += (double) wnorm * raw;
                    running_raw += raw;
                    n_micro++;
                }
                if (rc != 0) {
                    break;
                }
                if (!lm_optim_step(&opt, M.sched, &last_stats)) {
                    lm_fatal("vram", "optimizer step failed");
                    rc = 1;
                    break;
                }
                global_step++;
                const size_t vram_mb = tracker.sample();
                jl("{\"type\":\"step\",\"epoch\":%d,\"step\":%d,\"totalSteps\":%d,\"micro\":%d,\"loss\":%.6f,"
                   "\"rawLoss\":%.6f,\"lr\":%.9g,\"gradNorm\":%.6f,\"clipScale\":%.6f,\"t\":%.4f,\"crop\":%d,"
                   "\"cropStart\":%d,\"cfgDrop\":%d,\"ms\":%lld,\"vramMb\":%lld}",
                   epoch + 1, global_step, total_steps, wlen, win_loss / (double) wlen, win_raw / (double) wlen,
                   (double) last_stats.lr, (double) last_stats.grad_norm, (double) last_stats.clip,
                   (double) ts[(size_t) (wlen - 1)], crop_len, last_crop_start, last_cfg_drop,
                   (long long) (ggml_time_ms() - t_win0), (long long) vram_mb);
            }
            if (rc != 0) {
                break;
            }

            const double    avg     = running / std::max(1, n_micro);
            const double    avg_raw = running_raw / std::max(1, n_micro);
            const long long ems     = (long long) (ggml_time_ms() - t_ep0);
            ep_ms_sum += ems;
            ep_hist.push_back(avg);
            double ma5 = 0.0;
            {
                const int kk = (int) std::min<size_t>(5, ep_hist.size());
                for (int i = 0; i < kk; i++) {
                    ma5 += ep_hist[ep_hist.size() - 1 - (size_t) i];
                }
                ma5 /= (double) kk;
            }

            const bool best = (out->best_loss < 0.0) || (avg < out->best_loss);
            if (best) {
                out->best_loss  = avg;
                out->best_epoch = epoch + 1;
            }
            out->final_loss = avg;
            out->epochs_run = epoch + 1;

            DitEpochRec rec;
            rec.epoch     = epoch + 1;
            rec.loss      = avg;
            rec.ma5       = ma5;
            rec.raw_loss  = avg_raw;
            rec.lr        = last_stats.lr;
            rec.grad_norm = last_stats.grad_norm;
            rec.ms        = ems;
            log->epochs_log.push_back(rec);
            log->epochs_run = out->epochs_run;
            log->final_loss = out->final_loss;
            log->best_loss  = out->best_loss;
            log->best_epoch = out->best_epoch;

            // Export EVERY epoch, BEFORE the stop test (D16): the checkpoint that
            // triggered the stop is already on disk, and a cancel or crash always
            // leaves a usable adapter.
            {
                DitExportResult xr;
                std::string     xerr;
                if (!dit_export_peft(*ad, xmeta, a.out_dir.c_str(), &xr, &xerr)) {
                    lm_fatal("export", xerr);
                    rc = 1;
                    break;
                }
                out->exported       = true;
                out->export_tensors = xr.tensors;
            }

            if (a.milestone_step > 0.0f) {
                if (!ladder_seed) {
                    ladder      = floor(ma5 / (double) a.milestone_step) * (double) a.milestone_step;
                    ladder_seed = true;
                }
                while (ma5 <= ladder + 1e-12) {
                    // %.1f collides for any --milestone-step below 0.1 (1.15 and
                    // 1.10 both format to "1.1"), which makes two ladder rungs
                    // share one directory and lets dit_milestone_prune delete a
                    // path a surviving log entry still references. Widen the label
                    // just enough for the configured step.
                    const std::string label = dit_milestone_label(ladder, a.milestone_step);
                    const double      lval  = atof(label.c_str());
                    const std::string rel   = "milestones/loss_" + label;
                    const std::string mdir  = lm_join(a.out_dir, rel);
                    DitExportResult   mr;
                    std::string       merr;
                    if (dit_export_peft(*ad, xmeta, mdir.c_str(), &mr, &merr)) {
                        DitMilestoneRec ms;
                        ms.loss  = lval;
                        ms.epoch = epoch + 1;
                        ms.path  = rel;
                        log->milestones.push_back(ms);
                        jl("{\"type\":\"milestone\",\"loss\":%.4g,\"epoch\":%d,\"path\":\"%s\",\"bytes\":%lld}", lval,
                           epoch + 1, lm_json_escape(mdir).c_str(), mr.bytes);
                        dit_milestone_prune(a.out_dir, &log->milestones, a.milestone_keep);
                    } else {
                        lm_log("warn", "milestone snapshot failed: " + merr);
                    }
                    ladder -= (double) a.milestone_step;
                }
            }

            log->vram_peak_mb = tracker.peak_mb;
            log->total_ms     = (long long) (ggml_time_ms() - t_stage0);
            if (!dit_write_train_log(a.out_dir, *log)) {
                lm_log("warn", "cannot write dit_train_log.json in " + a.out_dir);
            }

            const long long eta =
                (long long) ((double) ep_ms_sum / (double) (epoch + 1) * (double) (a.epochs - epoch - 1));
            jl("{\"type\":\"epoch\",\"epoch\":%d,\"epochs\":%d,\"loss\":%.6f,\"ma5\":%.6f,\"rawLoss\":%.6f,"
               "\"lr\":%.9g,\"gradNorm\":%.6f,\"ms\":%lld,\"etaMs\":%lld,\"best\":%s}",
               epoch + 1, a.epochs, avg, ma5, avg_raw, (double) last_stats.lr, (double) last_stats.grad_norm, ems, eta,
               best ? "true" : "false");
            jl("{\"type\":\"progress\",\"completed\":%d,\"total\":%d,\"phase\":\"train\"}", epoch + 1, a.epochs);
            fprintf(stderr, "[train-dit] epoch %d/%d loss=%.6f ma5=%.6f raw=%.6f lr=%.3e gnorm=%.3f (%lld ms)\n",
                    epoch + 1, a.epochs, avg, ma5, avg_raw, (double) last_stats.lr, (double) last_stats.grad_norm, ems);

            if (a.target_loss > 0.0f && ma5 <= (double) a.target_loss) {
                out->stopped_on_target  = true;
                log->target_stop        = true;
                log->target_stop_epoch  = epoch + 1;
                log->target_stop_loss   = avg;
                log->target_stop_ma5    = ma5;
                jl("{\"type\":\"target_stop\",\"epoch\":%d,\"loss\":%.6f,\"ma5\":%.6f,\"targetLoss\":%.6g}", epoch + 1,
                   avg, ma5, (double) a.target_loss);
                break;
            }
        }

        log->vram_peak_mb = tracker.peak_mb;
        out->samples      = n;
        out->ms           = (long long) (ggml_time_ms() - t_stage0);

        char b[224];
        snprintf(b, sizeof(b), "leak counter: baseline %zu MB, peak %zu MB, max delta %lld MB over %d optimizer steps",
                 tracker.base_mb, tracker.peak_mb, tracker.max_delta, global_step);
        fprintf(stderr, "[train-dit] %s\n", b);
        jl("{\"type\":\"leak\",\"baselineMb\":%lld,\"peakMb\":%lld,\"deltaMb\":%lld,\"steps\":%d}",
           (long long) tracker.base_mb, (long long) tracker.peak_mb, tracker.max_delta, global_step);

        if (rc == 0) {
            jl("{\"type\":\"stage\",\"stage\":\"train\",\"state\":\"end\",\"epochsRun\":%d,\"finalLoss\":%.6f,"
               "\"bestLoss\":%.6f,\"stoppedOnTarget\":%s,\"ms\":%lld}",
               out->epochs_run, out->final_loss, out->best_loss, out->stopped_on_target ? "true" : "false", out->ms);
        }

        teardown();
        return rc;
    }
}

// ─── main entry ─────────────────────────────────────────────────────────────

static int dit_train_main(const DitTrainArgs & a) {
    const bool   is_lokr      = (a.adapter_type == "lokr");
    const char * weights_leaf = is_lokr ? "lokr_weights.safetensors" : "adapter_model.safetensors";
    if (a.self_test) {
        return dit_self_test(a.dit_path, a.tensors_dir, (uint64_t) a.seed, std::min(a.crop_max, 750),
                             a.vram_reserve_mb, a.vram_safety);
    }

    const int64_t t_run0 = ggml_time_ms();

    std::string stage_csv;
    for (size_t i = 0; i < a.stages.size(); i++) {
        if (i) {
            stage_csv += ",";
        }
        stage_csv += "\"" + a.stages[i] + "\"";
    }

    // LoKR carries its own geometry fields; `rank`/`alpha` stay at 0 so a relay
    // cannot mistake the LoRA defaults for the configuration that ran.
    char adapter_fields[160];
    if (is_lokr) {
        snprintf(adapter_fields, sizeof(adapter_fields),
                 "\"rank\":0,\"alpha\":0,\"lokrDim\":%d,\"lokrAlpha\":%.9g,\"lokrFactor\":%d,\"lokrDecomposeBoth\":%s",
                 a.lokr_dim, (double) a.lokr_alpha, a.lokr_factor, a.lokr_decompose_both ? "true" : "false");
    } else {
        snprintf(adapter_fields, sizeof(adapter_fields), "\"rank\":%d,\"alpha\":%d", a.rank, a.alpha);
    }
    jl("{\"type\":\"start\",\"stages\":[%s],\"tensors\":\"%s\",\"out\":\"%s\",\"dit\":\"%s\",\"adapterType\":\"%s\","
       "%s,\"targetMlp\":%s,\"lr\":%.9g,\"epochs\":%d,\"gradAccum\":%d,\"targetLoss\":%.9g,"
       "\"gradClip\":%.9g,\"seed\":%d,\"lossWeighting\":\"%s\",\"channelBalance\":%s,\"cfgRatio\":%.9g,"
       "\"genreRatio\":%d}",
       stage_csv.c_str(), lm_json_escape(a.tensors_dir).c_str(), lm_json_escape(a.out_dir).c_str(),
       lm_json_escape(a.dit_name.empty() ? a.dit_path : a.dit_name).c_str(), a.adapter_type.c_str(), adapter_fields,
       a.target_mlp ? "true" : "false", (double) a.lr, a.epochs, a.grad_accum, (double) a.target_loss,
       (double) a.grad_clip, a.seed, a.loss_weighting.c_str(), a.channel_balance ? "true" : "false",
       (double) a.cfg_ratio, a.genre_ratio);

    if (a.adapter_type != "lora" && !is_lokr) {
        lm_fatal("unsupported-adapter", "--adapter-type must be lora|lokr");
        return 1;
    }

    DitTrainLog     log;
    DitTrainOutcome out;
    log.producer        = std::string("ace-train ") + ACE_VERSION;
    log.created_at      = pm_iso8601_utc_now();
    log.adapter_type    = a.adapter_type;
    log.rank            = a.rank;
    log.alpha           = a.alpha;
    log.is_lokr             = is_lokr;
    log.lokr_dim            = a.lokr_dim;
    log.lokr_alpha          = a.lokr_alpha;
    log.lokr_factor         = a.lokr_factor;
    log.lokr_decompose_both = a.lokr_decompose_both;
    log.target_mlp      = a.target_mlp;
    log.dit_path        = a.dit_path;
    log.dit_name        = a.dit_name;
    log.tensors         = a.tensors_dir;
    // Trigger word (T5): CLI flags win, else the variant's preprocess_meta.json.
    log.trigger          = a.trigger;
    log.trigger_position = a.trigger_position;
    dit_resolve_trigger(a.tensors_dir, &log.trigger, &log.trigger_position);
    if (!log.trigger.empty()) {
        fprintf(stderr, "[train-dit] trigger \"%s\" (%s) will be embedded in the adapter\n", log.trigger.c_str(),
                log.trigger_position.c_str());
    }
    log.crop_min        = a.crop_min;
    log.crop_max        = a.crop_max;
    log.lr              = a.lr;
    log.epochs          = a.epochs;
    log.grad_accum      = a.grad_accum;
    log.grad_clip       = a.grad_clip;
    log.weight_decay    = a.weight_decay;
    log.warmup_ratio    = a.warmup_ratio;
    log.seed            = a.seed;
    log.order           = a.order;
    log.loss_weighting  = a.loss_weighting;
    log.snr_gamma       = a.snr_gamma;
    log.t_bias          = a.t_bias;
    log.channel_balance = a.channel_balance;
    log.timestep_mu     = a.timestep_mu;
    log.timestep_sigma  = a.timestep_sigma;
    log.t_min           = a.t_min;
    log.t_max           = a.t_max;
    log.cfg_ratio       = a.cfg_ratio;
    log.genre_ratio     = a.genre_ratio;
    log.target_loss     = a.target_loss;

    if (!pm_mkdir_p(a.out_dir)) {
        lm_fatal("export", "cannot create the adapter output directory " + a.out_dir);
        return 1;
    }
    if (a.overwrite) {
        // BOTH layouts, not just this run's: a leftover lokr_weights.safetensors
        // next to a fresh LoRA export (or vice versa) wins some loader probes and
        // loses others, so the directory would serve the stale adapter.
        remove(lm_join(a.out_dir, "adapter_model.safetensors").c_str());
        remove(lm_join(a.out_dir, "adapter_config.json").c_str());
        remove(lm_join(a.out_dir, "lokr_weights.safetensors").c_str());
    }

    if (dit_has_stage(a, "train")) {
        const int rc = dit_train_stage(a, &log, &out);
        if (rc != 0) {
            return rc;
        }
    }

    if (dit_has_stage(a, "export")) {
        jl("{\"type\":\"stage\",\"stage\":\"export\",\"state\":\"begin\"}");
        const int64_t t_x0 = ggml_time_ms();
        if (!out.exported) {
            // §2.2: a `done` line means the run completed. Exiting 0 having written
            // nothing left the server's post-check as the only backstop and gave a
            // CLI user no signal at all. The adapter only exists in memory during
            // dit_train_stage, so `export` without `train` can never produce one.
            lm_fatal("export", dit_has_stage(a, "train") ?
                                   "the train stage produced no epoch, so there is no adapter to export" :
                                   "--stages export needs the train stage: the adapter only exists in memory during "
                                   "training, so there is nothing to write");
            return 1;
        } else {
            log.total_ms = (long long) (ggml_time_ms() - t_run0);
            if (!dit_write_train_log(a.out_dir, log)) {
                lm_fatal("export", "cannot write dit_train_log.json in " + a.out_dir);
                return 1;
            }
            long long bytes = 0;
            pm_stat_file(lm_join(a.out_dir, weights_leaf), &bytes, NULL);
            jl("{\"type\":\"export\",\"path\":\"%s\",\"tensors\":%d,\"bytes\":%lld,\"ms\":%lld}",
               lm_json_escape(a.out_dir).c_str(), out.export_tensors, bytes, (long long) (ggml_time_ms() - t_x0));
            jl("{\"type\":\"stage\",\"stage\":\"export\",\"state\":\"end\",\"ms\":%lld}",
               (long long) (ggml_time_ms() - t_x0));
        }
    }

    const long long run_ms = (long long) (ggml_time_ms() - t_run0);
    jl("{\"type\":\"done\",\"stages\":[%s],\"epochsRun\":%d,\"finalLoss\":%.6f,\"stoppedOnTarget\":%s,"
       "\"adapter\":\"%s\",\"samples\":%d,\"crop\":%d,\"layers\":%d,\"ms\":%lld}",
       stage_csv.c_str(), out.epochs_run, out.final_loss, out.stopped_on_target ? "true" : "false",
       lm_json_escape(a.out_dir).c_str(), out.samples, out.crop, out.layers, run_ms);

    fprintf(stderr, "[train-dit] done in %.1f s\n", (double) run_ms / 1000.0);
    return 0;
}
