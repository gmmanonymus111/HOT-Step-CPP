#pragma once
// spike-dit2.h — ROUND 2 DiT spike driver.
//
// The gate spike-verdict.md §7b demanded before any DiT trainer is written:
//   D1 UNFUSED load of the full 32-layer decoder through dit_ggml_load()'s real
//      skip_fusion path; LoRA r16 on separate q/k/v/o (self) + q/o (cross).
//   D2 Real data: one crop window from the electriccallboy preprocess cache.
//   D3 Real rectified-flow loss (design §4.5), F32 mirror, manual attention.
//   D4 40+ optimizer steps on the lm-optim.h hand-rolled loop (clip 1.0,
//      warmup+cosine); bit-reproducibility across two runs.
//   D5 VRAM peak (trainer-owned AND device-wide) + ms/step at three crops.
//   D6 Feasibility map for 24 GB and 12 GB budgets.
//
// EVIDENCE CODE, NOT PRODUCT CODE.

#include "train/lm-optim.h"
#include "train/lm-vram.h"
#include "train/spike-dit2-data.h"
#include "train/spike-dit2-graph.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

struct D2Args {
    std::string dit_path;
    std::string sample_path;
    std::string stub_path;
    std::string export_dir;  // write a PEFT adapter after training (round-trip proof)
    int         crop        = 750;   // latent frames (750 ~ 30 s)
    int         crop_start  = 0;
    int         rank        = 16;
    int         alpha       = 32;
    int         steps       = 48;
    int         layer_lo    = 0;     // first LoRA-bearing layer
    float       lr          = 1e-4f;
    float       grad_clip   = 1.0f;
    float       weight_decay = 0.01f;
    float       mu          = -0.4f;  // logit-normal timestep
    float       sigma       = 1.0f;
    float       cfg_ratio   = 0.15f;
    float       fixed_t     = 0.0f;  // >0 -> freeze t and the noise draw
    uint64_t    seed        = 42;
    bool        sweep       = false;  // base-model loss(t) convention check
    bool        control     = false;  // sweep with a DELIBERATELY wrong temb (t -> 1-t)
    bool        mirror_ca_kv = false;
    // VERIFIER ADDITION (adversarial round-2 audit, default OFF — behaviour with
    // the flag absent is bit-identical to the implementer's build):
    // shuffle the velocity TARGET across latent frames while leaving xt built
    // from the real x0/x1.  Preserves E[v^2] exactly, destroys the input->target
    // relationship.  A run that descends identically on this control is only
    // memorising, and proves nothing about the objective.
    bool        shuffle_target = false;
    // VERIFIER ADDITION: load the --stub adapter in MERGE mode instead of runtime,
    // so the delta is folded into the base weights the mirror then copies.  Turns
    // the round-trip "the file loads" test into a numerical one: re-sweeping a
    // trained+exported adapter must reproduce the loss the trainer ended at.
    bool        merge_adapter  = false;
};

// ─── host-side timestep embedding (substitution S4) ─────────────────────────
//
// ggml_timestep_embedding has no backward, and nothing upstream of temb is
// trainable, so temb/tproj are computed once per timestep from the REAL
// time_embed / time_embed_r weights and uploaded as constants. This runs
// BEFORE the F32 mirror, because the mirror releases the BF16 weight buffer
// (and the temb weights with it — they are never needed again).
static bool d2_precompute_temb(DiTGGML * m, ggml_backend_sched_t sched, const std::vector<float> & tvals,
                               std::vector<std::vector<float>> * temb_out,
                               std::vector<std::vector<float>> * tproj_out) {
    const int H = m->cfg.hidden_size;

    ggml_context * ctxs;
    {
        ggml_init_params p = { 8 * ggml_tensor_overhead(), nullptr, true };
        ctxs               = ggml_init(p);
    }
    ggml_tensor * t_t = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 1);
    ggml_tensor * t_r = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 1);
    ggml_set_input(t_t);
    ggml_set_input(t_r);
    ggml_backend_buffer_t bufs = ggml_backend_alloc_ctx_tensors(ctxs, m->backend);
    if (!bufs) {
        fprintf(stderr, "[D2] FATAL: temb scalar alloc failed\n");
        return false;
    }

    std::vector<uint8_t> arena((size_t) 16 << 20);
    bool                 ok = true;
    temb_out->assign(tvals.size(), {});
    tproj_out->assign(tvals.size(), {});

    for (size_t i = 0; i < tvals.size() && ok; i++) {
        ggml_backend_tensor_set(t_t, &tvals[i], 0, sizeof(float));
        ggml_backend_tensor_set(t_r, &tvals[i], 0, sizeof(float));  // turbo: t_r == t, so t - t_r == 0

        ggml_init_params ip  = { arena.size(), arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 2048, false);

        ggml_tensor *tproj_t = nullptr, *tproj_r = nullptr;
        ggml_tensor * temb_t = dit_ggml_build_temb(ctx, m, &m->time_embed, t_t, &tproj_t, nullptr, nullptr, nullptr,
                                                   "_t");
        ggml_tensor * tdiff  = ggml_sub(ctx, t_t, t_r);
        ggml_tensor * temb_r = dit_ggml_build_temb(ctx, m, &m->time_embed_r, tdiff, &tproj_r, nullptr, nullptr, nullptr,
                                                   "_r");
        ggml_tensor * temb  = ggml_add(ctx, temb_t, temb_r);
        ggml_tensor * tproj = ggml_add(ctx, tproj_t, tproj_r);
        ggml_set_output(temb);
        ggml_set_output(tproj);
        ggml_build_forward_expand(gf, temb);
        ggml_build_forward_expand(gf, tproj);

        ggml_backend_sched_reset(sched);
        ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
        if (ok) {
            (*temb_out)[i].resize((size_t) H);
            (*tproj_out)[i].resize((size_t) 6 * H);
            ggml_backend_tensor_get(temb, (*temb_out)[i].data(), 0, (size_t) H * sizeof(float));
            ggml_backend_tensor_get(tproj, (*tproj_out)[i].data(), 0, (size_t) 6 * H * sizeof(float));
        }
        ggml_free(ctx);
    }

    ggml_backend_buffer_free(bufs);
    ggml_free(ctxs);
    if (!ok) {
        fprintf(stderr, "[D2] FATAL: temb precompute graph failed\n");
    }
    return ok;
}

// ─── the run ────────────────────────────────────────────────────────────────

static int spike_dit2(const D2Args & a) {
    fprintf(stderr,
            "\n=== DIT SPIKE ROUND 2 ===\n"
            "[D2] dit=%s\n[D2] sample=%s\n"
            "[D2] crop=%d frames (start %d) rank=%d alpha=%d steps=%d layers=%d.. lr=%.3g clip=%.2f seed=%llu\n"
            "[D2] flow: logit-normal mu=%.2f sigma=%.2f, cfg_ratio=%.2f%s\n",
            a.dit_path.c_str(), a.sample_path.c_str(), a.crop, a.crop_start, a.rank, a.alpha, a.steps, a.layer_lo,
            (double) a.lr, (double) a.grad_clip, (unsigned long long) a.seed, (double) a.mu, (double) a.sigma,
            (double) a.cfg_ratio, a.fixed_t > 0.0f ? "  [FIXED-t determinism mode]" : "");

    // ── D2: real sample ─────────────────────────────────────────────────
    D2Sample smp;
    if (!d2_load_sample(a.sample_path.c_str(), &smp)) {
        return 1;
    }

    // ── D1: UNFUSED load through the real skip_fusion path ──────────────
    const std::string prev_mode      = g_hotstep_params.adapter_mode;
    g_hotstep_params.adapter_mode    = a.merge_adapter ? "merge" : "runtime";  // VERIFIER ADDITION
    g_hotstep_params.adapters.clear();
    g_hotstep_params.adapter_sections.clear();

    DiTGGML m;
    {
        // The stub's shapes must match decoder.layers.0.self_attn.q_proj. We do
        // not know them before the load, so use the architecture's own values:
        // in = hidden_size, out = n_heads * head_dim.
        DiTGGMLConfig probe;
        if (!dit_ggml_load_config(&probe, a.dit_path.c_str())) {
            fprintf(stderr, "[D2] FATAL: cannot read DiT config from %s\n", a.dit_path.c_str());
            return 1;
        }
        // A directory is treated as an EXISTING adapter (used by the export
        // round-trip test); anything else is (re)written as the zero stub.
        if (pm_path_exists(a.stub_path + "/adapter_model.safetensors")) {
            fprintf(stderr, "[D2] unfused-load key: EXISTING adapter dir %s (round-trip load test)\n",
                    a.stub_path.c_str());
        } else if (!d2_stub_adapter(a.stub_path, probe.hidden_size, (int64_t) probe.n_heads * probe.head_dim)) {
            fprintf(stderr, "[D2] FATAL: cannot write the unfused-load stub adapter to %s\n", a.stub_path.c_str());
            return 1;
        } else {
            fprintf(stderr, "[D2] unfused-load key: zero rank-1 stub adapter at %s (adapter_mode=runtime)\n",
                    a.stub_path.c_str());
        }
    }

    const size_t vram_before = lm_vram_used_mb(nullptr);
    (void) vram_before;
    if (!dit_ggml_load(&m, a.dit_path.c_str(), a.stub_path.c_str(), 1.0f)) {
        fprintf(stderr, "[D2] FATAL: DiT load failed\n");
        g_hotstep_params.adapter_mode = prev_mode;
        return 1;
    }
    g_hotstep_params.adapter_mode = prev_mode;
    m.use_flash_attn              = false;  // SUBSTITUTION S1

    const DiTGGMLConfig & c = m.cfg;
    const int H = c.hidden_size, Oc = c.out_channels, P = c.patch_size, Lyr = c.n_layers;
    fprintf(stderr, "[D2] cfg: L=%d H=%d FFN=%d Nh=%d Nkv=%d D=%d in_ch=%d out_ch=%d patch=%d sw=%d convrot=%d\n", Lyr,
            H, c.intermediate_size, c.n_heads, c.n_kv_heads, c.head_dim, c.in_channels, Oc, P, c.sliding_window,
            (int) m.convrot.active);
    fprintf(stderr, "[D2] layer0 fusion after load: sa_qkv=%d sa_qk=%d ca_qkv=%d ca_kv=%d gate_up=%d  "
                    "(all zero == UNFUSED)\n",
            m.layers[0].sa_qkv != nullptr, m.layers[0].sa_qk != nullptr, m.layers[0].ca_qkv != nullptr,
            m.layers[0].ca_kv != nullptr, m.layers[0].gate_up != nullptr);
    fprintf(stderr, "[D2] VRAM after load (device-wide): %zu MB\n", lm_vram_used_mb(m.backend));

    // The stub's runtime delta is dead weight from here on.
    dit_lora_free(&m.lora);

    // Our graphs are far larger than the inference budget the loader sized.
    if (m.sched) {
        ggml_backend_sched_free(m.sched);
        m.sched = nullptr;
    }
    BackendPair bp;
    bp.backend     = m.backend;
    bp.cpu_backend = m.cpu_backend;
    bp.has_gpu     = m.backend != m.cpu_backend;
    m.sched        = backend_sched_new(bp, 65536);
    ggml_backend_sched_t sched = m.sched;

    // ── crop ────────────────────────────────────────────────────────────
    int crop = a.crop - (a.crop % P);
    int cs   = a.crop_start - (a.crop_start % P);
    if (cs + crop > smp.T) {
        cs = 0;
    }
    if (crop > smp.T) {
        crop = smp.T - (smp.T % P);
    }
    const int S     = crop / P;
    const int enc_S = smp.enc_S;
    fprintf(stderr, "[D2] crop: frames [%d, %d) of %d -> T=%d, patchified S=%d, enc_S=%d\n", cs, cs + crop, smp.T, crop,
            S, enc_S);

    const float * x0 = smp.lat.data() + (size_t) cs * smp.Oc;  // [crop, Oc]
    const size_t  n_lat = (size_t) crop * smp.Oc;

    // ── draw the whole schedule up front (3 independent RNG streams) ────
    LmRng rng_t, rng_noise, rng_cfg;
    lm_rng_seed(&rng_t, a.seed);
    lm_rng_seed(&rng_noise, a.seed ^ 0x9e3779b97f4a7c15ULL);
    lm_rng_seed(&rng_cfg, a.seed ^ 0xd1b54a32d192ed03ULL);

    std::vector<float> tvals;
    std::vector<bool>  cfg_drop;
    if (a.sweep) {
        for (int i = 1; i <= 9; i++) {
            tvals.push_back(0.1f * (float) i);
        }
        cfg_drop.assign(tvals.size(), false);
    } else {
        for (int i = 0; i < a.steps; i++) {
            tvals.push_back(a.fixed_t > 0.0f ? a.fixed_t : d2_logit_normal_t(&rng_t, a.mu, a.sigma));
            cfg_drop.push_back(a.fixed_t > 0.0f ? false : (lm_rng_uniform(&rng_cfg) < a.cfg_ratio));
        }
    }

    // ── temb/tproj (before the mirror releases the BF16 buffer) ─────────
    std::vector<std::vector<float>> temb_h, tproj_h;
    if (!d2_precompute_temb(&m, sched, tvals, &temb_h, &tproj_h)) {
        return 1;
    }
    fprintf(stderr, "[D2] precomputed temb[%d] + tproj[%d] for %zu timesteps (real time_embed weights)\n", H, 6 * H,
            tvals.size());

    // null_condition_emb -> host, for CFG dropout
    std::vector<float> null_emb;
    if (m.null_condition_emb) {
        const size_t ne = (size_t) ggml_nelements(m.null_condition_emb);
        null_emb.resize(ne);
        if (m.null_condition_emb->type == GGML_TYPE_BF16) {
            std::vector<uint16_t> b16(ne);
            ggml_backend_tensor_get(m.null_condition_emb, b16.data(), 0, ne * sizeof(uint16_t));
            for (size_t i = 0; i < ne; i++) {
                const uint32_t w = (uint32_t) b16[i] << 16;
                memcpy(&null_emb[i], &w, 4);
            }
        } else {
            ggml_backend_tensor_get(m.null_condition_emb, null_emb.data(), 0, ne * sizeof(float));
        }
        fprintf(stderr, "[D2] null_condition_emb present (%zu) — CFG dropout available\n", ne);
    } else {
        fprintf(stderr, "[D2] WARNING: no null_condition_emb — CFG dropout disabled\n");
    }

    // ── F32 mirror (SUBSTITUTION S2) ────────────────────────────────────
    D2Mirror mirror;
    {
        std::string err;
        const size_t before = lm_vram_used_mb(m.backend);
        if (!d2_build_mirror(&m, a.layer_lo, a.mirror_ca_kv, &mirror, &err)) {
            fprintf(stderr, "[D2] FATAL: mirror: %s\n", err.c_str());
            return 1;
        }
        fprintf(stderr, "[D2] VRAM device-wide: %zu MB before mirror -> %zu MB after (BF16 released)\n", before,
                lm_vram_used_mb(m.backend));
    }

    // ── static inputs ───────────────────────────────────────────────────
    ggml_context * ctxs;
    {
        ggml_init_params p = { 32 * ggml_tensor_overhead(), nullptr, true };
        ctxs               = ggml_init(p);
    }
    D2Inputs in;
    in.t_input        = ggml_new_tensor_2d(ctxs, GGML_TYPE_F32, c.in_channels, crop);
    in.t_enc          = ggml_new_tensor_2d(ctxs, GGML_TYPE_F32, smp.enc_H, enc_S);
    in.t_pos          = ggml_new_tensor_1d(ctxs, GGML_TYPE_I32, S);
    in.t_temb         = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, H);
    in.t_tproj        = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 6 * H);
    in.t_sa           = ggml_new_tensor_2d(ctxs, GGML_TYPE_F16, S, S);
    in.t_ca           = ggml_new_tensor_2d(ctxs, GGML_TYPE_F16, enc_S, S);
    in.t_vtgt         = ggml_new_tensor_2d(ctxs, GGML_TYPE_F32, Oc, crop);
    ggml_tensor * t_adamw    = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 7);
    ggml_tensor * t_lossgrad = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 1);
    ggml_tensor * t_clip     = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 1);
    ggml_tensor * t_eps      = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 1);
    ggml_tensor * t_gnorm2   = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 1);
    for (ggml_tensor * t : { in.t_input, in.t_enc, in.t_pos, in.t_temb, in.t_tproj, in.t_sa, in.t_ca, in.t_vtgt }) {
        ggml_set_input(t);
    }
    ggml_backend_buffer_t buf_static = ggml_backend_alloc_ctx_tensors(ctxs, m.backend);
    if (!buf_static) {
        fprintf(stderr, "[D2] FATAL: static input allocation failed\n");
        return 1;
    }
    {
        std::vector<int32_t> pos((size_t) S);
        for (int i = 0; i < S; i++) {
            pos[(size_t) i] = i;
        }
        ggml_backend_tensor_set(in.t_pos, pos.data(), 0, pos.size() * 4);
        std::vector<uint16_t> sa, ca;
        d2_sa_mask(S, c.sliding_window, &sa);
        d2_ca_mask(enc_S, S, smp.enc_mask, &ca);
        ggml_backend_tensor_set(in.t_sa, sa.data(), 0, sa.size() * 2);
        ggml_backend_tensor_set(in.t_ca, ca.data(), 0, ca.size() * 2);
        const float one = 1.0f, eps = 1e-6f, cl = a.grad_clip, lg = 1.0f;
        ggml_backend_tensor_set(t_eps, &eps, 0, 4);
        ggml_backend_tensor_set(t_clip, &cl, 0, 4);
        ggml_backend_tensor_set(t_lossgrad, &lg, 0, 4);
        (void) one;
    }

    // encoder hidden states (real, and the null broadcast for CFG dropout)
    std::vector<float> enc_real((size_t) smp.enc_H * enc_S), enc_null((size_t) smp.enc_H * enc_S, 0.0f);
    memcpy(enc_real.data(), smp.enc.data(), enc_real.size() * sizeof(float));
    if (!null_emb.empty() && (int) null_emb.size() == smp.enc_H) {
        for (int s = 0; s < enc_S; s++) {
            memcpy(&enc_null[(size_t) s * smp.enc_H], null_emb.data(), (size_t) smp.enc_H * sizeof(float));
        }
    }
    ggml_backend_tensor_set(in.t_enc, enc_real.data(), 0, enc_real.size() * sizeof(float));
    bool enc_is_null = false;

    // ── LoRA + optimizer ────────────────────────────────────────────────
    D2Lora L;
    {
        std::string err;
        if (!d2_lora_init(&L, &m, a.layer_lo, Lyr, a.rank, (float) a.alpha, a.seed, /*b_sigma=*/0.0f, &err)) {
            fprintf(stderr, "[D2] FATAL: LoRA init: %s\n", err.c_str());
            return 1;
        }
        // Report exactly which sites are trainable and whether their shapes are
        // exportable as PEFT (adapter-runtime.h:371 wants in==base.ne0, out==base.ne1).
        bool export_ok = true;
        for (int s = 0; s < D2_NSLOTS; s++) {
            ggml_tensor * w  = d2_site_weight(&m.layers[a.layer_lo], s);
            const D2Pair & p = L.layers[0][(size_t) s];
            const bool ok    = (p.A->ne[0] == w->ne[0]) && (p.B->ne[1] == w->ne[1]) && (p.A->ne[1] == a.rank) &&
                            (p.B->ne[0] == a.rank);
            export_ok = export_ok && ok;
            fprintf(stderr, "[D2]   site %-20s base[%lld,%lld]  A[%lld,%lld] B[%lld,%lld]  PEFT-shape=%s\n",
                    d2_slot_peft(s), (long long) w->ne[0], (long long) w->ne[1], (long long) p.A->ne[0],
                    (long long) p.A->ne[1], (long long) p.B->ne[0], (long long) p.B->ne[1], ok ? "OK" : "MISMATCH");
        }
        fprintf(stderr, "[D2]   exportable as a PEFT q/k/v/o adapter: %s\n", export_ok ? "YES" : "NO");
    }

    LmOptim opt;
    {
        std::string err;
        if (!lm_optim_init(&opt, L.params, m.backend, &err)) {
            fprintf(stderr, "[D2] FATAL: optimizer init: %s\n", err.c_str());
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
    opt.total_steps  = std::max(1, a.steps);
    opt.warmup_steps = std::max(1, a.steps / 20);

    // ── one step ────────────────────────────────────────────────────────
    std::vector<uint8_t> arena((size_t) 384 << 20);
    std::vector<float>   input_buf((size_t) c.in_channels * crop, 0.0f);
    D2Flow               flow;
    int                  graph_nodes = 0;

    // VERIFIER ADDITION: one fixed frame permutation for --shuffle-target.
    std::vector<int> perm((size_t) crop);
    if (a.shuffle_target) {
        for (int i = 0; i < crop; i++) {
            perm[(size_t) i] = i;
        }
        LmRng pr;
        lm_rng_seed(&pr, a.seed ^ 0xabcdef0123456789ULL);
        lm_rng_shuffle(&pr, perm);
        int fixed_pts = 0;
        for (int i = 0; i < crop; i++) {
            fixed_pts += (perm[(size_t) i] == i);
        }
        fprintf(stderr, "[D2] SHUFFLED-TARGET CONTROL: velocity target permuted across %d frames (%d fixed points); "
                        "xt still built from the real x0.  E[v^2] unchanged.\n",
                crop, fixed_pts);
    }
    std::vector<float> vshuf;

    // ti selects the flow sample (t, noise); tei selects the uploaded temb/tproj.
    // They differ only in --control, which feeds the model a deliberately WRONG
    // timestep embedding to show the objective's t-convention is load-bearing.
    auto micro2 = [&](size_t ti, size_t tei, bool backward, double * loss_out, double * base_out) -> bool {
        // rectified flow (design §4.5)
        LmRng noise = rng_noise;  // stream position advances only on the real path
        if (a.fixed_t > 0.0f || a.sweep) {
            lm_rng_seed(&noise, a.seed ^ 0x9e3779b97f4a7c15ULL);  // frozen noise draw
        }
        d2_flow_target(x0, n_lat, tvals[ti], &noise, &flow);
        if (!(a.fixed_t > 0.0f || a.sweep)) {
            rng_noise = noise;
        }
        if (a.shuffle_target) {  // VERIFIER ADDITION — permute the target only
            vshuf.resize(flow.v.size());
            for (int t = 0; t < crop; t++) {
                memcpy(&vshuf[(size_t) t * smp.Oc], &flow.v[(size_t) perm[(size_t) t] * smp.Oc],
                       (size_t) smp.Oc * sizeof(float));
            }
            flow.v.swap(vshuf);
        }
        if (base_out) {  // trivial predictor v_hat = 0
            double s = 0;
            for (size_t i = 0; i < flow.v.size(); i++) {
                s += (double) flow.v[i] * flow.v[i];
            }
            *base_out = s / (double) flow.v.size();
        }

        // pack [context(128) | xt(64)] per frame — hot-step-sampler.h:546/728
        for (int t = 0; t < crop; t++) {
            float * dst = &input_buf[(size_t) t * c.in_channels];
            memcpy(dst, &smp.ctxl[(size_t) (cs + t) * smp.Cc], (size_t) smp.Cc * sizeof(float));
            memcpy(dst + smp.Cc, &flow.xt[(size_t) t * smp.Oc], (size_t) smp.Oc * sizeof(float));
        }
        ggml_backend_tensor_set(in.t_input, input_buf.data(), 0, input_buf.size() * sizeof(float));
        ggml_backend_tensor_set(in.t_vtgt, flow.v.data(), 0, flow.v.size() * sizeof(float));
        ggml_backend_tensor_set(in.t_temb, temb_h[tei].data(), 0, temb_h[tei].size() * sizeof(float));
        ggml_backend_tensor_set(in.t_tproj, tproj_h[tei].data(), 0, tproj_h[tei].size() * sizeof(float));

        const bool want_null = ti < cfg_drop.size() && cfg_drop[ti] && !null_emb.empty();
        if (want_null != enc_is_null) {
            const std::vector<float> & src = want_null ? enc_null : enc_real;
            ggml_backend_tensor_set(in.t_enc, src.data(), 0, src.size() * sizeof(float));
            enc_is_null = want_null;
        }

        ggml_init_params ip  = { arena.size(), arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 65536, /*grads=*/backward);

        ggml_tensor * vpred = d2_build_forward(ctx, &m, &L, in, crop, enc_S);
        ggml_tensor * diff  = ggml_sub(ctx, vpred, in.t_vtgt);
        ggml_tensor * loss  = ggml_scale(ctx, ggml_sum(ctx, ggml_sqr(ctx, diff)), 1.0f / (float) n_lat);
        ggml_set_name(loss, "flow_mse");
        ggml_set_output(loss);
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

        ggml_backend_sched_reset(sched);
        const bool ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
        if (ok && loss_out) {
            float lv = 0.0f;
            ggml_backend_tensor_get(loss, &lv, 0, sizeof(float));
            *loss_out = (double) lv;
        }
        ggml_free(ctx);
        return ok;
    };
    auto micro = [&](size_t ti, bool backward, double * loss_out, double * base_out) -> bool {
        return micro2(ti, ti, backward, loss_out, base_out);
    };

    // ── sweep: does the base model already predict this velocity? ───────
    if (a.sweep) {
        fprintf(stderr, "\n[D2] BASE-MODEL CONVENTION CHECK (forward only; B=0 so the LoRA is a no-op)\n");
        fprintf(stderr, "[D2]   t      flow-MSE     E[v^2] (v_hat=0)   ratio%s\n",
                a.control ? "     WRONG-temb(1-t)   ratio" : "");
        for (size_t i = 0; i < tvals.size(); i++) {
            double lv = 0, bs = 0;
            if (!micro(i, false, &lv, &bs)) {
                fprintf(stderr, "[D2] FATAL: sweep compute failed at t=%.2f\n", (double) tvals[i]);
                return 1;
            }
            if (a.control) {
                double cv = 0;
                if (!micro2(i, tvals.size() - 1 - i, false, &cv, nullptr)) {
                    return 1;
                }
                fprintf(stderr, "[D2]   %.2f   %10.6f   %12.6f   %7.4f   %14.6f   %7.4f\n", (double) tvals[i], lv, bs,
                        lv / bs, cv, cv / bs);
            } else {
                fprintf(stderr, "[D2]   %.2f   %10.6f   %12.6f   %7.4f\n", (double) tvals[i], lv, bs, lv / bs);
            }
        }
        fprintf(stderr, "[D2] sweep VRAM device-wide %zu MB, graph %d nodes\n", lm_vram_used_mb(m.backend),
                graph_nodes);
        return 0;
    }

    // ── high-water probe + training loop (D4/D5) ────────────────────────
    LmVramTracker tracker;
    {
        double lv = 0;
        const int64_t t0 = ggml_time_ms();
        if (!micro(0, true, &lv, nullptr)) {
            fprintf(stderr, "[D2] FATAL: high-water probe failed — not enough VRAM for this crop\n");
            return 1;
        }
        lm_optim_zero_grad(&opt);
        size_t fixed = mirror.buf ? ggml_backend_buffer_get_size(mirror.buf) : mirror.bytes;
        fixed += ggml_backend_buffer_get_size(buf_static);
        if (L.buf) {
            fixed += ggml_backend_buffer_get_size(L.buf);
        }
        if (opt.buf_grad) {
            fixed += ggml_backend_buffer_get_size(opt.buf_grad);
        }
        if (opt.buf_mom) {
            fixed += ggml_backend_buffer_get_size(opt.buf_mom);
        }
        tracker.probe_baseline(m.backend, sched, fixed);
        fprintf(stderr,
                "\n[D2] graph: %d nodes (fwd+bwd)\n"
                "[D2] high-water probe: loss=%.6f, %lld ms, trainer-owned %zu MB, device-wide %zu MB\n",
                graph_nodes, lv, (long long) (ggml_time_ms() - t0), tracker.base_mb, lm_vram_used_mb(m.backend));
        fprintf(stderr, "[D2]   fixed buffers: mirror %.0f MB, static %.0f MB, lora %.0f MB, grad %.0f MB, mom %.0f MB;"
                        " sched arena %.0f MB\n",
                ggml_backend_buffer_get_size(mirror.buf) / 1048576.0,
                ggml_backend_buffer_get_size(buf_static) / 1048576.0, ggml_backend_buffer_get_size(L.buf) / 1048576.0,
                ggml_backend_buffer_get_size(opt.buf_grad) / 1048576.0,
                ggml_backend_buffer_get_size(opt.buf_mom) / 1048576.0,
                ggml_backend_sched_get_buffer_size(sched, m.backend) / 1048576.0);
    }

    fprintf(stderr, "\n[D2] step  t       loss        MA5        norm       nMA5       lr          |g|        clip  "
                    "   ms   trainerMB  deviceMB  cfg\n");
    std::vector<double> curve, ncurve;
    long long           ms_sum = 0;
    int                 ms_n   = 0;
    size_t              dev_peak = 0;
    for (int s = 0; s < a.steps; s++) {
        const int64_t t0 = ggml_time_ms();
        double        lv = 0, bs = 1.0;
        if (!micro((size_t) s, true, &lv, &bs)) {
            fprintf(stderr, "[D2] FATAL: compute failed at step %d\n", s);
            return 1;
        }
        LmStepStats st;
        if (!lm_optim_step(&opt, sched, &st)) {
            fprintf(stderr, "[D2] FATAL: optimizer step failed at step %d\n", s);
            return 1;
        }
        const long long ms = (long long) (ggml_time_ms() - t0);
        if (s >= 2) {  // skip warm-up allocations
            ms_sum += ms;
            ms_n++;
        }
        curve.push_back(lv);
        ncurve.push_back(bs > 0 ? lv / bs : lv);  // loss / E[v^2]: the t-independent view
        auto ma_of = [](const std::vector<double> & v) {
            const int k = (int) std::min<size_t>(5, v.size());
            double    q = 0;
            for (int i = 0; i < k; i++) {
                q += v[v.size() - 1 - i];
            }
            return q / k;
        };
        const double ma  = ma_of(curve);
        const double nma = ma_of(ncurve);
        const size_t tr  = tracker.sample();
        const size_t dv  = lm_vram_used_mb(m.backend);
        dev_peak         = std::max(dev_peak, dv);
        fprintf(stderr, "[D2] %4d  %.4f  %10.6f %10.6f %10.6f %10.6f  %.4e  %9.4f  %.4f  %4lld  %8zu  %8zu   %s\n", s,
                (double) tvals[(size_t) s], lv, ma, ncurve.back(), nma, (double) st.lr, (double) st.grad_norm,
                (double) st.clip, ms, tr, dv, cfg_drop[(size_t) s] ? "null" : "");
    }

    // ── summary ─────────────────────────────────────────────────────────
    const int    k    = (int) std::min<size_t>(10, curve.size());
    double       f5 = 0, l5 = 0, nf5 = 0, nl5 = 0;
    for (int i = 0; i < k; i++) {
        f5 += curve[(size_t) i];
        l5 += curve[curve.size() - 1 - i];
        nf5 += ncurve[(size_t) i];
        nl5 += ncurve[ncurve.size() - 1 - i];
    }
    f5 /= k;
    l5 /= k;
    nf5 /= k;
    nl5 /= k;
    fprintf(stderr, "[D2]   normalised loss/E[v^2]: first%d=%.6f  last%d=%.6f  ratio=%.4f\n", k, nf5, k, nl5,
            nl5 / nf5);
    fprintf(stderr,
            "\n[D2] SUMMARY crop=%d S=%d enc_S=%d layers=%d..%d rank=%d params=%zu\n"
            "[D2]   loss  first%d=%.6f  last%d=%.6f  ratio=%.4f  (min %.6f, max %.6f)\n"
            "[D2]   ms/step (steady, %d samples) = %.1f\n"
            "[D2]   VRAM trainer-owned peak %zu MB (baseline %zu, max delta %lld MB) | device-wide peak %zu MB\n"
            "[D2]   mirror %.1f MB (F32 promoted %.1f MB) | graph %d nodes\n",
            crop, S, enc_S, a.layer_lo, Lyr, a.rank, L.n_params, k, f5, k, l5, l5 / f5,
            *std::min_element(curve.begin(), curve.end()), *std::max_element(curve.begin(), curve.end()), ms_n,
            ms_n ? (double) ms_sum / ms_n : 0.0, tracker.peak_mb, tracker.base_mb, tracker.max_delta, dev_peak,
            mirror.bytes / 1048576.0, mirror.bytes_f32 / 1048576.0, graph_nodes);

    // machine-checkable reproducibility line
    fprintf(stderr, "[D2] CURVE");
    for (size_t i = 0; i < curve.size(); i++) {
        fprintf(stderr, " %.9e", curve[i]);
    }
    fprintf(stderr, "\n");

    if (!a.export_dir.empty()) {
        if (!d2_export_peft(&L, a.export_dir, a.dit_path)) {
            fprintf(stderr, "[D2] WARNING: PEFT export failed\n");
        }
    }

    lm_optim_free(&opt);
    d2_lora_free(&L);
    ggml_backend_buffer_free(buf_static);
    ggml_free(ctxs);
    d2_mirror_free(&mirror);
    dit_ggml_free(&m);
    return 0;
}
