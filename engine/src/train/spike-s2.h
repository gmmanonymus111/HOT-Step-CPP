#pragma once
// spike-s2.h — Rung 2: ONE real Qwen3-0.6B layer, real weights, full-sequence
// forward + backward + one AdamW step, LoRA on q/k/v/o/gate/up/down.

#include "spike-lm.h"

static ggml_opt_optimizer_params s2_opt_pars(void * ud) {
    return *(ggml_opt_optimizer_params *) ud;
}

struct S2Result {
    bool   ok          = false;
    double grad_l2_A   = 0;
    double grad_l2_B   = 0;
    int    n_zero_grad = 0;
    int    n_params    = 0;
    double param_delta = 0;
    long long fwd_bwd_ms = 0;
    size_t vram_mb     = 0;
    int    n_casts     = 0;
};

static int spike_s2(const char * lm_path, int S, int rank, bool cast_base_f32) {
    fprintf(stderr, "\n=== S2: one REAL Qwen3 layer (%s), S=%d rank=%d cast_base_f32=%d ===\n", lm_path, S, rank,
            (int) cast_base_f32);

    // no-fuse load: keeps q/k/v and gate/up individually addressable and
    // guarantees ggml_swiglu_split (fused ggml_swiglu has no backward).
    g_qwen3_load_no_fuse = true;

    Qwen3LM lm;
    const int64_t t_load0 = ggml_time_ms();
    if (!qw3lm_load(&lm, lm_path, /*max_seq_len=*/S + 8, /*n_kv_sets=*/1)) {
        fprintf(stderr, "[S2] FATAL: LM load failed\n");
        return 1;
    }
    fprintf(stderr, "[S2] LM loaded in %lld ms\n", (long long) (ggml_time_ms() - t_load0));
    const Qwen3LMConfig & c = lm.cfg;
    const size_t vram_after_load = spike_vram_used_mb(lm.backend);
    fprintf(stderr, "[S2] VRAM after LM load: %zu MB (device-wide, includes other processes)\n", vram_after_load);

    // ── LoRA on layer 0 only ────────────────────────────────────────────
    SpikeLora L;
    // b_sigma>0: with the strict PEFT B=0 init, dL/dA is identically zero at
    // step 0 (mathematically correct, but useless as a "grads are nonzero" test).
    if (!spike_lora_init(&L, &lm, 0, 1, rank, (float) (2 * rank), 0xbeefULL, 1e-2f)) {
        return 1;
    }

    // ── static tensors: hidden input, positions, mask, target ───────────
    const int H = c.hidden_size;
    ggml_context * ctx_static = nullptr;
    {
        ggml_init_params p = { 8 * ggml_tensor_overhead(), nullptr, true };
        ctx_static = ggml_init(p);
    }
    ggml_tensor * hin = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, H, S);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx_static, GGML_TYPE_I32, S);
    ggml_tensor * msk = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, S, S);
    ggml_tensor * tgt = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, H, S);
    ggml_set_name(hin, "hidden_in");
    ggml_set_input(hin); ggml_set_input(pos); ggml_set_input(msk); ggml_set_input(tgt);

    ggml_backend_buffer_t buf_static = ggml_backend_alloc_ctx_tensors(ctx_static, lm.backend);
    if (!buf_static) { fprintf(stderr, "[S2] FATAL: static alloc failed\n"); return 1; }

    {
        SpikeRng rng = { 0xc0ffeeULL };
        std::vector<float> h((size_t) H * S), t((size_t) H * S);
        srng_fill_normal(&rng, h, 1.0f);
        srng_fill_normal(&rng, t, 1.0f);
        spike_set(hin, h);
        spike_set(tgt, t);
        std::vector<int32_t> ip(S);
        for (int i = 0; i < S; i++) ip[i] = i;
        ggml_backend_tensor_set(pos, ip.data(), 0, ip.size() * sizeof(int32_t));
        std::vector<float> m = spike_causal_mask(S);
        spike_set(msk, m);
    }

    // ── compute graph ───────────────────────────────────────────────────
    ggml_context * ctx_compute = nullptr;
    {
        ggml_init_params p = { (size_t) 4096 * ggml_tensor_overhead() +
                                   8 * ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, true) + (1 << 20),
                               nullptr, true };
        ctx_compute = ggml_init(p);
    }

    SpikeFwdOpts fo;
    fo.cast_base_f32 = cast_base_f32;
    ggml_tensor * y   = spike_lm_layer(ctx_compute, c, &lm.layers[0], hin, pos, msk, S, &fo);
    ggml_tensor * out = ggml_sqr(ctx_compute, ggml_sub(ctx_compute, y, tgt));
    ggml_set_name(out, "sq_err");
    fprintf(stderr, "[S2] graph built, %d base-weight ggml_cast nodes\n", fo.n_casts);

    S2Result R;
    R.n_casts  = fo.n_casts;
    R.n_params = (int) L.params.size();

    // ── phase 1: gradients (opt_period huge => grads land in the persistent
    //    static buffer and no optimizer step ever fires) ──────────────────
    {
        ggml_opt_params op = ggml_opt_default_params(lm.sched, GGML_OPT_LOSS_TYPE_SUM);
        op.ctx_compute = ctx_compute;
        op.inputs      = hin;
        op.outputs     = out;
        op.build_type  = GGML_OPT_BUILD_TYPE_OPT;
        op.opt_period  = 1 << 20;
        ggml_opt_optimizer_params pars = ggml_opt_get_default_optimizer_params(nullptr);
        op.get_opt_pars = s2_opt_pars; op.get_opt_pars_ud = &pars;

        ggml_opt_context_t ctx = ggml_opt_init(op);
        fprintf(stderr, "[S2] backward graph built OK (no missing-backward abort)\n");

        const int64_t t0 = ggml_time_ms();
        ggml_opt_alloc(ctx, /*backward=*/true);
        ggml_opt_eval(ctx, nullptr);
        ggml_backend_sched_synchronize(lm.sched);
        R.fwd_bwd_ms = (long long) (ggml_time_ms() - t0);

        float loss = 0;
        ggml_backend_tensor_get(ggml_opt_loss(ctx), &loss, 0, sizeof(float));
        fprintf(stderr, "[S2] loss(SSE) = %.6e   fwd+bwd = %lld ms\n", (double) loss, R.fwd_bwd_ms);
        R.vram_mb = spike_vram_used_mb(lm.backend);

        bool all_finite = true;
        for (size_t i = 0; i < L.params.size(); i++) {
            ggml_tensor * g = ggml_opt_grad_acc(ctx, L.params[i]);
            if (!g) { fprintf(stderr, "[S2] FATAL: no grad for %s\n", L.params[i]->name); return 1; }
            std::vector<float> gv = spike_get(g);
            double n = spike_l2(gv);
            if (!spike_all_finite(gv)) { all_finite = false; fprintf(stderr, "[S2] NON-FINITE grad: %s\n", L.params[i]->name); }
            if (n == 0.0) { R.n_zero_grad++; fprintf(stderr, "[S2] ZERO grad: %s\n", L.params[i]->name); }
            if (strstr(L.params[i]->name, "lora_A")) R.grad_l2_A += n * n; else R.grad_l2_B += n * n;
            if (i < 14) fprintf(stderr, "[S2]   |g| %-22s = %.6e\n", L.params[i]->name, n);
        }
        R.grad_l2_A = sqrt(R.grad_l2_A);
        R.grad_l2_B = sqrt(R.grad_l2_B);
        fprintf(stderr, "[S2] total |grad A| = %.6e, |grad B| = %.6e, zero-grad tensors = %d/%d, all finite = %d\n",
                R.grad_l2_A, R.grad_l2_B, R.n_zero_grad, R.n_params, (int) all_finite);
        R.ok = all_finite && R.n_zero_grad == 0;
        ggml_opt_free(ctx);
    }

    // ── phase 2: one real AdamW step ────────────────────────────────────
    {
        std::vector<std::vector<float>> before;
        for (auto * p : L.params) before.push_back(spike_get(p));

        ggml_opt_params op = ggml_opt_default_params(lm.sched, GGML_OPT_LOSS_TYPE_SUM);
        op.ctx_compute = ctx_compute;
        op.inputs      = hin;
        op.outputs     = out;
        op.build_type  = GGML_OPT_BUILD_TYPE_OPT;
        op.opt_period  = 1;
        ggml_opt_optimizer_params pars = ggml_opt_get_default_optimizer_params(nullptr);
        pars.adamw.alpha = 1e-3f;
        op.get_opt_pars = s2_opt_pars; op.get_opt_pars_ud = &pars;

        ggml_opt_context_t ctx = ggml_opt_init(op);
        float l0 = 0, l1 = 0;
        ggml_opt_alloc(ctx, true); ggml_opt_eval(ctx, nullptr);
        ggml_backend_tensor_get(ggml_opt_loss(ctx), &l0, 0, sizeof(float));
        ggml_opt_alloc(ctx, true); ggml_opt_eval(ctx, nullptr);
        ggml_backend_tensor_get(ggml_opt_loss(ctx), &l1, 0, sizeof(float));
        ggml_backend_sched_synchronize(lm.sched);

        double delta = 0;
        bool   fin   = true;
        for (size_t i = 0; i < L.params.size(); i++) {
            std::vector<float> after = spike_get(L.params[i]);
            if (!spike_all_finite(after)) fin = false;
            for (size_t j = 0; j < after.size(); j++) delta += fabs((double) after[j] - before[i][j]);
        }
        R.param_delta = delta;
        fprintf(stderr, "[S2] AdamW: loss %.6e -> %.6e, sum|dparam| = %.6e, params finite = %d\n", (double) l0,
                (double) l1, delta, (int) fin);
        R.ok = R.ok && fin && delta > 0.0 && l1 < l0;
        ggml_opt_free(ctx);
    }

    ggml_free(ctx_compute);
    ggml_backend_buffer_free(buf_static);
    ggml_free(ctx_static);
    spike_lora_free(&L);

    fprintf(stderr,
            "\n[S2] SUMMARY  params=%d  |gA|=%.4e |gB|=%.4e  zero=%d  fwd+bwd=%lld ms  casts=%d  vram=%zu MB  -> %s\n",
            R.n_params, R.grad_l2_A, R.grad_l2_B, R.n_zero_grad, R.fwd_bwd_ms, R.n_casts, R.vram_mb,
            R.ok ? "PASS" : "FAIL");
    return R.ok ? 0 : 1;
}
