#pragma once
// spike-s1.h — Rung 1: toy LoRA convergence + finite-difference gradient check.
//
// Model:  y = W@x + B@(scale * (A@x))     W frozen f32, A/B trainable f32
// Loss:   SSE = sum((y - target)^2), expressed as ggml ops + LOSS_TYPE_SUM
//         (SUM is the only ggml-opt loss type that is NOT rescaled by
//          opt_period, which matters for the finite-difference rung.)

#include "spike-common.h"

struct S1Model {
    int in = 0, out = 0, rank = 0, batch = 0;
    float alpha_over_r = 1.0f;

    ggml_context *        ctx_static = nullptr;
    ggml_backend_buffer_t buf_static = nullptr;
    ggml_context *        ctx_compute = nullptr;

    ggml_tensor * W = nullptr;  // [in, out]  frozen
    ggml_tensor * A = nullptr;  // [in, r]    trainable
    ggml_tensor * B = nullptr;  // [r, out]   trainable
    ggml_tensor * x = nullptr;  // [in, batch] input
    ggml_tensor * tgt = nullptr;  // [out, batch] constant target
    ggml_tensor * outputs = nullptr;  // [out, batch] squared error
};

static void s1_free(S1Model * m) {
    if (m->buf_static) ggml_backend_buffer_free(m->buf_static);
    if (m->ctx_static) ggml_free(m->ctx_static);
    if (m->ctx_compute) ggml_free(m->ctx_compute);
    *m = {};
}

// Build the static tensors + compute graph. Returns false on failure.
static bool s1_build(S1Model * m, ggml_backend_t backend, int in, int out, int rank, int batch, uint64_t seed) {
    m->in = in; m->out = out; m->rank = rank; m->batch = batch;
    m->alpha_over_r = 2.0f;  // alpha = 2r, the PEFT default ratio

    {
        ggml_init_params p = { 16 * ggml_tensor_overhead(), nullptr, true };
        m->ctx_static = ggml_init(p);
    }
    {
        // 3 graphs (gf / gb_grad / gb_opt) per ggml-opt's "Intended Usage".
        ggml_init_params p = { GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + 3 * ggml_graph_overhead(),
                               nullptr, true };
        m->ctx_compute = ggml_init(p);
    }

    m->W   = ggml_new_tensor_2d(m->ctx_static, GGML_TYPE_F32, in, out);
    m->A   = ggml_new_tensor_2d(m->ctx_static, GGML_TYPE_F32, in, rank);
    m->B   = ggml_new_tensor_2d(m->ctx_static, GGML_TYPE_F32, rank, out);
    m->x   = ggml_new_tensor_2d(m->ctx_static, GGML_TYPE_F32, in, batch);
    m->tgt = ggml_new_tensor_2d(m->ctx_static, GGML_TYPE_F32, out, batch);
    ggml_set_name(m->W, "W"); ggml_set_name(m->A, "lora_A"); ggml_set_name(m->B, "lora_B");
    ggml_set_name(m->x, "x"); ggml_set_name(m->tgt, "target");

    // ONLY A and B are marked trainable. W is never flagged -> frozen.
    ggml_set_param(m->A);
    ggml_set_param(m->B);

    m->buf_static = ggml_backend_alloc_ctx_tensors(m->ctx_static, backend);
    if (!m->buf_static) {
        fprintf(stderr, "[S1] FATAL: static buffer alloc failed\n");
        return false;
    }

    // ── init ────────────────────────────────────────────────────────────
    SpikeRng rng = { seed };
    std::vector<float> vW((size_t) in * out), vA((size_t) in * rank), vB((size_t) rank * out);
    std::vector<float> vx((size_t) in * batch);
    srng_fill_normal(&rng, vW, 1.0f / sqrtf((float) in));
    srng_fill_normal(&rng, vx, 1.0f);
    // PEFT init: A ~ N(0, 1/sqrt(in)), B = 0.
    srng_fill_normal(&rng, vA, 1.0f / sqrtf((float) in));
    std::fill(vB.begin(), vB.end(), 0.0f);

    // Ground-truth low-rank delta so the target is exactly reachable at rank r.
    std::vector<float> tA((size_t) in * rank), tB((size_t) rank * out);
    srng_fill_normal(&rng, tA, 1.0f / sqrtf((float) in));
    srng_fill_normal(&rng, tB, 1.0f / sqrtf((float) rank));

    // target[o,b] = sum_i W[i,o]*x[i,b] + sum_r tB[r,o] * (sum_i tA[i,r]*x[i,b])
    std::vector<float> vt((size_t) out * batch, 0.0f);
    for (int b = 0; b < batch; b++) {
        std::vector<float> h((size_t) rank, 0.0f);
        for (int r = 0; r < rank; r++) {
            double s = 0;
            for (int i = 0; i < in; i++) s += (double) tA[(size_t) r * in + i] * vx[(size_t) b * in + i];
            h[r] = (float) s;
        }
        for (int o = 0; o < out; o++) {
            double s = 0;
            for (int i = 0; i < in; i++) s += (double) vW[(size_t) o * in + i] * vx[(size_t) b * in + i];
            for (int r = 0; r < rank; r++) s += (double) tB[(size_t) o * rank + r] * h[r];
            vt[(size_t) b * out + o] = (float) s;
        }
    }

    spike_set(m->W, vW); spike_set(m->A, vA); spike_set(m->B, vB);
    spike_set(m->x, vx); spike_set(m->tgt, vt);

    // ── forward graph ───────────────────────────────────────────────────
    ggml_context * c = m->ctx_compute;
    ggml_tensor * base = ggml_mul_mat(c, m->W, m->x);                     // [out, batch]
    ggml_tensor * t    = ggml_mul_mat(c, m->A, m->x);                     // [rank, batch]
    t                  = ggml_scale(c, t, m->alpha_over_r);
    ggml_tensor * y    = ggml_add(c, base, ggml_mul_mat(c, m->B, t));     // [out, batch]
    ggml_tensor * e    = ggml_sub(c, y, m->tgt);
    m->outputs         = ggml_sqr(c, e);                                  // ggml-opt SUMs this
    ggml_set_name(m->outputs, "sq_err");
    return true;
}

// Re-seed A/B to their PEFT init (used between sub-runs).
static void s1_reset_params(S1Model * m, uint64_t seed) {
    SpikeRng rng = { seed };
    std::vector<float> vA((size_t) m->in * m->rank), vB((size_t) m->rank * m->out);
    srng_fill_normal(&rng, vA, 1.0f / sqrtf((float) m->in));
    std::fill(vB.begin(), vB.end(), 0.0f);
    spike_set(m->A, vA);
    spike_set(m->B, vB);
}

static ggml_opt_optimizer_params s1_opt_pars(void * ud) {
    return *(ggml_opt_optimizer_params *) ud;
}

// ─── S1a: convergence ───────────────────────────────────────────────────────

static bool s1_run_convergence(ggml_backend_sched_t sched, S1Model * m, int nsteps, float lr,
                               double * out_loss0, double * out_lossN) {
    ggml_opt_params op = ggml_opt_default_params(sched, GGML_OPT_LOSS_TYPE_SUM);
    op.ctx_compute = m->ctx_compute;
    op.inputs      = m->x;
    op.outputs     = m->outputs;
    op.build_type  = GGML_OPT_BUILD_TYPE_OPT;
    op.opt_period  = 1;

    ggml_opt_optimizer_params pars = ggml_opt_get_default_optimizer_params(nullptr);
    pars.adamw.alpha = lr;
    pars.adamw.wd    = 0.0f;
    op.get_opt_pars    = s1_opt_pars;
    op.get_opt_pars_ud = &pars;

    ggml_opt_context_t ctx = ggml_opt_init(op);
    if (!ctx) return false;

    const int64_t nel = ggml_nelements(m->outputs);
    double first = 0, last = 0;
    fprintf(stderr, "[S1a] step  loss(MSE)\n");
    for (int s = 0; s < nsteps; s++) {
        ggml_opt_alloc(ctx, /*backward=*/true);
        ggml_opt_eval(ctx, nullptr);
        float l = 0;
        ggml_backend_tensor_get(ggml_opt_loss(ctx), &l, 0, sizeof(float));
        double mse = (double) l / (double) nel;
        if (s == 0) first = mse;
        last = mse;
        if (s < 3 || s % (nsteps / 8 == 0 ? 1 : nsteps / 8) == 0 || s == nsteps - 1) {
            fprintf(stderr, "[S1a] %4d  %.6e\n", s, mse);
        }
    }
    ggml_opt_free(ctx);
    *out_loss0 = first;
    *out_lossN = last;
    return true;
}

// ─── S1b: finite differences ────────────────────────────────────────────────

// Forward-only loss evaluation (uses gf, no optimizer step).
static double s1_fwd_loss(ggml_opt_context_t ctx) {
    ggml_opt_alloc(ctx, /*backward=*/false);
    ggml_opt_eval(ctx, nullptr);
    float l = 0;
    ggml_backend_tensor_get(ggml_opt_loss(ctx), &l, 0, sizeof(float));
    return (double) l;
}

static bool s1_run_fd(ggml_backend_sched_t sched, S1Model * m, double * out_max_rel, int * out_ncheck) {
    ggml_opt_params op = ggml_opt_default_params(sched, GGML_OPT_LOSS_TYPE_SUM);
    op.ctx_compute = m->ctx_compute;
    op.inputs      = m->x;
    op.outputs     = m->outputs;
    op.build_type  = GGML_OPT_BUILD_TYPE_OPT;
    // opt_period huge => `accumulate` is true (grad accumulators live in the
    // persistent static buffer and are readable), and the optimizer step never
    // actually fires within this run, so params stay put across FD probes.
    // (build_type=GRAD alone is unusable with static graphs: ggml_opt_alloc
    //  computes build_type = (opt_i_next==0 ? OPT : GRAD) regardless of
    //  build_type_alloc, and gb_opt is NULL in that case -> assert.)
    op.opt_period  = 1 << 20;

    ggml_opt_optimizer_params pars = ggml_opt_get_default_optimizer_params(nullptr);
    op.get_opt_pars    = s1_opt_pars;
    op.get_opt_pars_ud = &pars;

    ggml_opt_context_t ctx = ggml_opt_init(op);
    if (!ctx) return false;

    // analytic gradients
    ggml_opt_alloc(ctx, /*backward=*/true);
    ggml_opt_eval(ctx, nullptr);

    ggml_tensor * gA = ggml_opt_grad_acc(ctx, m->A);
    ggml_tensor * gB = ggml_opt_grad_acc(ctx, m->B);
    if (!gA || !gB) {
        fprintf(stderr, "[S1b] FATAL: no grad accumulator for A/B\n");
        ggml_opt_free(ctx);
        return false;
    }
    std::vector<float> ga = spike_get(gA), gb = spike_get(gB);
    fprintf(stderr, "[S1b] |grad A| = %.6e  |grad B| = %.6e (finite: %d/%d)\n", spike_l2(ga), spike_l2(gb),
            (int) spike_all_finite(ga), (int) spike_all_finite(gb));

    struct Probe { ggml_tensor * t; std::vector<float> * g; const char * name; };
    std::vector<float> vA = spike_get(m->A), vB = spike_get(m->B);
    Probe probes[2] = { { m->A, &ga, "A" }, { m->B, &gb, "B" } };
    std::vector<float> * vals[2] = { &vA, &vB };

    double max_rel = 0.0;
    int    ncheck  = 0;
    for (int pi = 0; pi < 2; pi++) {
        std::vector<float> & v = *vals[pi];
        const int n = (int) v.size();
        const int stride = n > 8 ? n / 8 : 1;
        for (int i = 0; i < n; i += stride) {
            const float x0 = v[i];
            // relative-ish step; params here are O(0.1..1)
            const float eps = 1e-3f;
            v[i] = x0 + eps;  spike_set(probes[pi].t, v);
            double lp = s1_fwd_loss(ctx);
            v[i] = x0 - eps;  spike_set(probes[pi].t, v);
            double lm = s1_fwd_loss(ctx);
            v[i] = x0;        spike_set(probes[pi].t, v);

            double fd  = (lp - lm) / (2.0 * eps);
            double an  = (*probes[pi].g)[i];
            double den = std::max(1e-6, std::max(fabs(fd), fabs(an)));
            double rel = fabs(fd - an) / den;
            if (rel > max_rel) max_rel = rel;
            ncheck++;
            if (ncheck <= 6) {
                fprintf(stderr, "[S1b] %s[%d] analytic=%+.6e  fd=%+.6e  rel=%.3e\n", probes[pi].name, i, an, fd, rel);
            }
        }
    }
    ggml_opt_free(ctx);
    *out_max_rel = max_rel;
    *out_ncheck  = ncheck;
    return true;
}

// ─── driver ─────────────────────────────────────────────────────────────────

static int spike_s1(void) {
    BackendPair bp = backend_init("SPIKE");
    ggml_backend_sched_t sched = backend_sched_new(bp, 4096);

    fprintf(stderr, "\n=== S1a: toy LoRA convergence (in=64 out=64 r=8 batch=32) ===\n");
    S1Model m;
    if (!s1_build(&m, bp.backend, 64, 64, 8, 32, 0x1234abcdULL)) return 1;
    double l0 = 0, lN = 0;
    if (!s1_run_convergence(sched, &m, 600, 0.02f, &l0, &lN)) return 1;
    const double ratio = (lN > 0) ? l0 / lN : 1e30;
    const bool pass_a = ratio > 100.0;
    fprintf(stderr, "[S1a] loss0=%.6e lossN=%.6e  ratio=%.1fx  -> %s\n", l0, lN, ratio, pass_a ? "PASS" : "FAIL");
    s1_free(&m);

    fprintf(stderr, "\n=== S1b: finite-difference gradient check (in=8 out=6 r=3 batch=4) ===\n");
    S1Model f;
    if (!s1_build(&f, bp.backend, 8, 6, 3, 4, 0x55aa1234ULL)) return 1;
    // B=0 makes dL/dA identically zero; nudge B so both grads are meaningful.
    {
        SpikeRng rng = { 0xfeed1ULL };
        std::vector<float> vB((size_t) f.rank * f.out);
        srng_fill_normal(&rng, vB, 0.3f);
        spike_set(f.B, vB);
    }
    double max_rel = 0; int ncheck = 0;
    if (!s1_run_fd(sched, &f, &max_rel, &ncheck)) return 1;
    const bool pass_b = max_rel < 5e-2;
    fprintf(stderr, "[S1b] checked %d elements, max rel err = %.4e -> %s\n", ncheck, max_rel, pass_b ? "PASS" : "FAIL");
    s1_free(&f);

    ggml_backend_sched_free(sched);
    fprintf(stderr, "\n[S1] RESULT: %s\n", (pass_a && pass_b) ? "PASS" : "FAIL");
    return (pass_a && pass_b) ? 0 : 1;
}
