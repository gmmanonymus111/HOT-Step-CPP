#pragma once
// lm-optim.h — the hand-rolled optimizer (L1).
//
// ggml-opt cannot be used: it structurally cannot do grad-norm clipping and
// cannot expose gradients in dynamic-graph mode (ggml_opt_eval nulls gb_opt,
// ggml_opt_grad_acc then derefs null — access violation reproduced in the
// Phase-0 spike). So we drive ggml_build_backward_expand ourselves with our own
// persistent accumulators and emit our own ggml_opt_step_adamw nodes.
//
// Buffer split (mandatory, plan §3.5.1):
//   buf_grad — accumulators; ggml_backend_buffer_clear() at the start of every
//              optimizer window.
//   buf_mom  — AdamW m/v; cleared exactly once at init and never again.
//   (scalars live in the caller's static buffer)
//
// ggml_graph_reset() zeroes AdamW momenta whenever the graph contains an
// OPT_STEP_ADAMW node, so we never call it.
//
// docs/plans/2026-07-27-lm-trainer-implementation.md §3.5

#include "train/lm-common.h"
#include "train/lm-graph.h"

#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

// Side-Step's lr_lambda (train_lm.py:268-272) with L6(b) applied:
// `step`, NOT `step + 1` — lr == 0 at optimizer step 0.
static inline float lm_lr_lambda(int step, int total, int warmup) {
    if (warmup > 0 && step < warmup) {
        return (float) step / (float) warmup;
    }
    const float p = (float) (step - warmup) / (float) std::max(1, total - warmup);
    return 0.1f + 0.9f * 0.5f * (1.0f + cosf(3.14159265358979f * p));
}

// ─── per-parameter update rule (2026-07-30) ─────────────────────────────────
//
// The optimizer is chosen PER PARAMETER, not globally, because Muon only means
// anything on a matrix: it orthogonalizes the update, and "orthogonalize" is
// vacuous on a LoKR w1 of [4,5]. Every published Muon puts the non-matrix
// parameters on AdamW for exactly this reason, so the hybrid is the design, not
// a compromise.
//
// This split is also the seam a Lua optimizer plugin would slot into later: a
// rule is (state buffers it needs) + (nodes it emits per parameter). Adding
// LM_RULE_LUA means implementing those two things against an interface that
// already has two real users, rather than guessing from AdamW alone.
enum LmOptRule {
    LM_RULE_ADAMW = 0,
    LM_RULE_MUON  = 1,
};

struct LmMuonCfg {
    // Muon's update is normalized by construction, so its LR does not mean what
    // AdamW's means. Kept as a multiplier on base_lr so the shared cosine
    // schedule (lm_lr_lambda) still drives both classes coherently.
    float lr_scale = 1.0f;
    float momentum = 0.95f;
    int   ns_steps = 5;
    bool  nesterov = true;
    // Below this on the SHORT side a matrix is not worth orthogonalizing: the
    // Newton-Schulz iteration would be shaping a subspace of that many
    // dimensions. LoKR w1 ([4,5] at factor 6) sits here and falls to AdamW.
    int   min_dim = 16;
    // < 0 inherits LmOptim::weight_decay. Decoupled either way.
    float wd = -1.0f;
};

// Quintic Newton-Schulz coefficients (Jordan 2024). Tuned so the iteration
// converges on the singular values from the [0,1] range the pre-normalization
// guarantees; they are NOT the coefficients of an exact matrix sign iteration.
static const float LM_MUON_NS_A = 3.4445f;
static const float LM_MUON_NS_B = -4.7750f;
static const float LM_MUON_NS_C = 2.0315f;

struct LmOptim {
    ggml_context *        ctx_grad = nullptr;
    ggml_backend_buffer_t buf_grad = nullptr;
    ggml_context *        ctx_mom  = nullptr;
    ggml_backend_buffer_t buf_mom  = nullptr;

    std::vector<ggml_tensor *> params;  // == LmLora::params (already ggml_set_param'd)
    std::vector<ggml_tensor *> acc, mom_m, mom_v;
    std::unordered_map<ggml_tensor *, int> param_slot;

    // Per-parameter rule, and the config that produced it. `mom_v[j]` is
    // nullptr for a Muon parameter — Muon carries ONE momentum buffer where
    // AdamW carries two, which is worth ~900 MB at LoKR dim512 (228.6M params).
    std::string            optimizer = "adamw";
    LmMuonCfg              muon;
    std::vector<uint8_t>   rule;
    int                    n_muon = 0;   // parameters actually on Muon
    int                    est_nodes = 0;  // sized in init, drives the graph cap

    // scalars, allocated by the caller in its static buffer
    ggml_tensor * t_adamw    = nullptr;  // [7] {alpha,beta1,beta2,eps,wd,beta1h,beta2h}
    ggml_tensor * t_lossgrad = nullptr;  // [1] 1/grad_accum
    ggml_tensor * t_clip     = nullptr;  // [1] grad_clip
    ggml_tensor * t_eps      = nullptr;  // [1] 1e-6
    ggml_tensor * t_gnorm2   = nullptr;  // [1] readback only

    float base_lr      = 1e-4f;
    float weight_decay = 0.01f;
    float grad_clip    = 1.0f;
    int   total_steps  = 1;
    int   warmup_steps = 1;
    int   opt_step     = 0;  // 0-based; drives the LR schedule
    int   opt_iter     = 0;  // 1-based; drives AdamW bias correction

    std::vector<uint8_t> arena;  // reused every optimizer step
};

// A parameter is Muon-eligible when it is a genuine 2-D matrix whose SHORT side
// is wide enough for orthogonalization to mean something.
static inline bool lm_muon_eligible(const ggml_tensor * p, const LmMuonCfg & c) {
    if (p->ne[2] != 1 || p->ne[3] != 1) {
        return false;
    }
    const int64_t lo = p->ne[0] < p->ne[1] ? p->ne[0] : p->ne[1];
    return lo >= (int64_t) c.min_dim;
}

// ─── Newton-Schulz orthogonalization, as ggml nodes ─────────────────────────
//
// Returns an approximately semi-orthogonal matrix with G's shape, computed
// entirely on the backend — the host never sees an element. Reference:
//
//   X = G / (||G||_F + eps)                     (puts the spectrum in [0,1])
//   repeat ns_steps:  A = X X^T
//                     B = b*A + c*A*A
//                     X = a*X + B X
//
// GGML INDEXING, because it is the whole difficulty here. A ggml tensor
// [ne0=C, ne1=R] is an R-row, C-column matrix with rows contiguous, and
// ggml_mul_mat(u, v) contracts over ne0: out(m,n) = sum_k u(k,m)*v(k,n).
// Therefore:
//   X X^T   -> mul_mat(X, X)                     [R,R], symmetric
//   A A     -> mul_mat(A, A)                     (equals A*A^T; A is symmetric)
//   B X     -> mul_mat(cont(transpose(X)), B)    [C,R], back in X's layout
//
// The reference implementation transposes when rows > cols so the Gram matrix
// is the smaller one; NS(G^T) == NS(G)^T, so this is free accuracy AND a large
// cost saving (a LoRA B of [out 2560, rank 128] would otherwise build a
// 2560x2560 Gram instead of 128x128).
static ggml_tensor * lm_muon_newton_schulz(ggml_context * ctx, ggml_tensor * G, int steps, ggml_tensor * t_eps) {
    const bool tall = G->ne[1] > G->ne[0];  // more rows than cols in matrix terms
    ggml_tensor * X = tall ? ggml_cont(ctx, ggml_transpose(ctx, G)) : G;

    // Frobenius normalization. [1] broadcasts against any shape (ggml_can_repeat).
    ggml_tensor * nrm = ggml_sqrt(ctx, ggml_sum(ctx, ggml_sqr(ctx, X)));
    X                 = ggml_div(ctx, X, ggml_add(ctx, nrm, t_eps));

    for (int i = 0; i < steps; i++) {
        ggml_tensor * A  = ggml_mul_mat(ctx, X, X);                       // [R,R]
        ggml_tensor * A2 = ggml_mul_mat(ctx, A, A);                       // [R,R]
        ggml_tensor * B  = ggml_add(ctx, ggml_scale(ctx, A, LM_MUON_NS_B),
                                    ggml_scale(ctx, A2, LM_MUON_NS_C));   // [R,R]
        ggml_tensor * BX = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, X)), B);
        X                = ggml_add(ctx, ggml_scale(ctx, X, LM_MUON_NS_A), BX);
    }
    return tall ? ggml_cont(ctx, ggml_transpose(ctx, X)) : X;
}

// Nodes emitted per Muon parameter, for sizing the optimizer graph. COUNTED
// FROM THE EMITTED GRAPH, not estimated — an earlier guess of 7 per iteration
// undersized the cap and ggml aborts on overflow (ggml.c:7001) partway through
// the first optimizer step, i.e. minutes into a run.
//
// Per NS iteration: mul_mat(X,X), mul_mat(A,A), 2 scale, 1 add, transpose+cont,
// mul_mat, scale, add = 10. Fixed: momentum (2), Nesterov (2), the Frobenius
// normalization sqr/sum/sqrt/add/div (5), shape scale (1), decoupled weight
// decay (2), two write-backs (2), and up to 4 for the tall-matrix transpose in
// and out.
static inline int lm_muon_nodes_per_param(int ns_steps) {
    return 20 + ns_steps * 10;
}

static void lm_optim_free(LmOptim * o) {
    if (o->buf_grad) {
        ggml_backend_buffer_free(o->buf_grad);
    }
    if (o->ctx_grad) {
        ggml_free(o->ctx_grad);
    }
    if (o->buf_mom) {
        ggml_backend_buffer_free(o->buf_mom);
    }
    if (o->ctx_mom) {
        ggml_free(o->ctx_mom);
    }
    o->buf_grad = nullptr;
    o->ctx_grad = nullptr;
    o->buf_mom  = nullptr;
    o->ctx_mom  = nullptr;
    o->acc.clear();
    o->mom_m.clear();
    o->mom_v.clear();
    o->params.clear();
    o->param_slot.clear();
    o->arena.clear();
    o->arena.shrink_to_fit();
}

// `params` must already be ggml_set_param'd and backend-allocated in a buffer
// that nothing clears (LmLora::buf, or the self-test's toy buffer).
static bool lm_optim_init(LmOptim * o, const std::vector<ggml_tensor *> & params, ggml_backend_t backend,
                          std::string * err) {
    o->params      = params;
    const size_t n = o->params.size();

    {
        ggml_init_params p = { (n + 8) * ggml_tensor_overhead(), nullptr, true };
        o->ctx_grad        = ggml_init(p);
    }
    {
        ggml_init_params p = { (2 * n + 8) * ggml_tensor_overhead(), nullptr, true };
        o->ctx_mom         = ggml_init(p);
    }
    if (!o->ctx_grad || !o->ctx_mom) {
        *err = "cannot create the optimizer contexts";
        return false;
    }

    o->acc.resize(n);
    o->mom_m.resize(n);
    o->mom_v.resize(n);
    o->rule.assign(n, (uint8_t) LM_RULE_ADAMW);
    o->n_muon = 0;
    const bool want_muon = (o->optimizer == "muon");
    for (size_t j = 0; j < n; j++) {
        ggml_tensor * pj = o->params[j];
        if (want_muon && lm_muon_eligible(pj, o->muon)) {
            o->rule[j] = (uint8_t) LM_RULE_MUON;
            o->n_muon++;
        }
        o->acc[j]   = ggml_new_tensor_2d(o->ctx_grad, GGML_TYPE_F32, pj->ne[0], pj->ne[1]);
        o->mom_m[j] = ggml_new_tensor_2d(o->ctx_mom, GGML_TYPE_F32, pj->ne[0], pj->ne[1]);
        // Muon carries one momentum buffer, AdamW two. Skipping v where it is
        // unused is the difference between 2x and 1x the parameter set in
        // optimizer state — ~900 MB at LoKR dim512. dit-vram.h still charges for
        // both, so the auto-fit stays CONSERVATIVE rather than wrong; teaching
        // it the rule split is a follow-up, not a correctness fix.
        o->mom_v[j] = (o->rule[j] == LM_RULE_MUON)
                          ? nullptr
                          : ggml_new_tensor_2d(o->ctx_mom, GGML_TYPE_F32, pj->ne[0], pj->ne[1]);
        char nm[96];
        snprintf(nm, sizeof(nm), "acc.%s", pj->name);
        ggml_set_name(o->acc[j], nm);
        snprintf(nm, sizeof(nm), "m.%s", pj->name);
        ggml_set_name(o->mom_m[j], nm);
        if (o->mom_v[j]) {
            snprintf(nm, sizeof(nm), "v.%s", pj->name);
            ggml_set_name(o->mom_v[j], nm);
        }
        GGML_ASSERT(ggml_are_same_shape(o->acc[j], pj));
        o->param_slot[pj] = (int) j;
    }
    if (want_muon) {
        fprintf(stderr, "[optim] muon: %d of %zu parameters on Muon (min_dim %d), %zu on AdamW\n", o->n_muon, n,
                o->muon.min_dim, n - (size_t) o->n_muon);
    }

    o->buf_grad = ggml_backend_alloc_ctx_tensors(o->ctx_grad, backend);
    o->buf_mom  = ggml_backend_alloc_ctx_tensors(o->ctx_mom, backend);
    if (!o->buf_grad || !o->buf_mom) {
        *err = "optimizer state allocation failed";
        return false;
    }
    ggml_backend_buffer_clear(o->buf_grad, 0);
    ggml_backend_buffer_clear(o->buf_mom, 0);  // exactly once, never again

    // Graph size. AdamW is ~6 nodes/param (392 params -> ~2400), for which 16 MB
    // was generous. Muon is ~50, and a LoKR dim512+MLP run has ~1000 parameters,
    // so both the node cap and the arena have to be derived rather than fixed —
    // an 8192 cap would abort partway through the first optimizer step.
    // n*8 covers the global-norm reduction (sqr + sum per parameter plus the
    // balanced-tree adds) and the AdamW nodes; +25% because overflowing the cap
    // is a hard abort mid-run, and the headroom costs only address space.
    o->est_nodes = (int) (((double) n * 8.0 + (double) o->n_muon * lm_muon_nodes_per_param(o->muon.ns_steps)) * 1.25) + 1024;
    const size_t cap   = (size_t) o->est_nodes;
    const size_t bytes = ggml_graph_overhead_custom(cap, false) + cap * (ggml_tensor_overhead() + 64) + (4u << 20);
    o->arena.resize(bytes > (16u << 20) ? bytes : (16u << 20));
    return true;
}

// Clear the accumulators for the next optimizer window.
static inline void lm_optim_zero_grad(LmOptim * o) {
    ggml_backend_buffer_clear(o->buf_grad, 0);
}

// Fill the gacc[] array for one forward graph: PARAM nodes -> our persistent
// accumulator, the LOSS node -> the persistent 1/grad_accum constant.
//
// ggml_build_backward_expand indexes grad_accs[] by FORWARD-GRAPH NODE INDEX
// (ggml.c: `if (grad_accs && grad_accs[i]) cgraph->grad_accs[ihash] = ...`),
// and ggml_set_param tensors ARE nodes, not leafs (ggml_visit_parents_graph).
static void lm_optim_fill_gacc(const LmOptim * o, ggml_cgraph * gf, std::vector<ggml_tensor *> * gacc) {
    const int n_nodes = ggml_graph_n_nodes(gf);
    gacc->assign((size_t) n_nodes, nullptr);
    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor * nd = ggml_graph_node(gf, i);
        if (nd->flags & GGML_TENSOR_FLAG_PARAM) {
            auto it = o->param_slot.find(nd);
            GGML_ASSERT(it != o->param_slot.end());
            (*gacc)[(size_t) i] = o->acc[(size_t) it->second];
        } else if (nd->flags & GGML_TENSOR_FLAG_LOSS) {
            (*gacc)[(size_t) i] = o->t_lossgrad;
        }
    }
}

struct LmStepStats {
    float lr        = 0.0f;
    float grad_norm = 0.0f;  // pre-clip global L2
    float clip      = 1.0f;  // factor actually applied
};

// One optimizer step: global norm + clip + AdamW in a single graph, with no
// host round-trip in the critical path. Zeroes the accumulators afterwards.
static bool lm_optim_step(LmOptim * o, ggml_backend_sched_t sched, LmStepStats * out) {
    ggml_init_params ip  = { o->arena.size(), o->arena.data(), true };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        return false;
    }
    ggml_cgraph * go = ggml_new_graph_custom(ctx, (size_t) (o->est_nodes > 0 ? o->est_nodes : 8192), /*grads=*/false);

    // (a) global squared L2 norm of every accumulator.
    //
    // Reduced as a BALANCED TREE, not the plan's left-fold: a left-fold is a
    // 392-deep serial chain of single-element adds with no available parallelism
    // in its tail, i.e. 392 dependent kernel launches per optimizer step (twice
    // that on 1.7B). Same node count (n-1 adds), same result, depth log2(n).
    if (o->acc.empty()) {
        ggml_free(ctx);
        return false;
    }
    std::vector<ggml_tensor *> sums;
    sums.reserve(o->acc.size());
    for (size_t j = 0; j < o->acc.size(); j++) {
        sums.push_back(ggml_sum(ctx, ggml_sqr(ctx, o->acc[j])));  // scalar
    }
    while (sums.size() > 1) {
        std::vector<ggml_tensor *> next;
        next.reserve((sums.size() + 1) / 2);
        for (size_t j = 0; j + 1 < sums.size(); j += 2) {
            next.push_back(ggml_add(ctx, sums[j], sums[j + 1]));
        }
        if (sums.size() % 2) {
            next.push_back(sums.back());
        }
        sums.swap(next);
    }
    ggml_tensor * gn2 = sums[0];
    ggml_build_forward_expand(go, ggml_cpy(ctx, gn2, o->t_gnorm2));  // 4-byte readback, LOGGING ONLY

    // (b) clip factor, entirely in-graph: c = clamp(clip / (sqrt(gn2) + eps), 0, 1)
    ggml_tensor * c = nullptr;
    if (o->grad_clip > 0.0f) {
        c = ggml_clamp(ctx, ggml_div(ctx, o->t_clip, ggml_add(ctx, ggml_sqrt(ctx, gn2), o->t_eps)), 0.0f, 1.0f);
    }

    // (c) the update, per parameter, by rule
    const float lr_now = o->base_lr * lm_lr_lambda(o->opt_step, o->total_steps, o->warmup_steps);
    const float mu_wd  = (o->muon.wd >= 0.0f) ? o->muon.wd : o->weight_decay;
    for (size_t j = 0; j < o->acc.size(); j++) {
        ggml_tensor * g = c ? ggml_mul(ctx, o->acc[j], c) : o->acc[j];  // [1] broadcasts

        if (o->rule[j] != LM_RULE_MUON) {
            ggml_build_forward_expand(
                go, ggml_opt_step_adamw(ctx, o->params[j], g, o->mom_m[j], o->mom_v[j], o->t_adamw));
            continue;
        }

        // Muon. WRITE-AFTER-READ ORDER IS LOAD-BEARING: `m_new` is computed from
        // the OLD momentum and is an ancestor of both write-backs, so the cpy
        // into `mom_m` can never be scheduled ahead of a node that still needs
        // the old value. ggml_opt_step_adamw avoids this by fusing; we have no
        // fused op, so the dependency has to be built deliberately.
        ggml_tensor * m_new = ggml_add(ctx, ggml_scale(ctx, o->mom_m[j], o->muon.momentum), g);
        ggml_tensor * u     = o->muon.nesterov ? ggml_add(ctx, g, ggml_scale(ctx, m_new, o->muon.momentum)) : m_new;

        ggml_tensor * ortho = lm_muon_newton_schulz(ctx, u, o->muon.ns_steps, o->t_eps);

        // Reference scale sqrt(max(1, rows/cols)); ne1 is rows, ne0 is cols.
        const float rows  = (float) o->params[j]->ne[1];
        const float cols  = (float) o->params[j]->ne[0];
        const float shape = sqrtf(rows > cols ? rows / cols : 1.0f);
        const float step  = -lr_now * o->muon.lr_scale * shape;

        // Decoupled weight decay, then the orthogonalized step, written back in
        // one cpy so the parameter is read exactly once.
        ggml_tensor * decayed = (mu_wd > 0.0f)
                                    ? ggml_scale(ctx, o->params[j], 1.0f - lr_now * o->muon.lr_scale * mu_wd)
                                    : o->params[j];
        ggml_tensor * p_new   = ggml_add(ctx, decayed, ggml_scale(ctx, ortho, step));
        ggml_build_forward_expand(go, ggml_cpy(ctx, p_new, o->params[j]));
        ggml_build_forward_expand(go, ggml_cpy(ctx, m_new, o->mom_m[j]));
    }

    // host side, immediately BEFORE running `go`. `lr` is the same value the
    // Muon branch above already baked into its ggml_scale constants — one
    // schedule drives both rules, so a mixed run cannot drift between classes.
    o->opt_iter++;  // 1-based, matches ggml-opt's opt_ctx->iter
    const float lr = lr_now;
    const float p7[7] = { lr,
                          0.9f,
                          0.999f,
                          1e-8f,
                          o->weight_decay,
                          1.0f / (1.0f - powf(0.9f, (float) o->opt_iter)),
                          1.0f / (1.0f - powf(0.999f, (float) o->opt_iter)) };
    ggml_backend_tensor_set(o->t_adamw, p7, 0, sizeof(p7));

    ggml_backend_sched_reset(sched);
    const bool ok = ggml_backend_sched_graph_compute(sched, go) == GGML_STATUS_SUCCESS;

    float gn2v = 0.0f;
    ggml_backend_tensor_get(o->t_gnorm2, &gn2v, 0, sizeof(float));
    const float gnorm = (gn2v > 0.0f) ? sqrtf(gn2v) : 0.0f;

    out->lr        = lr;
    out->grad_norm = gnorm;
    out->clip      = (o->grad_clip > 0.0f) ? std::min(1.0f, o->grad_clip / (gnorm + 1e-6f)) : 1.0f;

    lm_optim_zero_grad(o);
    o->opt_step++;
    ggml_free(ctx);
    return ok;
}
