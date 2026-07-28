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

struct LmOptim {
    ggml_context *        ctx_grad = nullptr;
    ggml_backend_buffer_t buf_grad = nullptr;
    ggml_context *        ctx_mom  = nullptr;
    ggml_backend_buffer_t buf_mom  = nullptr;

    std::vector<ggml_tensor *> params;  // == LmLora::params (already ggml_set_param'd)
    std::vector<ggml_tensor *> acc, mom_m, mom_v;
    std::unordered_map<ggml_tensor *, int> param_slot;

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
    for (size_t j = 0; j < n; j++) {
        ggml_tensor * pj = o->params[j];
        o->acc[j]        = ggml_new_tensor_2d(o->ctx_grad, GGML_TYPE_F32, pj->ne[0], pj->ne[1]);
        o->mom_m[j]      = ggml_new_tensor_2d(o->ctx_mom, GGML_TYPE_F32, pj->ne[0], pj->ne[1]);
        o->mom_v[j]      = ggml_new_tensor_2d(o->ctx_mom, GGML_TYPE_F32, pj->ne[0], pj->ne[1]);
        char nm[96];
        snprintf(nm, sizeof(nm), "acc.%s", pj->name);
        ggml_set_name(o->acc[j], nm);
        snprintf(nm, sizeof(nm), "m.%s", pj->name);
        ggml_set_name(o->mom_m[j], nm);
        snprintf(nm, sizeof(nm), "v.%s", pj->name);
        ggml_set_name(o->mom_v[j], nm);
        GGML_ASSERT(ggml_are_same_shape(o->acc[j], pj));
        o->param_slot[pj] = (int) j;
    }

    o->buf_grad = ggml_backend_alloc_ctx_tensors(o->ctx_grad, backend);
    o->buf_mom  = ggml_backend_alloc_ctx_tensors(o->ctx_mom, backend);
    if (!o->buf_grad || !o->buf_mom) {
        *err = "optimizer state allocation failed";
        return false;
    }
    ggml_backend_buffer_clear(o->buf_grad, 0);
    ggml_backend_buffer_clear(o->buf_mom, 0);  // exactly once, never again

    // 392 params -> ~2400 nodes; 16 MB of tensor overhead is generous.
    o->arena.resize(16u << 20);
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
    ggml_cgraph * go = ggml_new_graph_custom(ctx, 8192, /*grads=*/false);

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

    // (c) AdamW, one node per parameter
    for (size_t j = 0; j < o->acc.size(); j++) {
        ggml_tensor * g = c ? ggml_mul(ctx, o->acc[j], c) : o->acc[j];  // [1] broadcasts
        ggml_build_forward_expand(go, ggml_opt_step_adamw(ctx, o->params[j], g, o->mom_m[j], o->mom_v[j], o->t_adamw));
    }

    // host side, immediately BEFORE running `go`
    o->opt_iter++;  // 1-based, matches ggml-opt's opt_ctx->iter
    const float lr = o->base_lr * lm_lr_lambda(o->opt_step, o->total_steps, o->warmup_steps);
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
