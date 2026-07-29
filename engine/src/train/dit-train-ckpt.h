#pragma once
// dit-train-ckpt.h — gradient checkpointing: hand-rolled segmented recompute
// (design C1/C2 + spike amendments A3/A4).
//
// ggml has no builtin checkpointing left in this tree, so a micro-batch runs as
// 1 + N graphs instead of one:
//
//   phase 1  ONE no-grad forward over the whole stack, ggml_cpy-ing the hidden
//            state that ENTERS each segment into a persistent [H,S,B] buffer.
//   phase 2  segments LAST -> FIRST. Each is rebuilt WITH grads over its own
//            layers only, starting from its boundary buffer:
//              * the LAST segment carries the real head + the real loss, exactly
//                as the monolithic graph does (t_lossgrad seeds it as always);
//              * every EARLIER segment carries the spike-s3 surrogate loss
//                sum(seg_out (*) G), whose gradient w.r.t. seg_out is exactly G
//                = the boundary gradient the segment above just produced. Its
//                seed is 1, NOT the micro-batch scale — G already carries it,
//                and seeding again would square it.
//            The boundary that ENTERS a segment is ggml_set_param'd, so
//            ggml_build_backward_expand gives it a gradient slot, which the A3
//            wrapper points at a persistent buffer.
//
// Adapter gradients accumulate into the SAME persistent LmOptim::acc[] across
// every segment compute (ggml's caller-supplied grad accumulators are in-place
// adds), so lm-optim.h is UNCHANGED and lm_optim_zero_grad is called only at
// window boundaries — never between segments.
//
// Only the TRAINED layer range [lora_lo, n_layers) is segmented. Layers below it
// hold no adapter parameters, so no gradient ever flows through them AND
// ggml_build_backward_expand asserts every graph has at least one PARAM node —
// they run once, in the phase-1 pre-stack, and enter segment 0 as its boundary.
//
// Proven end-to-end by the S-C1 spike (engine/src/train/spike-batch.h, 0.000e+00
// on both backends) and gated here by the SC1-SC3 self-test rungs.
//
// docs/plans/2026-07-29-dit-batching-checkpointing.md §0 C1-C4 / §2.5

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "train/dit-train-graph.h"
#include "train/lm-optim.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

// ─── segment plan ───────────────────────────────────────────────────────────
//
// `bound` has segments+1 entries; segment j is decoder layers
// [bound[j], bound[j+1]), evenly spread over the trained range. Segment j's
// INPUT is boundary buffer j; boundary 0 is the pre-stack output (proj_in plus
// every untrained layer), which is why there are `segments` buffers and not
// `segments - 1`.
struct DitCkptPlan {
    int              segments = 1;
    std::vector<int> bound;
};

static DitCkptPlan dit_ckpt_plan(int train_lo, int n_layers, int segments) {
    DitCkptPlan p;
    const int   K = std::max(1, n_layers - train_lo);
    p.segments    = std::max(1, std::min(segments, K));
    p.bound.assign((size_t) p.segments + 1, train_lo);
    for (int j = 0; j <= p.segments; j++) {
        p.bound[(size_t) j] = train_lo + (int) ((int64_t) K * (int64_t) j / (int64_t) p.segments);
    }
    return p;
}

// ─── persistent boundary buffers ────────────────────────────────────────────
//
// One backend allocation for the whole run, re-carved every micro-batch: the
// padded crop length varies with the batch (dit_batch_assemble pads to the batch
// maximum), and a boundary must be a REAL leaf tensor — ggml_set_param asserts
// op == GGML_OP_NONE, so a strided view of a fixed-size base is not an option.
// ggml_tallocr hands out the same addresses from offset 0 every time, so nothing
// allocates or frees on the device per step.
struct DitCkptBufs {
    ggml_backend_buffer_t buf   = nullptr;
    size_t                bytes = 0;
    int                   slots = 0;
    // re-created per micro-batch by dit_ckpt_begin
    ggml_context *             ctx     = nullptr;
    std::vector<ggml_tensor *> bnd;
    ggml_tensor *              grad[2] = { nullptr, nullptr };
};

// segments boundary activations + two ping-ponging boundary gradients.
static size_t dit_ckpt_bytes(int H, int S, int B, int segments) {
    const size_t one = sizeof(float) * (size_t) H * (size_t) S * (size_t) B;
    return one * (size_t) (segments + 2);
}

static void dit_ckpt_end(DitCkptBufs * b) {
    if (b->ctx) {
        ggml_free(b->ctx);
        b->ctx = nullptr;
    }
    b->bnd.clear();
    b->grad[0] = nullptr;
    b->grad[1] = nullptr;
}

static void dit_ckpt_free(DitCkptBufs * b) {
    dit_ckpt_end(b);
    if (b->buf) {
        ggml_backend_buffer_free(b->buf);
        b->buf = nullptr;
    }
    b->bytes = 0;
    b->slots = 0;
}

// Sized for the WIDEST micro-batch the run can build (S_max, B); a short crop or
// a short final micro-batch simply carves less of it.
static bool dit_ckpt_alloc(DitCkptBufs * b, ggml_backend_t backend, int H, int S_max, int B, int segments,
                           std::string * err) {
    dit_ckpt_free(b);
    b->slots = segments + 2;
    b->bytes = dit_ckpt_bytes(H, S_max, B, segments) + (size_t) b->slots * 256 + 1024;  // + alignment slack
    b->buf   = ggml_backend_alloc_buffer(backend, b->bytes);
    if (!b->buf) {
        char m[192];
        snprintf(m, sizeof(m), "cannot allocate %lld MB of checkpoint boundary buffers (%d segments)",
                 (long long) (b->bytes / 1048576), segments);
        *err = m;
        return false;
    }
    return true;
}

static bool dit_ckpt_begin(DitCkptBufs * b, int H, int S, int B, int segments) {
    dit_ckpt_end(b);
    if (!b->buf || segments + 2 > b->slots) {
        return false;
    }
    ggml_init_params p = { (size_t) (segments + 4) * ggml_tensor_overhead(), nullptr, /*no_alloc=*/true };
    b->ctx             = ggml_init(p);
    if (!b->ctx) {
        return false;
    }
    ggml_tallocr ta = ggml_tallocr_new(b->buf);
    b->bnd.assign((size_t) segments, nullptr);
    char nm[32];
    for (int j = 0; j < segments; j++) {
        ggml_tensor * t = ggml_new_tensor_3d(b->ctx, GGML_TYPE_F32, H, S, B);
        snprintf(nm, sizeof(nm), "ckpt.bnd%d", j);
        ggml_set_name(t, nm);
        if (ggml_tallocr_alloc(&ta, t) != GGML_STATUS_SUCCESS) {
            dit_ckpt_end(b);
            return false;
        }
        b->bnd[(size_t) j] = t;
    }
    for (int j = 0; j < 2; j++) {
        ggml_tensor * t = ggml_new_tensor_3d(b->ctx, GGML_TYPE_F32, H, S, B);
        snprintf(nm, sizeof(nm), "ckpt.grad%d", j);
        ggml_set_name(t, nm);
        if (ggml_tallocr_alloc(&ta, t) != GGML_STATUS_SUCCESS) {
            dit_ckpt_end(b);
            return false;
        }
        b->grad[j] = t;
    }
    return true;
}

// ─── amendment A3 ───────────────────────────────────────────────────────────
//
// lm_optim_fill_gacc() asserts that every PARAM-flagged node is a known
// optimizer slot, and a checkpoint boundary deliberately is not one. The flag is
// hidden for the duration of that call — the node list is already fixed by then,
// so nothing else changes — and the boundary is then handed its own persistent
// accumulator. lm-optim.h is untouched.
static void dit_ckpt_fill_gacc(const LmOptim * o, ggml_cgraph * gf, ggml_tensor * bnd, ggml_tensor * bnd_grad,
                               std::vector<ggml_tensor *> * gacc) {
    const bool was_param = bnd && (bnd->flags & GGML_TENSOR_FLAG_PARAM) != 0;
    if (was_param) {
        bnd->flags &= ~(int32_t) GGML_TENSOR_FLAG_PARAM;
    }
    lm_optim_fill_gacc(o, gf, gacc);
    if (was_param) {
        bnd->flags |= GGML_TENSOR_FLAG_PARAM;
        const int n = ggml_graph_n_nodes(gf);
        for (int i = 0; i < n; i++) {
            if (ggml_graph_node(gf, i) == bnd) {
                (*gacc)[(size_t) i] = bnd_grad;
            }
        }
    }
}

// ─── the segment driver ─────────────────────────────────────────────────────

struct DitCkptReq {
    DitTrainModel *        M     = nullptr;
    const DitAdapter *     ad    = nullptr;
    LmOptim *              opt   = nullptr;
    ggml_backend_sched_t   sched = nullptr;
    std::vector<uint8_t> * arena = nullptr;  // the caller's graph arena, reused per segment
    // Rebuilds the graph inputs in a segment's context. EVERY segment gets its
    // own views of the same static buffers.
    std::function<DitInputs(ggml_context *)> mk_inputs;
    // §3.0 clobber rule: re-uploads every graph INPUT buffer. Called before EVERY
    // one of the 1+N computes, not once for the whole driver — ggml is free to
    // scribble on an input buffer during a compute, and a segment that read a
    // clobbered input would be silently wrong. It is a host->device copy of a few
    // MB against a whole segment's forward+backward, so the insurance is free.
    // Must NOT touch t_lossgrad: the driver re-seeds that per segment below.
    // (The persistent boundary buffers are ours and are deliberately untouched.)
    std::function<void()> upload;

    int   T = 0, enc_S = 0, B = 1, Oc = 0;
    bool  chan_bal    = false;
    float gscale      = 0.0f;
    bool  want_report = false;
    // What t_lossgrad carries for the HEAD segment (the micro-batch scale). Every
    // earlier segment is seeded with 1 — see the header comment.
    float loss_scale = 1.0f;

    // out
    int      nodes        = 0;  // widest graph built (fwd, or fwd+bwd)
    int      graphs       = 0;
    double * bnd_grad_max = nullptr;  // optional |dL/dboundary|max readback (self-test only)
};

static bool dit_ckpt_micro_batch(DitCkptReq & r, DitCkptBufs * bufs, const DitCkptPlan & plan, double * loss_out,
                                 double * raw_out) {
    DitTrainModel *       M = r.M;
    const DiTGGMLConfig & c = M->m.cfg;
    const int             S = r.T / c.patch_size;
    const int             N = plan.segments;
    r.nodes                 = 0;
    r.graphs                = 0;

    auto track = [&](ggml_cgraph * gf) {
        const int n = ggml_graph_n_nodes(gf);
        r.nodes     = std::max(r.nodes, n);
        r.graphs++;
        GGML_ASSERT(n < 65536);
    };

    // ── phase 1: no-grad forward, boundaries copied out ──────────────────
    {
        if (r.upload) {
            r.upload();
        }
        ggml_init_params ip  = { r.arena->size(), r.arena->data(), true };
        ggml_context *   ctx = ggml_init(ip);
        if (!ctx) {
            return false;
        }
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 65536, /*grads=*/false);
        DitInputs     in = r.mk_inputs(ctx);
        ggml_tensor * h  = dit_train_proj_in(ctx, M, in, r.T, r.B, nullptr);
        ggml_tensor * e  = dit_train_cond(ctx, M, in, nullptr);
        // The untrained layers below the adapter: no PARAM, so no gradient ever
        // flows through them and they are never recomputed.
        h = dit_train_stack(ctx, M, r.ad, in, h, e, 0, plan.bound[0], S, r.enc_S, r.B, nullptr);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, h, bufs->bnd[0]));
        for (int j = 0; j + 1 < N; j++) {
            h = dit_train_stack(ctx, M, r.ad, in, h, e, plan.bound[(size_t) j], plan.bound[(size_t) j + 1], S,
                                r.enc_S, r.B, nullptr);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, h, bufs->bnd[(size_t) j + 1]));
        }
        track(gf);
        ggml_backend_sched_reset(r.sched);
        const bool ok = ggml_backend_sched_graph_compute(r.sched, gf) == GGML_STATUS_SUCCESS;
        ggml_free(ctx);
        if (!ok) {
            return false;
        }
    }

    // ── phase 2: segments LAST -> FIRST ──────────────────────────────────
    int cur = 0;  // grad[cur] holds dL/d(this segment's OUTPUT)
    for (int j = N - 1; j >= 0; j--) {
        const bool    is_head  = (j == N - 1);
        const bool    want_bnd = (j > 0);  // segment 0's input is the frozen pre-stack output
        ggml_tensor * g_in     = bufs->grad[cur];
        ggml_tensor * g_out    = bufs->grad[1 - cur];
        if (want_bnd) {
            // The gacc contributions are in-place ADDS onto whatever is live.
            ggml_backend_tensor_memset(g_out, 0, 0, ggml_nbytes(g_out));
        }
        // Inputs first, THEN the seed: r.upload() is not allowed to touch
        // t_lossgrad, but ordering it this way makes that harmless either way.
        if (r.upload) {
            r.upload();
        }
        const float lg = is_head ? r.loss_scale : 1.0f;
        ggml_backend_tensor_set(r.opt->t_lossgrad, &lg, 0, sizeof(float));

        ggml_init_params ip  = { r.arena->size(), r.arena->data(), true };
        ggml_context *   ctx = ggml_init(ip);
        if (!ctx) {
            return false;
        }
        ggml_cgraph * gf     = ggml_new_graph_custom(ctx, 65536, /*grads=*/true);
        DitInputs     in     = r.mk_inputs(ctx);
        ggml_tensor * bnd_in = bufs->bnd[(size_t) j];
        if (want_bnd) {
            ggml_set_param(bnd_in);
        }
        ggml_tensor * e = dit_train_cond(ctx, M, in, nullptr);  // A4: recomputed, never stored
        ggml_tensor * h = dit_train_stack(ctx, M, r.ad, in, bnd_in, e, plan.bound[(size_t) j],
                                          plan.bound[(size_t) j + 1], S, r.enc_S, r.B, nullptr);
        ggml_tensor * report = nullptr;
        ggml_tensor * loss   = nullptr;
        if (is_head) {
            ggml_tensor * vel = dit_train_head(ctx, M, h, in, r.T, r.B, nullptr);
            loss = dit_train_loss(ctx, vel, in, r.Oc, r.T, r.chan_bal, r.gscale, r.want_report ? &report : nullptr);
        } else {
            loss = ggml_sum(ctx, ggml_mul(ctx, h, g_in));  // dL/dh == g_in, exactly
            ggml_set_name(loss, "ckpt_surrogate");
            ggml_set_output(loss);
        }
        ggml_set_loss(loss);
        ggml_build_forward_expand(gf, loss);
        if (report) {
            ggml_build_forward_expand(gf, report);
        }
        std::vector<ggml_tensor *> gacc;
        dit_ckpt_fill_gacc(r.opt, gf, want_bnd ? bnd_in : nullptr, g_out, &gacc);
        ggml_build_backward_expand(ctx, gf, gacc.data());
        track(gf);

        ggml_backend_sched_reset(r.sched);
        bool ok = ggml_backend_sched_graph_compute(r.sched, gf) == GGML_STATUS_SUCCESS;
        if (ok && is_head && loss_out) {
            float lv = 0.0f;
            ggml_backend_tensor_get(loss, &lv, 0, sizeof(float));
            *loss_out = (double) lv;
            if (raw_out) {
                if (report) {
                    float rv = 0.0f;
                    ggml_backend_tensor_get(report, &rv, 0, sizeof(float));
                    *raw_out = (double) rv;
                } else {
                    *raw_out = (double) lv;
                }
            }
        }
        ggml_free(ctx);
        if (want_bnd) {
            // It re-enters the next micro-batch (and the next phase-1 ggml_cpy) as
            // a plain buffer, so the flag must not survive this graph.
            bnd_in->flags &= ~(int32_t) GGML_TENSOR_FLAG_PARAM;
            if (ok && r.bnd_grad_max && is_head) {
                std::vector<float> g((size_t) ggml_nelements(g_out));
                ggml_backend_tensor_get(g_out, g.data(), 0, g.size() * sizeof(float));
                double mx = 0.0;
                for (size_t k = 0; k < g.size(); k++) {
                    mx = std::max(mx, fabs((double) g[k]));
                }
                *r.bnd_grad_max = mx;
            }
            cur = 1 - cur;
        }
        if (!ok) {
            return false;
        }
    }
    return true;
}
