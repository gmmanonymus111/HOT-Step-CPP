#pragma once
// mm3-dit-train-ckpt.h — gradient checkpointing for the MM3 flow-DiT trainer.
//
// WHY THIS EXISTS: attention scores are retained per layer for the backward, so
// the monolithic graph costs n_blocks * n_heads * S^2 * 4 bytes. Measured on a
// 5090 (32.6 GB, ~4 GB desktop):
//
//     crop  689 ( 8 s)   ~2.2 GB attention   20.4 GB total   fits
//     crop 1378 (16 s)   ~8.7 GB attention   31.8 GB total   831 MB spare
//     crop 2584 (30 s)  ~30.8 GB attention                   IMPOSSIBLE
//
// 30 s is SimpleTuner's own max_duration_seconds and matches our generation
// length, so it is the length we actually want to train at. ggml's
// GGML_OP_FLASH_ATTN_EXT has NO BACKWARD (see mm3-dit-train-graph.h), so the
// O(S^2) term cannot be removed — only paid in smaller instalments.
//
// The design is a direct port of dit-train-ckpt.h (ACE), which is proven by the
// S-C1 spike and its self-test rungs. A micro-batch runs as 1 + N graphs:
//
//   phase 1  ONE no-grad forward over the whole stack, ggml_cpy-ing the hidden
//            state ENTERING each segment into a persistent [E, S+1] buffer.
//   phase 2  segments LAST -> FIRST, each rebuilt WITH grads over its own block
//            range only:
//              * the LAST segment carries the real epilogue and the real loss;
//              * every EARLIER segment carries the surrogate loss
//                sum(seg_out (*) G), whose gradient w.r.t. seg_out is exactly
//                G = the boundary gradient the segment above just produced.
//            The boundary ENTERING a segment is ggml_set_param'd so that
//            ggml_build_backward_expand gives it a gradient slot, which
//            mm3_ckpt_fill_gacc points at a persistent buffer.
//
// Adapter gradients accumulate into the SAME LmOptim::acc[] across every
// segment (ggml's caller-supplied accumulators are in-place adds), so
// lm-optim.h is untouched and zero_grad is only called at window boundaries.
//
// ── ONE MM3-SPECIFIC DIFFERENCE FROM THE ACE DRIVER ────────────────────────
// ACE segments only the trained layer range and lets everything below it run
// once, frozen, in phase 1. MM3'S PROLOGUE CONTAINS A TRAINED PARAMETER
// (ad.proj_in, and the epilogue has ad.proj_out), so segment 0 must RECOMPUTE
// the prologue inside its own gradient graph rather than start from a frozen
// boundary. Hence boundaries are only needed for segments 1..N-1, and segment 0
// is the one with want_bnd == false.
//
// ACCEPTANCE GATE: at the same seed and crop, checkpointed and monolithic loss
// must match. A checkpointed run that quietly disagreed would be worse than no
// checkpointing at all. `--ckpt-verify` runs both and prints the difference.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "train/lm-optim.h"
#include "train/mm3-dit-train-graph.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

struct MM3CkptPlan {
    int              segments = 1;
    std::vector<int> bound;  // size segments+1; bound[0]=0, bound[segments]=n_blk
};

// Even split, remainder spread over the first segments. Segment count is a
// pure memory/compute trade: peak attention scales as 1/segments, total compute
// rises by roughly one extra forward over the stack.
static MM3CkptPlan mm3_ckpt_plan(int n_blk, int segments) {
    MM3CkptPlan p;
    p.segments = std::max(1, std::min(segments, n_blk));
    p.bound.assign((size_t) p.segments + 1, 0);
    const int per = n_blk / p.segments;
    const int rem = n_blk % p.segments;
    int       at  = 0;
    for (int j = 0; j < p.segments; j++) {
        p.bound[(size_t) j] = at;
        at += per + (j < rem ? 1 : 0);
    }
    p.bound[(size_t) p.segments] = n_blk;
    return p;
}

struct MM3CkptBufs {
    ggml_context *            ctx = nullptr;
    ggml_backend_buffer_t     buf = nullptr;
    std::vector<ggml_tensor *> bnd;      // [E, S+1]; index j is the INPUT to segment j
    ggml_tensor *             grad[2] = { nullptr, nullptr };
};

static void mm3_ckpt_free(MM3CkptBufs * b) {
    if (b->buf) ggml_backend_buffer_free(b->buf);
    if (b->ctx) ggml_free(b->ctx);
    b->ctx = nullptr;
    b->buf = nullptr;
    b->bnd.clear();
    b->grad[0] = b->grad[1] = nullptr;
}

// These live OUTSIDE the scheduler's arena on purpose: they must survive across
// the 1+N graph computes, and anything the scheduler allocates is recycled the
// moment its last consumer has run (the allocator-reuse trap Larry hit reading
// intermediates back — see the MOSS notes in agentcoordination.md).
static bool mm3_ckpt_alloc(MM3CkptBufs * b, ggml_backend_t backend, int64_t E, int64_t S1, int segments,
                           std::string * err) {
    mm3_ckpt_free(b);
    const size_t n_t = (size_t) segments + 2;   // bnd[0..segments-1] + 2 grads
    ggml_init_params ip = { ggml_tensor_overhead() * (n_t + 8), nullptr, true };
    b->ctx = ggml_init(ip);
    if (!b->ctx) { if (err) *err = "ckpt ctx init failed"; return false; }

    b->bnd.assign((size_t) segments, nullptr);
    // bnd[0] is never read (segment 0 recomputes the prologue) but is allocated
    // so indices line up with segment numbers and no off-by-one can hide.
    for (int j = 0; j < segments; j++) {
        b->bnd[(size_t) j] = ggml_new_tensor_2d(b->ctx, GGML_TYPE_F32, E, S1);
    }
    b->grad[0] = ggml_new_tensor_2d(b->ctx, GGML_TYPE_F32, E, S1);
    b->grad[1] = ggml_new_tensor_2d(b->ctx, GGML_TYPE_F32, E, S1);

    b->buf = ggml_backend_alloc_ctx_tensors(b->ctx, backend);
    if (!b->buf) { mm3_ckpt_free(b); if (err) *err = "ckpt buffer alloc failed"; return false; }
    return true;
}

// lm_optim_fill_gacc ASSERTS that every PARAM node is a known optimizer slot,
// so the boundary's PARAM flag must be hidden while it runs and its gacc entry
// patched afterwards. Straight from dit_ckpt_fill_gacc.
static void mm3_ckpt_fill_gacc(const LmOptim * o, ggml_cgraph * gf, ggml_tensor * bnd, ggml_tensor * bnd_grad,
                               std::vector<ggml_tensor *> * gacc) {
    const bool was_param = bnd && (bnd->flags & GGML_TENSOR_FLAG_PARAM) != 0;
    if (was_param) bnd->flags &= ~(int32_t) GGML_TENSOR_FLAG_PARAM;
    lm_optim_fill_gacc(o, gf, gacc);
    if (was_param) {
        bnd->flags |= GGML_TENSOR_FLAG_PARAM;
        const int n = ggml_graph_n_nodes(gf);
        for (int i = 0; i < n; i++) {
            if (ggml_graph_node(gf, i) == bnd) (*gacc)[(size_t) i] = bnd_grad;
        }
    }
}

// One micro-batch, checkpointed. Same contract as mm3_train_micro's backward
// path: adapter gradients land in opt->acc[], loss_out gets the real loss.
static bool mm3_train_micro_ckpt(MM3Model & m, ggml_backend_sched_t sched, const MM3TrainAdapters & ad,
                                 LmOptim * opt, const std::vector<float> & xt_h,
                                 const std::vector<float> & cond_h, const std::vector<float> & tgt_h,
                                 const std::vector<float> & fourier_h, int64_t crop,
                                 const MM3CkptPlan & plan, MM3CkptBufs * bufs, float * loss_out,
                                 std::string * err) {
    const MM3DitConfig & c  = m.synth_cfg.dit;
    const int64_t        IC = (int64_t) c.in_channels;
    const int64_t        CD = (int64_t) c.condition_dim;
    const int64_t        E  = (int64_t) c.embedding_length;
    const int            N  = plan.segments;

    const size_t MAX_NODES = 32768;
    const size_t meta = ggml_tensor_overhead() * (MAX_NODES * 4)
                      + ggml_graph_overhead_custom(MAX_NODES, true)
                      + (size_t) 32 * 1024 * 1024;

    // Rebuild inputs in a segment's own context and re-upload them. Called for
    // EVERY one of the 1+N computes, never once for the driver: ggml may
    // scribble on an input buffer during a compute, and a later segment that
    // read a clobbered input would be silently wrong. A few MB of host->device
    // against a whole segment's forward+backward is free insurance. This is the
    // same clobber rule the ACE driver documents, and the same one the
    // per-section adapter work hit.
    auto make_inputs = [&](ggml_context * ctx, MM3TrainInputs * in) -> bool {
        in->xt        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, crop, IC);
        in->cond      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, CD, crop);
        in->fourier   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) c.fourier_dim, 1);
        in->positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, crop + 1);
        in->vtarget   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, crop, IC);
        in->seq_zeros = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, crop + 1);
        if (!in->xt || !in->cond || !in->fourier || !in->positions || !in->vtarget || !in->seq_zeros)
            return false;
        ggml_set_input(in->xt); ggml_set_input(in->cond); ggml_set_input(in->fourier);
        ggml_set_input(in->positions); ggml_set_input(in->vtarget); ggml_set_input(in->seq_zeros);
        return true;
    };
    auto upload_inputs = [&](const MM3TrainInputs & in) {
        ggml_backend_tensor_set(in.xt,      xt_h.data(),      0, xt_h.size()      * sizeof(float));
        ggml_backend_tensor_set(in.cond,    cond_h.data(),    0, cond_h.size()    * sizeof(float));
        ggml_backend_tensor_set(in.vtarget, tgt_h.data(),     0, tgt_h.size()     * sizeof(float));
        ggml_backend_tensor_set(in.fourier, fourier_h.data(), 0, fourier_h.size() * sizeof(float));
        std::vector<int32_t> pos((size_t) crop + 1);
        for (int64_t i = 0; i <= crop; i++) pos[(size_t) i] = (int32_t) i;
        ggml_backend_tensor_set(in.positions, pos.data(), 0, pos.size() * sizeof(int32_t));
        std::vector<float> z((size_t) (E * (crop + 1)), 0.0f);
        ggml_backend_tensor_set(in.seq_zeros, z.data(), 0, z.size() * sizeof(float));
    };

    // ── phase 1: no-grad forward, boundaries copied out ─────────────────────
    if (N > 1) {
        ggml_init_params ip  = { meta, nullptr, true };
        ggml_context *   ctx = ggml_init(ip);
        if (!ctx) { if (err) *err = "ckpt phase1 ctx failed"; return false; }
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, MAX_NODES, false);
        MM3TrainInputs   in{};
        if (!gf || !make_inputs(ctx, &in)) { ggml_free(ctx); if (err) *err = "ckpt phase1 build failed"; return false; }

        ggml_tensor * h = mm3_dt_prologue(ctx, m, ad, in);
        for (int j = 0; j + 1 < N; j++) {
            h = mm3_dt_stack(ctx, m, ad, in, h, plan.bound[(size_t) j], plan.bound[(size_t) j + 1]);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, h, bufs->bnd[(size_t) j + 1]));
        }
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            ggml_free(ctx); if (err) *err = "ckpt phase1 alloc failed (out of VRAM?)"; return false;
        }
        upload_inputs(in);
        const bool ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
        ggml_free(ctx);
        if (!ok) { if (err) *err = "ckpt phase1 compute failed"; return false; }
    }

    // ── phase 2: segments LAST -> FIRST ─────────────────────────────────────
    int cur = 0;  // grad[cur] holds dL/d(this segment's OUTPUT)
    for (int j = N - 1; j >= 0; j--) {
        const bool    is_head  = (j == N - 1);
        const bool    want_bnd = (j > 0);   // segment 0 recomputes the prologue
        ggml_tensor * g_in     = bufs->grad[cur];
        ggml_tensor * g_out    = bufs->grad[1 - cur];
        if (want_bnd) ggml_backend_tensor_memset(g_out, 0, 0, ggml_nbytes(g_out));

        ggml_init_params ip  = { meta, nullptr, true };
        ggml_context *   ctx = ggml_init(ip);
        if (!ctx) { if (err) *err = "ckpt segment ctx failed"; return false; }
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, MAX_NODES, true);
        MM3TrainInputs   in{};
        if (!gf || !make_inputs(ctx, &in)) { ggml_free(ctx); if (err) *err = "ckpt segment build failed"; return false; }

        ggml_tensor * h = nullptr;
        ggml_tensor * bnd_in = nullptr;
        if (want_bnd) {
            bnd_in = bufs->bnd[(size_t) j];
            ggml_set_param(bnd_in);
            h = bnd_in;
        } else {
            h = mm3_dt_prologue(ctx, m, ad, in);   // carries ad.proj_in's gradient
        }
        h = mm3_dt_stack(ctx, m, ad, in, h, plan.bound[(size_t) j], plan.bound[(size_t) j + 1]);

        ggml_tensor * loss = nullptr;
        if (is_head) {
            ggml_tensor * pred = mm3_dt_epilogue(ctx, m, ad, h, crop);
            loss = mm3_dt_loss(ctx, pred, in.vtarget);
        } else {
            // dL/dh IS g_in, exactly — so sum(h * g_in) has the right gradient
            // and nothing above this segment needs rebuilding.
            loss = ggml_sum(ctx, ggml_mul(ctx, h, g_in));
            ggml_set_name(loss, "mm3_ckpt_surrogate");
        }
        ggml_set_loss(loss);
        ggml_set_output(loss);
        ggml_build_forward_expand(gf, loss);

        std::vector<ggml_tensor *> gacc;
        mm3_ckpt_fill_gacc(opt, gf, want_bnd ? bnd_in : nullptr, g_out, &gacc);
        ggml_build_backward_expand(ctx, gf, gacc.data());

        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            if (want_bnd) bnd_in->flags &= ~(int32_t) GGML_TENSOR_FLAG_PARAM;
            ggml_free(ctx); if (err) *err = "ckpt segment alloc failed (out of VRAM?)"; return false;
        }
        upload_inputs(in);
        // The head carries the real loss seed; every earlier segment is seeded
        // with 1 because g_in ALREADY carries the scale. Seeding again squares it.
        const float lg = 1.0f;
        ggml_backend_tensor_set(opt->t_lossgrad, &lg, 0, sizeof(float));

        const bool ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
        if (ok && is_head && loss_out) {
            ggml_backend_tensor_get(loss, loss_out, 0, sizeof(float));
        }
        if (want_bnd) {
            // Must not survive this graph: the boundary re-enters the next
            // micro-batch's phase-1 ggml_cpy as a plain destination buffer.
            bnd_in->flags &= ~(int32_t) GGML_TENSOR_FLAG_PARAM;
            cur = 1 - cur;
        }
        ggml_free(ctx);
        if (!ok) { if (err) *err = "ckpt segment compute failed"; return false; }
    }
    return true;
}
