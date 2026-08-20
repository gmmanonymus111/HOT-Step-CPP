#pragma once
// minimax/mm3-rvq-encode.h — native audio->RVQ-codes encoder (open-RVQ 169M).
//
// HOT-Step file (does not exist upstream). Included by tools/ace-train.cpp only.
//
// ── WHY THIS EXISTS ─────────────────────────────────────────────────────────
//
// MiniMax-Music3 ships DECODE-side only for the audio-token path: there is no
// released audio -> codes encoder, which is why LM training needed one from the
// community. Every `.codes` file the MM3 LM trainer and `mm3-condition --codes`
// consume is produced by such an encoder, and until now that meant a WSL python
// exporter. This is the native port, so the training data path runs in-engine.
//
// The architecture is PurpleOrc's `V4Encoder` (their `train_v4.py`), which the
// whole current encoder lineage shares — PurpleOrc 53k, Mothersuperior
// pooled-v4/53k-pooled, our hotstep-v1/r2/r3. The GRAPH IS WEIGHTS-AGNOSTIC:
// the checkpoint is an argument, and the adopted one (Rob's ear, 2026-08-20) is
// 53k-pooled. Convert with engine/tools/convert-rvq-encoder.py.
//
//   DAV latents [128, L]  (mm3-dav-encode.h, 44.1 kHz, 86.13 Hz latent rate)
//     -> conv_in Conv1d(128 -> 1088, k=7, pad=3)
//     -> 3x ResBlock(dilation 1/3/9)
//     -> frame pooling: bmm(pool[128, Lw], x^T) + pos     -> [1088, 128 frames]
//     -> 8x pre-LN TransformerEncoderLayer(1088, 17 heads, ff 4352, gelu)
//     -> LayerNorm                                        (OUTSIDE the stack)
//     -> sem_head Linear(1088 -> 16384)   argmax          -> semantic code
//     -> DepthDecoder greedy over 7 acoustic books        -> acoustic codes
//
// ── THINGS THAT ARE EASY TO GET WRONG, AND WHERE EACH IS PINNED ─────────────
//
// 1. GELU IS THE ERF FORM, NOT THE TANH APPROXIMATION. `torch.nn.functional
//    .gelu` with the default `approximate='none'` is erf-exact; `ggml_gelu` is
//    the tanh approximation and differs by ~1e-3 relative. With a semantic
//    argmax margin whose 5th percentile is ~0.06 logits, that is enough to flip
//    codes. `ggml_gelu_erf` throughout — never "simplify" to ggml_gelu.
//
// 2. `GroupNorm(1, 1088)` IS NOT A LAYERNORM OVER CHANNELS. One group means the
//    mean/variance are taken over CHANNELS AND POSITIONS JOINTLY, per sample.
//    Reached here by reshaping [T, C] to [T, 1, C, 1] and calling
//    ggml_group_norm with n_groups = 1, which reduces over ne0*ne1 within the
//    group — i.e. over T and all C. Doing it as a per-position LayerNorm would
//    look plausible and be wrong.
//
// 3. THE ENCODER STACK IS PRE-LN AND CARRIES NO FINAL NORM. `nn.
//    TransformerEncoder(layer, 8)` is constructed without a `norm=`, so the
//    residual stream leaves the stack unnormalised and `norm_out` is applied by
//    V4Encoder itself. Adding a norm inside the stack (or dropping norm_out)
//    are both silent.
//
// 4. THE WINDOWING CONTRACT IS THE MODEL'S, NOT OURS. `frame_latent_starts` is
//    integer arithmetic over the DIT-chunk stitched timeline (chunk 200 frames,
//    hop 100, 345 latents per hop, frames owned from 25), reproduced exactly in
//    mm3_rvq_frame_starts(). The constants ride in the GGUF as KV so a future
//    checkpoint that changes them cannot silently disagree with C++ literals.
//    Second-guessing a model's own windowing is how misalignment hides.
//
// 5. WINDOWS ARE FIRST-WINS. Frames are covered by 128-frame windows stepping
//    by 128, plus a final window flush against the end; where the tail overlaps
//    the previous window the EARLIER window's codes stand. The reference does
//    this with a `done` mask and so does this port — taking the later window
//    would change codes near every track's end.
//
// 6. THE DEPTH DECODER IS SEQUENTIAL AND POSITION-OFFSET. Sequence per frame is
//    [ctx, sem, ac1..ac6] and head k reads position k+1, feeding back its own
//    argmax. Seven dependent steps; there is no parallel form.
//
// ── PARITY ─────────────────────────────────────────────────────────────────
//
// Unit gate: engine/tools/rvq-encoder-fixture.py writes golden feats / semantic
// logits / greedy codes for a deterministic synthetic window;
// `ace-train mm3-codes --fixture <f>` checks against it. Note the fixture also
// reports the semantic argmax MARGIN distribution: near-ties are real (p05
// ~0.06), so a handful of flipped frames at tiny margins is arithmetic order,
// not a port bug — the fixture check reports both counts separately.
//
// End-to-end gate (the one that matters): export the same tracks with the
// python exporter and with this, and diff the `.codes` files.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

// mm3-model.h supplies GGUFModel/gf_*, WeightCtx/wctx_*, MM3Loader, mm3_fmt,
// backend_init/backend_release — the same loader surface every MM3 module uses.
#include "mm3-model.h"

#define MM3_RVQ_MAX_NODES 4096

// ── Config ──────────────────────────────────────────────────────────────────

struct MM3RvqConfig {
    uint32_t d_model = 0, n_layers = 0, n_heads = 0, ff = 0;
    uint32_t sem_vocab = 0, ac_vocab = 0, n_ac = 0;
    uint32_t latent_channels = 0, frames = 0, latent_window_max = 0;
    // windowing (frame_latent_starts)
    uint32_t ratio_num = 0, ratio_den = 0;
    uint32_t chunk_frames = 0, chunk_hop = 0, hop_latents = 0, owned_from = 0;
    uint32_t depth_d = 0, depth_layers = 0, depth_heads = 0, depth_ff = 0;
    std::vector<int32_t> dilations;
};

struct MM3RvqLayer {
    ggml_tensor * n1_w = nullptr, * n1_b = nullptr;
    ggml_tensor * qkv_w = nullptr, * qkv_b = nullptr;
    ggml_tensor * o_w = nullptr, * o_b = nullptr;
    ggml_tensor * n2_w = nullptr, * n2_b = nullptr;
    ggml_tensor * l1_w = nullptr, * l1_b = nullptr;
    ggml_tensor * l2_w = nullptr, * l2_b = nullptr;
};

struct MM3RvqResBlock {
    ggml_tensor * norm_w = nullptr, * norm_b = nullptr;
    ggml_tensor * c1_w = nullptr, * c1_b = nullptr;
    ggml_tensor * c2_w = nullptr, * c2_b = nullptr;
};

struct MM3RvqWeights {
    ggml_tensor *               pos = nullptr;             // [d, frames]
    ggml_tensor *               conv_in_w = nullptr, * conv_in_b = nullptr;
    std::vector<MM3RvqResBlock> blocks;
    std::vector<MM3RvqLayer>    layers;
    ggml_tensor *               norm_out_w = nullptr, * norm_out_b = nullptr;
    ggml_tensor *               sem_w = nullptr, * sem_b = nullptr;
    // depth decoder
    ggml_tensor *               d_pos = nullptr;           // [dd, 8]
    ggml_tensor *               d_proj_w = nullptr, * d_proj_b = nullptr;
    ggml_tensor *               d_sem_emb = nullptr;       // [dd, sem_vocab]
    ggml_tensor *               d_ac_emb = nullptr;        // [dd, 6*ac_vocab]
    std::vector<MM3RvqLayer>    d_layers;
    std::vector<ggml_tensor *>  d_head_w, d_head_b;
};

struct MM3Rvq {
    bool           loaded = false;
    std::string    path;
    MM3RvqConfig   cfg;
    MM3RvqWeights  w;
    ggml_backend_t backend = nullptr, cpu_backend = nullptr;
    bool           backend_ref = false;
    WeightCtx      wctx = {};
    size_t         vram = 0;

    // cached encoder graph (one window; the shape never varies)
    ggml_context *        e_ctx = nullptr;
    ggml_cgraph *         e_graph = nullptr;
    ggml_gallocr_t        e_alloc = nullptr;
    ggml_tensor *         e_lat = nullptr, * e_pool = nullptr;
    ggml_tensor *         e_feats = nullptr, * e_logits = nullptr;

    // cached depth graph, rebuilt when (batch, step) changes
    ggml_context *        d_ctx = nullptr;
    ggml_cgraph *         d_graph = nullptr;
    ggml_gallocr_t        d_alloc = nullptr;
    ggml_tensor *         d_ctx_in = nullptr, * d_sem_in = nullptr, * d_ac_in = nullptr;
    ggml_tensor *         d_mask = nullptr, * d_out = nullptr;
    int64_t               d_N = 0;
    int                   d_step = -1;
};

// ── Windowing (the model's own contract — see design note 4) ────────────────

// Exactly train_v4.py::frame_latent_starts, in integer arithmetic. Returns
// n_frames+1 latent-index boundaries.
static std::vector<int64_t> mm3_rvq_frame_starts(const MM3RvqConfig & c, int64_t n_frames) {
    // n_dit_windows: how many stitched DiT chunks the track spans.
    const int64_t n_win = n_frames <= (int64_t) c.chunk_frames
                        ? 1
                        : 1 + (n_frames - (int64_t) c.chunk_frames + (int64_t) c.chunk_hop - 1) / (int64_t) c.chunk_hop;
    std::vector<int64_t> out((size_t) n_frames + 1);
    for (int64_t t = 0; t <= n_frames; t++) {
        int64_t k = (t - (int64_t) c.owned_from) / (int64_t) c.chunk_hop;
        // python's // is floor division; t < owned_from must floor to a
        // negative quotient before the clamp, which C++ truncation would round
        // the wrong way. The clamp lands both on 0, but only because the clamp
        // is here — do not remove it as "obviously non-negative".
        if (t < (int64_t) c.owned_from) k = -1;
        if (k < 0) k = 0;
        if (k > n_win - 1) k = n_win - 1;
        const int64_t tau = t - k * (int64_t) c.chunk_hop;
        const int64_t Fw  = std::min<int64_t>((int64_t) c.chunk_frames, n_frames - k * (int64_t) c.chunk_hop);
        const int64_t L   = Fw * (int64_t) c.ratio_num / (int64_t) c.ratio_den;
        out[(size_t) t]   = k * (int64_t) c.hop_latents + (tau * L + Fw - 1) / Fw;
    }
    return out;
}

// train_v4.py::pool_matrix, written straight into the [Lmax, frames] ggml
// layout (ne0 = latent index) the pooling mul_mat wants.
static void mm3_rvq_pool_matrix(const MM3RvqConfig & c, const int64_t * bounds, float * dst) {
    const int64_t F = (int64_t) c.frames, Lmax = (int64_t) c.latent_window_max;
    memset(dst, 0, (size_t) (F * Lmax) * sizeof(float));
    for (int64_t j = 0; j < F; j++) {
        const int64_t a = bounds[j], b = bounds[j + 1];
        if (b > a) {
            const float v = 1.0f / (float) (b - a);
            for (int64_t i = a; i < b && i < Lmax; i++) {
                dst[j * Lmax + i] = v;
            }
        }
    }
}

// ── Free ────────────────────────────────────────────────────────────────────

static void mm3_rvq_free_enc_graph(MM3Rvq * r) {
    if (r->e_alloc) { ggml_gallocr_free(r->e_alloc); r->e_alloc = nullptr; }
    if (r->e_ctx)   { ggml_free(r->e_ctx); r->e_ctx = nullptr; }
    r->e_graph = nullptr; r->e_lat = nullptr; r->e_pool = nullptr;
    r->e_feats = nullptr; r->e_logits = nullptr;
}

static void mm3_rvq_free_depth_graph(MM3Rvq * r) {
    if (r->d_alloc) { ggml_gallocr_free(r->d_alloc); r->d_alloc = nullptr; }
    if (r->d_ctx)   { ggml_free(r->d_ctx); r->d_ctx = nullptr; }
    r->d_graph = nullptr; r->d_ctx_in = nullptr; r->d_sem_in = nullptr;
    r->d_ac_in = nullptr; r->d_mask = nullptr; r->d_out = nullptr; r->d_N = 0; r->d_step = -1;
}

static void mm3_rvq_free(MM3Rvq * r) {
    mm3_rvq_free_enc_graph(r);
    mm3_rvq_free_depth_graph(r);
    if (r->wctx.buffer) { ggml_backend_buffer_free(r->wctx.buffer); r->wctx.buffer = nullptr; }
    if (r->wctx.ctx)    { ggml_free(r->wctx.ctx); r->wctx.ctx = nullptr; }
    r->wctx = {};
    if (r->backend_ref) {
        backend_release(r->backend, r->cpu_backend);
        r->backend = r->cpu_backend = nullptr;
        r->backend_ref = false;
    }
    r->w = {};
    r->loaded = false;
    r->vram = 0;
}

// ── Load ────────────────────────────────────────────────────────────────────

// One nn.TransformerEncoderLayer's twelve tensors. Names are the checkpoint's
// own (the converter is a pass-through — see its header).
static void mm3_rvq_load_layer(MM3Loader & ld, MM3RvqLayer & l, const std::string & prefix,
                               int64_t d, int64_t ff) {
    l.qkv_w = ld.req(prefix + ".self_attn.in_proj_weight", d, 3 * d);
    l.qkv_b = ld.req(prefix + ".self_attn.in_proj_bias", 3 * d);
    l.o_w   = ld.req(prefix + ".self_attn.out_proj.weight", d, d);
    l.o_b   = ld.req(prefix + ".self_attn.out_proj.bias", d);
    l.l1_w  = ld.req(prefix + ".linear1.weight", d, ff);
    l.l1_b  = ld.req(prefix + ".linear1.bias", ff);
    l.l2_w  = ld.req(prefix + ".linear2.weight", ff, d);
    l.l2_b  = ld.req(prefix + ".linear2.bias", d);
    l.n1_w  = ld.req(prefix + ".norm1.weight", d);
    l.n1_b  = ld.req(prefix + ".norm1.bias", d);
    l.n2_w  = ld.req(prefix + ".norm2.weight", d);
    l.n2_b  = ld.req(prefix + ".norm2.bias", d);
}

static bool mm3_rvq_load(MM3Rvq * r, const char * path, std::string * err) {
    if (r->loaded) {
        return true;
    }
    GGUFModel gf = {};
    if (!gf_load(&gf, path)) {
        if (err) *err = std::string("cannot open ") + path;
        return false;
    }

    MM3RvqConfig & c = r->cfg;
    c.d_model           = gf_get_u32(gf, "mm3rvq.embedding_length");
    c.n_layers          = gf_get_u32(gf, "mm3rvq.block_count");
    c.n_heads           = gf_get_u32(gf, "mm3rvq.head_count");
    c.ff                = gf_get_u32(gf, "mm3rvq.feed_forward_length");
    c.sem_vocab         = gf_get_u32(gf, "mm3rvq.sem_vocab_size");
    c.ac_vocab          = gf_get_u32(gf, "mm3rvq.ac_vocab_size");
    c.n_ac              = gf_get_u32(gf, "mm3rvq.num_acoustic");
    c.latent_channels   = gf_get_u32(gf, "mm3rvq.latent_channels");
    c.frames            = gf_get_u32(gf, "mm3rvq.frames");
    c.latent_window_max = gf_get_u32(gf, "mm3rvq.latent_window_max");
    c.ratio_num         = gf_get_u32(gf, "mm3rvq.ratio_num");
    c.ratio_den         = gf_get_u32(gf, "mm3rvq.ratio_den");
    c.chunk_frames      = gf_get_u32(gf, "mm3rvq.chunk_frames");
    c.chunk_hop         = gf_get_u32(gf, "mm3rvq.chunk_hop");
    c.hop_latents       = gf_get_u32(gf, "mm3rvq.hop_latents");
    c.owned_from        = gf_get_u32(gf, "mm3rvq.owned_from");
    c.depth_d           = gf_get_u32(gf, "mm3rvq.depth.embedding_length");
    c.depth_layers      = gf_get_u32(gf, "mm3rvq.depth.block_count");
    c.depth_heads       = gf_get_u32(gf, "mm3rvq.depth.head_count");
    c.depth_ff          = gf_get_u32(gf, "mm3rvq.depth.feed_forward_length");
    c.dilations         = mm3_get_i32_arr(gf, "mm3rvq.dilations");

    // Design note 1/3: a checkpoint that changed either convention would load
    // and produce plausible-but-wrong codes, so refuse rather than guess.
    const std::string norm_order = gf_get_str(gf, "mm3rvq.norm_order");
    const std::string activation = gf_get_str(gf, "mm3rvq.activation");
    if (norm_order != "pre" || activation != "gelu") {
        gf_close(&gf);
        if (err) *err = "mm3rvq: unsupported norm_order/activation '" + norm_order + "'/'" + activation + "'";
        return false;
    }
    if (!c.d_model || !c.n_layers || !c.n_heads || c.dilations.empty() || !c.frames) {
        gf_close(&gf);
        if (err) *err = "mm3rvq GGUF is missing required metadata";
        return false;
    }
    if (c.d_model % c.n_heads || c.depth_d % c.depth_heads) {
        gf_close(&gf);
        if (err) *err = "mm3rvq: d_model not divisible by head_count";
        return false;
    }

    const int64_t D  = c.d_model, FF = c.ff, DD = c.depth_d, DFF = c.depth_ff;
    const int     NB = (int) c.dilations.size(), NL = (int) c.n_layers, NDL = (int) c.depth_layers;

    std::vector<std::string> errs;
    wctx_init(&r->wctx, 16 + NB * 6 + (NL + NDL) * 12 + 8 + (int) c.n_ac * 2);
    MM3Loader ld{ &r->wctx, &gf, nullptr, &errs };
    MM3RvqWeights & w = r->w;

    w.pos       = ld.req("pos", D, (int64_t) c.frames);
    w.conv_in_w = ld.req("conv_in.weight", 7, (int64_t) c.latent_channels, D);
    w.conv_in_b = ld.req("conv_in.bias", D);

    w.blocks.assign((size_t) NB, MM3RvqResBlock{});
    for (int i = 0; i < NB; i++) {
        MM3RvqResBlock & b = w.blocks[(size_t) i];
        b.norm_w = ld.req(mm3_fmt("blocks.%d.norm.weight", i), D);
        b.norm_b = ld.req(mm3_fmt("blocks.%d.norm.bias", i), D);
        b.c1_w   = ld.req(mm3_fmt("blocks.%d.conv1.weight", i), 3, D, D);
        b.c1_b   = ld.req(mm3_fmt("blocks.%d.conv1.bias", i), D);
        b.c2_w   = ld.req(mm3_fmt("blocks.%d.conv2.weight", i), 1, D, D);
        b.c2_b   = ld.req(mm3_fmt("blocks.%d.conv2.bias", i), D);
    }

    w.layers.assign((size_t) NL, MM3RvqLayer{});
    for (int i = 0; i < NL; i++) {
        mm3_rvq_load_layer(ld, w.layers[(size_t) i], mm3_fmt("transformer.layers.%d", i), D, FF);
    }
    w.norm_out_w = ld.req("norm_out.weight", D);
    w.norm_out_b = ld.req("norm_out.bias", D);
    w.sem_w      = ld.req("sem_head.weight", D, (int64_t) c.sem_vocab);
    w.sem_b      = ld.req("sem_head.bias", (int64_t) c.sem_vocab);

    w.d_pos     = ld.req("depth.pos", DD, 8);
    w.d_proj_w  = ld.req("depth.proj.weight", D, DD);
    w.d_proj_b  = ld.req("depth.proj.bias", DD);
    w.d_sem_emb = ld.req("depth.sem_emb.weight", DD, (int64_t) c.sem_vocab);
    w.d_ac_emb  = ld.req("depth.ac_emb.weight", DD, (int64_t) (c.n_ac - 1) * (int64_t) c.ac_vocab);
    w.d_layers.assign((size_t) NDL, MM3RvqLayer{});
    for (int i = 0; i < NDL; i++) {
        mm3_rvq_load_layer(ld, w.d_layers[(size_t) i], mm3_fmt("depth.tr.layers.%d", i), DD, DFF);
    }
    w.d_head_w.assign((size_t) c.n_ac, nullptr);
    w.d_head_b.assign((size_t) c.n_ac, nullptr);
    for (uint32_t k = 0; k < c.n_ac; k++) {
        w.d_head_w[k] = ld.req(mm3_fmt("depth.heads.%d.weight", (int) k), DD, (int64_t) c.ac_vocab);
        w.d_head_b[k] = ld.req(mm3_fmt("depth.heads.%d.bias", (int) k), (int64_t) c.ac_vocab);
    }

    if (!errs.empty()) {
        gf_close(&gf);
        mm3_rvq_free(r);
        if (err) *err = errs[0];
        return false;
    }

    if (!r->backend_ref) {
        BackendPair bp = backend_init("MM3Rvq");
        r->backend     = bp.backend;
        r->cpu_backend = bp.cpu_backend;
        r->backend_ref = true;
    }
    for (const auto & pc : r->wctx.pending) {
        r->vram += pc.nbytes;
    }
    if (!wctx_alloc(&r->wctx, r->backend)) {
        gf_close(&gf);
        mm3_rvq_free(r);
        if (err) *err = "backend buffer allocation failed for the RVQ encoder";
        return false;
    }
    gf_close(&gf);
    r->path   = path;
    r->loaded = true;
    fprintf(stderr, "[MM3Rvq] loaded %s (%u layers, d=%u, depth %u x %u, %.1f MB)\n", path, c.n_layers,
            c.d_model, c.depth_layers, c.depth_d, (double) r->vram / (1024.0 * 1024.0));
    return true;
}

// ── Graph pieces ────────────────────────────────────────────────────────────

// Conv1d(+bias) with dilation. w [K, IC, OC], x [T, IC] -> [OL, OC].
// Explicit F32 im2col for the same reason mm3-dav-encode.h does it:
// ggml_conv_1d forces its im2col to F16.
static ggml_tensor * mm3_rvq_conv1d(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x,
                                    int pad, int dilation) {
    ggml_tensor * col = ggml_im2col(ctx, w, x, /*s0*/ 1, /*s1*/ 0, pad, 0, dilation, 0, /*is_2D*/ false,
                                    GGML_TYPE_F32);
    ggml_tensor * y = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, col, col->ne[0], col->ne[1] * col->ne[2]),
                                   ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]));
    if (b) {
        y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    }
    return y;
}

// LayerNorm over ne0 (the feature axis) + affine. eps is torch's default.
static ggml_tensor * mm3_rvq_layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * g, ggml_tensor * b) {
    ggml_tensor * y = ggml_norm(ctx, x, 1e-5f);
    y = ggml_mul(ctx, y, g);
    return ggml_add(ctx, y, b);
}

// GroupNorm(1, C) over x [T, C] — design note 2. Reduction is over T AND C.
static ggml_tensor * mm3_rvq_group_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * g, ggml_tensor * b) {
    const int64_t T = x->ne[0], C = x->ne[1];
    ggml_tensor * y = ggml_reshape_4d(ctx, x, T, 1, C, 1);
    y = ggml_group_norm(ctx, y, 1, 1e-5f);
    y = ggml_reshape_2d(ctx, y, T, C);
    // Affine is PER CHANNEL, i.e. along ne1 — broadcast over the T axis.
    y = ggml_mul(ctx, y, ggml_reshape_2d(ctx, g, 1, C));
    return ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, C));
}

// ResBlock: x + conv2(gelu(conv1(gelu(gn(x)))))  — note conv1 sees the FIRST
// gelu's output and conv2 the second's, matching train_v4.py exactly.
static ggml_tensor * mm3_rvq_res_block(ggml_context * ctx, const MM3RvqResBlock & b, ggml_tensor * x,
                                       int dilation) {
    ggml_tensor * h = mm3_rvq_group_norm(ctx, x, b.norm_w, b.norm_b);
    h = ggml_gelu_erf(ctx, h);
    h = mm3_rvq_conv1d(ctx, b.c1_w, b.c1_b, h, dilation, dilation);
    h = ggml_gelu_erf(ctx, h);
    h = mm3_rvq_conv1d(ctx, b.c2_w, b.c2_b, h, 0, 1);
    return ggml_add(ctx, x, h);
}

// One pre-LN nn.TransformerEncoderLayer over x [d, S, B] (B = 1 for the
// encoder, one-per-frame for the depth decoder, where every frame is an
// independent 8-token sequence and attention must never cross frames).
// `mask` may be null; when present it is additive [S(kv), >= S(q)] F32.
static ggml_tensor * mm3_rvq_enc_layer(ggml_context * ctx, const MM3RvqLayer & l, ggml_tensor * x,
                                       int n_heads, ggml_tensor * mask) {
    const int64_t d = x->ne[0], S = x->ne[1], B = x->ne[2], hd = d / n_heads;

    ggml_tensor * h = mm3_rvq_layer_norm(ctx, x, l.n1_w, l.n1_b);

    // in_proj is the q|k|v stack: rows [0,d) [d,2d) [2d,3d) of a [d, 3d]
    // weight. Each slice starts on a row boundary, so the views are contiguous.
    ggml_tensor * qw = ggml_view_2d(ctx, l.qkv_w, d, d, l.qkv_w->nb[1], 0);
    ggml_tensor * kw = ggml_view_2d(ctx, l.qkv_w, d, d, l.qkv_w->nb[1], (size_t) d * l.qkv_w->nb[1]);
    ggml_tensor * vw = ggml_view_2d(ctx, l.qkv_w, d, d, l.qkv_w->nb[1], (size_t) 2 * d * l.qkv_w->nb[1]);
    ggml_tensor * qb = ggml_view_1d(ctx, l.qkv_b, d, 0);
    ggml_tensor * kb = ggml_view_1d(ctx, l.qkv_b, d, (size_t) d * sizeof(float));
    ggml_tensor * vb = ggml_view_1d(ctx, l.qkv_b, d, (size_t) 2 * d * sizeof(float));

    ggml_tensor * q = ggml_add(ctx, ggml_mul_mat(ctx, qw, h), qb);         // [d, S, B]
    ggml_tensor * k = ggml_add(ctx, ggml_mul_mat(ctx, kw, h), kb);
    ggml_tensor * v = ggml_add(ctx, ggml_mul_mat(ctx, vw, h), vb);

    // [d, S, B] -> [hd, heads, S, B] -> [hd, S, heads, B]
    q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, q, hd, n_heads, S, B), 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, k, hd, n_heads, S, B), 0, 2, 1, 3));
    // v is transposed on the way in so the value contraction runs over kv.
    ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, v, hd, n_heads, S, B), 1, 2, 0, 3));

    ggml_tensor * scores = ggml_mul_mat(ctx, k, q);                        // [S(kv), S(q), heads, B]
    // torch's MultiheadAttention scales by 1/sqrt(head_dim).
    scores = ggml_soft_max_ext(ctx, scores, mask, 1.0f / sqrtf((float) hd), 0.0f);

    ggml_tensor * out = ggml_mul_mat(ctx, vt, scores);                     // [hd, S(q), heads, B]
    out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));              // [hd, heads, S, B]
    out = ggml_reshape_3d(ctx, out, d, S, B);
    out = ggml_add(ctx, ggml_mul_mat(ctx, l.o_w, out), l.o_b);

    x = ggml_add(ctx, x, out);

    ggml_tensor * f = mm3_rvq_layer_norm(ctx, x, l.n2_w, l.n2_b);
    f = ggml_add(ctx, ggml_mul_mat(ctx, l.l1_w, f), l.l1_b);
    f = ggml_gelu_erf(ctx, f);
    f = ggml_add(ctx, ggml_mul_mat(ctx, l.l2_w, f), l.l2_b);
    return ggml_add(ctx, x, f);
}

// ── Encoder graph (one 128-frame window) ────────────────────────────────────

static bool mm3_rvq_ensure_enc_graph(MM3Rvq * r, std::string * err) {
    if (r->e_graph) {
        return true;
    }
    const MM3RvqConfig & c = r->cfg;
    const int64_t Lmax = (int64_t) c.latent_window_max, CH = (int64_t) c.latent_channels;
    const int64_t F = (int64_t) c.frames, D = (int64_t) c.d_model;

    const size_t     meta = ggml_tensor_overhead() * MM3_RVQ_MAX_NODES + ggml_graph_overhead_custom(MM3_RVQ_MAX_NODES, false);
    ggml_init_params ip   = { meta, nullptr, true };
    r->e_ctx = ggml_init(ip);
    if (!r->e_ctx) {
        if (err) *err = "MM3Rvq: ggml_init failed";
        return false;
    }
    ggml_context * ctx = r->e_ctx;

    // [Lmax, CH]: ne0 is the latent index, which is also the layout the DAV
    // encoder's [CH][L] output slices into with one memcpy per channel.
    r->e_lat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Lmax, CH);
    ggml_set_name(r->e_lat, "latents");
    ggml_set_input(r->e_lat);
    // [Lmax, F] — pool_matrix's own row-major [F, Lmax] bytes, reinterpreted.
    r->e_pool = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Lmax, F);
    ggml_set_name(r->e_pool, "pool");
    ggml_set_input(r->e_pool);

    ggml_tensor * x = mm3_rvq_conv1d(ctx, r->w.conv_in_w, r->w.conv_in_b, r->e_lat, 3, 1);
    for (size_t i = 0; i < r->w.blocks.size(); i++) {
        x = mm3_rvq_res_block(ctx, r->w.blocks[i], x, c.dilations[i]);
    }
    // Frame pooling: out[c, f] = sum_t x[t, c] * pool[t, f]. Contracting over
    // ne0 of both operands is exactly ggml_mul_mat, and it lands feature-fastest
    // — the [d, S] layout the transformer wants, with no transpose.
    ggml_tensor * h = ggml_mul_mat(ctx, x, r->e_pool);                     // [D, F]
    h = ggml_add(ctx, h, r->w.pos);

    for (size_t i = 0; i < r->w.layers.size(); i++) {
        h = mm3_rvq_enc_layer(ctx, r->w.layers[i], h, (int) c.n_heads, nullptr);
    }
    h = mm3_rvq_layer_norm(ctx, h, r->w.norm_out_w, r->w.norm_out_b);      // design note 3

    r->e_feats = h;
    ggml_set_name(r->e_feats, "feats");
    ggml_set_output(r->e_feats);

    r->e_logits = ggml_add(ctx, ggml_mul_mat(ctx, r->w.sem_w, h), r->w.sem_b);   // [sem_vocab, F]
    ggml_set_name(r->e_logits, "sem_logits");
    ggml_set_output(r->e_logits);

    r->e_graph = ggml_new_graph_custom(ctx, MM3_RVQ_MAX_NODES, false);
    ggml_build_forward_expand(r->e_graph, r->e_feats);
    ggml_build_forward_expand(r->e_graph, r->e_logits);

    if (r->e_feats->ne[0] != D || r->e_feats->ne[1] != F) {
        if (err) *err = "MM3Rvq: encoder graph produced the wrong feature shape";
        mm3_rvq_free_enc_graph(r);
        return false;
    }

    r->e_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(r->backend));
    if (!r->e_alloc || !ggml_gallocr_reserve(r->e_alloc, r->e_graph)) {
        if (err) *err = "MM3Rvq: encoder graph allocation failed (out of VRAM?)";
        mm3_rvq_free_enc_graph(r);
        return false;
    }
    return true;
}

// One window. lat: [CH][Lmax] channel-major, pool: [F][Lmax] row-major (both
// exactly as the host builds them). feats out [F][D], logits out [F][sem_vocab].
static bool mm3_rvq_encode_window(MM3Rvq * r, const float * lat, const float * pool,
                                  float * feats, float * logits, std::string * err) {
    if (!mm3_rvq_ensure_enc_graph(r, err)) {
        return false;
    }
    if (!ggml_gallocr_alloc_graph(r->e_alloc, r->e_graph)) {
        if (err) *err = "MM3Rvq: gallocr_alloc_graph failed";
        return false;
    }
    ggml_backend_tensor_set(r->e_lat, lat, 0, ggml_nbytes(r->e_lat));
    ggml_backend_tensor_set(r->e_pool, pool, 0, ggml_nbytes(r->e_pool));
    if (ggml_backend_graph_compute(r->backend, r->e_graph) != GGML_STATUS_SUCCESS) {
        if (err) *err = "MM3Rvq: encoder graph compute failed";
        return false;
    }
    if (feats)  ggml_backend_tensor_get(r->e_feats, feats, 0, ggml_nbytes(r->e_feats));
    if (logits) ggml_backend_tensor_get(r->e_logits, logits, 0, ggml_nbytes(r->e_logits));
    return true;
}

// ── Depth decoder (greedy over the 7 acoustic books) ────────────────────────

// Step `step` of the chain over N frames: builds the [step+2]-token prefix and
// returns head[step]'s logits. The prefix is rebuilt from scratch each step —
// exactly what the reference does, and it keeps the graph a pure function of
// (ctx, sem, ac) with no cross-step state to get stale.
static bool mm3_rvq_ensure_depth_graph(MM3Rvq * r, int64_t N, int step, std::string * err) {
    if (r->d_graph && r->d_N == N && r->d_step == step) {
        return true;
    }
    mm3_rvq_free_depth_graph(r);

    const MM3RvqConfig & c = r->cfg;
    const int64_t DD = (int64_t) c.depth_d, S = step + 2;

    const size_t     meta = ggml_tensor_overhead() * MM3_RVQ_MAX_NODES + ggml_graph_overhead_custom(MM3_RVQ_MAX_NODES, false);
    ggml_init_params ip   = { meta, nullptr, true };
    r->d_ctx = ggml_init(ip);
    if (!r->d_ctx) {
        if (err) *err = "MM3Rvq: depth ggml_init failed";
        return false;
    }
    ggml_context * ctx = r->d_ctx;

    r->d_ctx_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) c.d_model, N);
    ggml_set_input(r->d_ctx_in);
    r->d_sem_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_set_input(r->d_sem_in);
    // Book priors 1..6, already offset by k*ac_vocab on the host (the reference
    // indexes a single [(n_ac-1)*ac_vocab] table that way). Step 0 needs none,
    // but the tensor is always created so the input set is shape-stable.
    r->d_ac_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N * (int64_t) (c.n_ac - 1));
    ggml_set_input(r->d_ac_in);

    // Sequence [DD, S, N]: pos0 = proj(ctx), pos1 = sem_emb, pos2.. = ac_emb.
    std::vector<ggml_tensor *> toks;
    toks.push_back(ggml_add(ctx, ggml_mul_mat(ctx, r->w.d_proj_w, r->d_ctx_in), r->w.d_proj_b));  // [DD, N]
    toks.push_back(ggml_get_rows(ctx, r->w.d_sem_emb, r->d_sem_in));                              // [DD, N]
    for (int k = 0; k + 2 < S; k++) {
        ggml_tensor * idx = ggml_view_1d(ctx, r->d_ac_in, N, (size_t) k * N * sizeof(int32_t));
        toks.push_back(ggml_get_rows(ctx, r->w.d_ac_emb, idx));
    }
    // -> [DD, N, S] -> [DD, S, N]: each frame is an independent 8-token
    // sequence, so N rides the batch axis and attention never crosses frames.
    ggml_tensor * seq = toks[0];
    for (size_t i = 1; i < toks.size(); i++) {
        seq = ggml_concat(ctx, seq, toks[i], 2);
    }
    seq = ggml_cont(ctx, ggml_permute(ctx, seq, 0, 2, 1, 3));               // [DD, S, N]
    ggml_tensor * pos = ggml_view_2d(ctx, r->w.d_pos, DD, S, r->w.d_pos->nb[1], 0);
    seq = ggml_add(ctx, seq, pos);

    // Causal mask [S(kv), S(q)] — F32 additive, exactly the shape
    // ggml_soft_max_ext asserts (ne0 == kv, ne1 >= q), broadcast over heads and
    // over the N frames on ne2/ne3. No KV-cache padding applies: this is a
    // plain soft_max path, not flash attention.
    r->d_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S, S);
    ggml_set_input(r->d_mask);
    ggml_set_name(r->d_mask, "causal");

    ggml_tensor * h = seq;
    for (size_t i = 0; i < r->w.d_layers.size(); i++) {
        h = mm3_rvq_enc_layer(ctx, r->w.d_layers[i], h, (int) c.depth_heads, r->d_mask);
    }
    // Head `step` reads position step+1 (the last one in this prefix).
    ggml_tensor * last = ggml_view_2d(ctx, h, DD, N, h->nb[2], (size_t) (S - 1) * h->nb[1]);
    last = ggml_cont(ctx, last);
    r->d_out = ggml_add(ctx, ggml_mul_mat(ctx, r->w.d_head_w[(size_t) step], last),
                        r->w.d_head_b[(size_t) step]);                     // [ac_vocab, N]
    ggml_set_output(r->d_out);

    r->d_graph = ggml_new_graph_custom(ctx, MM3_RVQ_MAX_NODES, false);
    ggml_build_forward_expand(r->d_graph, r->d_out);

    r->d_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(r->backend));
    if (!r->d_alloc || !ggml_gallocr_reserve(r->d_alloc, r->d_graph)) {
        if (err) *err = "MM3Rvq: depth graph allocation failed";
        mm3_rvq_free_depth_graph(r);
        return false;
    }
    r->d_N    = N;
    r->d_step = step;
    return true;
}

// feats [N][d_model], sem [N] -> ac [N][n_ac]. Greedy, seven dependent steps.
static bool mm3_rvq_depth_greedy(MM3Rvq * r, const float * feats, const int32_t * sem, int64_t N,
                                 int32_t * ac, std::string * err) {
    const MM3RvqConfig & c = r->cfg;
    const int64_t NAC = (int64_t) c.n_ac, AV = (int64_t) c.ac_vocab;

    // Priors for books 1..6, pre-offset by k*ac_vocab. Zeroed like the
    // reference's `torch.zeros` — steps beyond the current prefix are unused.
    std::vector<int32_t> prior((size_t) (N * (NAC - 1)), 0);
    for (int64_t k = 0; k < NAC - 1; k++) {
        for (int64_t i = 0; i < N; i++) {
            prior[(size_t) (k * N + i)] = (int32_t) (k * AV);
        }
    }
    std::vector<float> logits((size_t) (AV * N));

    // The causal mask is the same every step apart from its size.
    std::vector<float> maskbuf;

    for (int step = 0; step < (int) NAC; step++) {
        if (!mm3_rvq_ensure_depth_graph(r, N, step, err)) {
            return false;
        }
        if (!ggml_gallocr_alloc_graph(r->d_alloc, r->d_graph)) {
            if (err) *err = "MM3Rvq: depth gallocr_alloc_graph failed";
            return false;
        }
        ggml_backend_tensor_set(r->d_ctx_in, feats, 0, ggml_nbytes(r->d_ctx_in));
        ggml_backend_tensor_set(r->d_sem_in, sem, 0, ggml_nbytes(r->d_sem_in));
        // Step 0's prefix is [ctx, sem] only, so no ac_emb lookup enters the
        // graph and the allocator never gives d_ac_in a buffer. Uploading to an
        // unallocated tensor is an assert, not a no-op.
        if (step > 0) {
            ggml_backend_tensor_set(r->d_ac_in, prior.data(), 0, ggml_nbytes(r->d_ac_in));
        }

        const int64_t S = r->d_mask->ne[0];
        maskbuf.assign((size_t) (S * S), 0.0f);
        for (int64_t qi = 0; qi < S; qi++) {
            for (int64_t ki = 0; ki < S; ki++) {
                maskbuf[(size_t) (qi * S + ki)] = ki > qi ? -INFINITY : 0.0f;
            }
        }
        ggml_backend_tensor_set(r->d_mask, maskbuf.data(), 0, ggml_nbytes(r->d_mask));

        if (ggml_backend_graph_compute(r->backend, r->d_graph) != GGML_STATUS_SUCCESS) {
            if (err) *err = "MM3Rvq: depth graph compute failed";
            return false;
        }
        ggml_backend_tensor_get(r->d_out, logits.data(), 0, (size_t) (AV * N) * sizeof(float));

        for (int64_t i = 0; i < N; i++) {
            const float * row = logits.data() + (size_t) (i * AV);
            int64_t       best = 0;
            float         bv   = row[0];
            for (int64_t j = 1; j < AV; j++) {
                if (row[j] > bv) { bv = row[j]; best = j; }
            }
            ac[(size_t) (i * NAC + step)] = (int32_t) best;
            // Feed back as the prior for book step+1 (offset by its own block).
            if (step < (int) NAC - 1) {
                prior[(size_t) (step * N + i)] = (int32_t) (best + step * AV);
            }
        }
    }
    return true;
}
