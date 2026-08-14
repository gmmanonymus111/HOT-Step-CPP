#pragma once
// minimax/mm3-cond-graph.h — MiniMax-Music3 condition encoder (~25M).
//
// HOT-Step file (does not exist upstream). Included only by minimax/mm3-server.h,
// which is itself the single hook include into tools/hot-step-server.cpp.
//
// SCOPE (increment 4b): the bridge between the AR stage and the flow stage.
// Takes one window's fused per-frame hidden block and produces the DiT's
// `condition` tensor on the latent timeline. The AR stage that fills the input
// and the window orchestration that slices it are later increments. Validated
// standalone through POST /mm3/cond-encode against the diffusers reference dump
// in mm3-weights/fixtures/.
//
// ── Architecture (contract: docs/plans/mm3-gguf-layout.md §3.3, and the diffusers
//    reference MiniMaxMusic3ConditionEncoder at commit dafe3733) ───────────────
//
//   hidden [F, 8*4096]                       8 = LM last_hidden + 7 depth hiddens
//     -> view [8, 4096, F]                   the 32768 axis is LAYER-MAJOR
//     -> einsum with softmax(layer_logits)   [4096, F]
//     -> * layer_scale                       one learned global gain
//     -> Conv1d(4096 -> 2048, k=3, pad=1)    WITH bias
//     -> nearest resample F -> L             L = int(F * 3.4453125)
//     -> [L, 2048]
//
// ── Conventions that are easy to get wrong, and where each is pinned ──────────
//
// 1. THE 32768 AXIS IS LAYER-MAJOR, NOT FEATURE-MAJOR. The reference does
//    `hidden.transpose(1,2).reshape(B, 8, 4096, F)`, so layer l occupies
//    `[l*4096 : (l+1)*4096]` within each frame. The fixture manifest states this
//    outright for `cond_in_w0.bin`. Interleaving the two would be silent.
//
// 2. LAYER 0 IS THE LM's last_hidden, LAYERS 1..7 ARE THE DEPTH DECODER's — in
//    codebook order c1..c7, conditional row only.
//
// 3. THE RESAMPLE IS `F.interpolate(mode="nearest")`, WHOSE INDEX MAP IS
//    `src = min(floor(dst * F / L), F - 1)` — a plain truncation of
//    `dst * scale` with `scale = F / L`. It is NOT "nearest-exact" (which would
//    use `(dst + 0.5) * scale - 0.5`) and it is NOT rounding. At F=200, L=689
//    the two differ on 199 of 689 positions, so this is a real fork, not a
//    rounding quibble. Reproduced with integer arithmetic (`dst * F / L`), which
//    is exact where the float form is merely nearly-exact. Verified elementwise
//    identical to torch on the fixture window.
//
// 4. THE LENGTH FORMULA IS EVALUATED LEFT-TO-RIGHT IN FLOATING POINT AND THEN
//    TRUNCATED: `int(F * out_sr / in_sr * in_hop / out_hop)`, i.e.
//    `int(F * 3.4453125)`. 200 -> 689.0625 -> 689. Computed in double here, in
//    the reference's own order.
//
// ── Design decisions, and why ─────────────────────────────────────────────────
//
// A. THE SOFTMAX AND `layer_scale` ARE FOLDED INTO ONE 8-ELEMENT HOST VECTOR.
//    Both are constants over the whole generation and the mix is linear, so
//    `mix[l] = softmax(layer_logits)[l] * layer_scale` is exactly the reference's
//    two steps. Computed in double from the F32 tensors (the layout doc pins both
//    F32 for this reason — the softmax weights the ENTIRE conditioning signal)
//    and staged as a derived weight.
//
// B. THE LAYER MIX IS A MATMUL OVER A PERMUTED VIEW. ggml has no reduction over
//    ne1, so the input is permuted to [8, 4096, F] and contracted with the mix
//    vector by `ggml_mul_mat`. The permute costs one 26 MB transposed copy per
//    window (~74 windows for a 5-minute song); the alternative — eight strided
//    slabs, each needing its own `ggml_cont` before `ggml_scale` will accept it —
//    copies the same bytes and adds 23 nodes.
//
// C. THE RESAMPLE IS A `ggml_get_rows` GATHER. Nearest-neighbour resampling IS a
//    gather; ggml_interpolate would force the float index math of note 3 through
//    a different rounding path for no gain.
//
// D. OUTPUT IS [2048, L] — feature-fastest, i.e. the memory order of a torch
//    [1, L, 2048] tensor. That is byte-identical to the reference dump AND is
//    exactly what mm3_dit_run() wants for its `cond` argument, so the two
//    increments bolt together with no repacking. (Contrast the DiT's LATENTS,
//    which are channel-major — the asymmetry is the reference's, see
//    mm3-dit-graph.h design note A.)
//
// ── Parity, measured 2026-08-13 (window 0, F=200 -> L=689, RTX 5090, f16 GGUF) ─
//
// `cond_in_w0.bin` in, compared against `cond_out_w0.bin` and against the same
// reference module re-run in float32 on the same input:
//
//                        corr        rel RMSE   max abs
//   ours vs bf16 dump    0.9999962   2.8e-3     3.0e-2
//   ours vs fp32 ref     0.9999982   2.6e-3     2.0e-2
//   dump vs fp32 ref     0.9999967   3.3e-3     3.7e-2
//
// We are closer to float32 truth than the capture we are validated against. The
// residual is precision, not structure: `cond.proj.weight` is stored F16 (k=3
// makes ne0 = 3, so Q8_0's 32-element blocks cannot apply and the converter's
// HALF policy pins it — layout doc §7), and one 4096-wide dot product per output
// is where a 2.6e-3 relative error comes from. The resample index map was
// separately checked ELEMENTWISE against torch's `F.interpolate` and is exact, so
// note 3 is verified rather than argued.
//
// Speed: 2.7 ms median for F=200 (10 runs, warm), i.e. ~34 ms for a full 5-minute
// generation's ~74 windows. Not a budget item.

#include "mm3-model.h"

#include "backend.h"
#include "ggml.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// ~20 nodes measured; the cap is deliberately loose.
#define MM3_COND_MAX_NODES 256
// Frame-count ceiling for one window. The reference window is 200 frames; the
// compute buffer grows linearly (8 * 4096 * F * 4 bytes for the input alone).
#define MM3_COND_MAX_FRAMES 4096

struct MM3CondGraph {
    ggml_backend_t       backend     = nullptr;
    ggml_backend_t       cpu_backend = nullptr;
    bool                 backend_ref = false;
    ggml_backend_sched_t sched       = nullptr;
    WeightCtx            prep        = {};  // the folded mix vector
    const void *         weights_token = nullptr;

    ggml_tensor * mix = nullptr;  // [8, 1] F32 — softmax(layer_logits) * layer_scale

    // cached graph (rebuilt when F changes)
    ggml_context * gctx   = nullptr;
    uint8_t *      gbuf   = nullptr;
    ggml_cgraph *  graph  = nullptr;
    ggml_tensor *  input  = nullptr;  // [4096, 8, F] F32 — raw torch [1, F, 8*4096] order
    ggml_tensor *  in_idx = nullptr;  // [L]          I32 — nearest-neighbour source frames
    ggml_tensor *  output = nullptr;  // [2048, L]    F32 — torch [1, L, 2048] order
    int64_t        graph_F = 0;
    int64_t        graph_L = 0;
    size_t         compute_bytes = 0;
};

// L = int(F * output_sampling_rate / input_sampling_rate * input_hop / output_hop),
// evaluated in the reference's own left-to-right order (note 4).
static int64_t mm3_cond_latent_length(const MM3CondConfig & c, int64_t frames) {
    const double r = (double) frames * (double) c.output_sampling_rate / (double) c.input_sampling_rate *
                     (double) c.input_hop_length / (double) c.output_hop_length;
    const int64_t l = (int64_t) r;  // truncation, matching Python's int()
    return l < 1 ? 1 : l;
}

// ── Prep / free ─────────────────────────────────────────────────────────────

static void mm3_cond_free_graph(MM3CondGraph * g) {
    if (g->gctx) {
        if (g->sched) {
            ggml_backend_sched_reset(g->sched);
        }
        ggml_free(g->gctx);
        free(g->gbuf);
    }
    g->gctx    = nullptr;
    g->gbuf    = nullptr;
    g->graph   = nullptr;
    g->input   = nullptr;
    g->in_idx  = nullptr;
    g->output  = nullptr;
    g->graph_F = 0;
    g->graph_L = 0;
}

static void mm3_cond_free(MM3CondGraph * g) {
    mm3_cond_free_graph(g);
    if (g->sched) {
        ggml_backend_sched_free(g->sched);
        g->sched = nullptr;
    }
    wctx_free(&g->prep);
    g->mix           = nullptr;
    g->weights_token = nullptr;
    if (g->backend_ref) {
        backend_release(g->backend, g->cpu_backend);
        g->backend     = nullptr;
        g->cpu_backend = nullptr;
        g->backend_ref = false;
    }
}

static bool mm3_cond_readback_f32(const ggml_tensor * t, std::vector<float> * out, std::string * err,
                                  const char * what) {
    if (!t) {
        if (err) {
            *err = std::string("condition-encoder tensor missing: ") + what;
        }
        return false;
    }
    if (t->type != GGML_TYPE_F32) {
        if (err) {
            *err = std::string("condition-encoder tensor '") + what + "' is not F32 (type " +
                   std::to_string((int) t->type) + "); the layout contract pins it to F32";
        }
        return false;
    }
    out->resize((size_t) ggml_nelements(t));
    ggml_backend_tensor_get((ggml_tensor *) t, out->data(), 0, ggml_nbytes(t));
    return true;
}

// Fold softmax(layer_logits) * layer_scale into one staged 8-vector (note A).
static bool mm3_cond_prepare(const MM3Model & m, MM3CondGraph * g, std::string * err) {
    // Staged residency: a stage-2 graph needs only the cond/dit/voc buffer.
    // The LM is deliberately gone by this point.
    if (!m.rest_resident) {
        if (err) {
            *err = "MiniMax-Music3 is not warm (POST /mm3/warm first)";
        }
        return false;
    }
    const void * token = (const void *) m.wctx_synth.buffer;
    if (g->weights_token == token && g->sched) {
        return true;
    }
    mm3_cond_free(g);

    const MM3CondConfig &  c = m.synth_cfg.cond;
    const MM3CondWeights & w = m.synth.cond;
    if (c.num_layers == 0 || c.hidden_dim == 0 || c.out_dim == 0) {
        if (err) {
            *err = "condition-encoder config is empty — mm3.cond.* KVs missing from the synth GGUF";
        }
        return false;
    }
    if (c.layer_mix != "softmax") {
        if (err) {
            *err = "mm3.cond.layer_mix is '" + c.layer_mix + "', this port only implements 'softmax'";
        }
        return false;
    }
    if (c.interpolation != "nearest") {
        if (err) {
            *err = "mm3.cond.interpolation is '" + c.interpolation + "', this port only implements 'nearest'";
        }
        return false;
    }
    if (c.kernel_size != 3 || c.padding != 1) {
        if (err) {
            *err = "mm3.cond kernel/padding is " + std::to_string(c.kernel_size) + "/" + std::to_string(c.padding) +
                   ", this port only implements 3/1";
        }
        return false;
    }

    BackendPair bp = backend_init("MM3-Cond");
    g->backend     = bp.backend;
    g->cpu_backend = bp.cpu_backend;
    g->backend_ref = true;

    std::vector<float> logits, scale;
    std::string        e;
    if (!mm3_cond_readback_f32(w.layer_logits, &logits, &e, "cond.layer_logits") ||
        !mm3_cond_readback_f32(w.layer_scale, &scale, &e, "cond.layer_scale")) {
        if (err) {
            *err = e;
        }
        mm3_cond_free(g);
        return false;
    }
    if (logits.size() != (size_t) c.num_layers || scale.size() != 1) {
        if (err) {
            *err = "cond.layer_logits/layer_scale have unexpected element counts";
        }
        mm3_cond_free(g);
        return false;
    }

    double mx = -1e30;
    for (float v : logits) {
        mx = v > mx ? (double) v : mx;
    }
    double sum = 0.0;
    for (float v : logits) {
        sum += std::exp((double) v - mx);
    }
    auto mix = std::make_unique<float[]>(logits.size());
    for (size_t i = 0; i < logits.size(); i++) {
        mix[i] = (float) (std::exp((double) logits[i] - mx) / sum * (double) scale[0]);
    }

    wctx_init(&g->prep, 1);
    g->mix = ggml_new_tensor_2d(g->prep.ctx, GGML_TYPE_F32, (int64_t) logits.size(), 1);
    ggml_set_name(g->mix, "cond.layer_mix");
    g->prep.pending.push_back({ g->mix, mix.get(), logits.size() * sizeof(float), 0 });
    g->prep.staging.push_back(std::move(mix));
    if (!wctx_alloc(&g->prep, g->backend)) {
        if (err) {
            *err = "backend buffer allocation failed for the condition-encoder mix vector";
        }
        mm3_cond_free(g);
        return false;
    }

    g->sched         = backend_sched_new(bp, MM3_COND_MAX_NODES * 2);
    g->weights_token = token;

    fprintf(stderr, "[MM3-Cond] Prepared: %u layers -> %u -> %u, conv k=%u p=%u, %s resample %u/%u Hz\n",
            c.num_layers, c.hidden_dim, c.out_dim, c.kernel_size, c.padding, c.interpolation.c_str(),
            c.input_sampling_rate, c.output_sampling_rate);
    return true;
}

// ── Graph ───────────────────────────────────────────────────────────────────

// Conv1d (+bias), stride 1, dilation 1. w [K, IC, OC], x [T, IC] -> [OC, T_out].
//
// Explicit F32 im2col rather than ggml_conv_1d's forced F16 — same reasoning as
// the vocoder (mm3-vocoder-graph.h note 5). Two deliberate differences from the
// vocoder's helper: the WEIGHT is mul_mat's src0 (it is F16 here, and src0 is the
// operand ggml allows to be non-F32), and the result is therefore left
// CHANNEL-fastest, which is the layout the caller wants anyway.
static ggml_tensor * mm3_cond_conv1d(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x, int pad) {
    ggml_tensor * col = ggml_im2col(ctx, w, x, /*s0*/ 1, /*s1*/ 0, pad, 0, /*d0*/ 1, 0, /*is_2D*/ false,
                                    GGML_TYPE_F32);  // [IC*K, OL]
    ggml_tensor * y = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]),
                                   ggml_reshape_2d(ctx, col, col->ne[0], col->ne[1] * col->ne[2]));  // [OC, OL]
    if (b) {
        y = ggml_add(ctx, y, b);
    }
    return y;
}

static ggml_tensor * mm3_cond_build(ggml_context * ctx, const MM3Model & m, const MM3CondGraph & g) {
    const MM3CondConfig &  c = m.synth_cfg.cond;
    const MM3CondWeights & w = m.synth.cond;
    const int64_t          F = g.input->ne[2];

    // [4096, 8, F] -> [8, 4096, F], then contract the layer axis with the folded
    // softmax*scale vector (design note B).
    ggml_tensor * xp = ggml_cont(ctx, ggml_permute(ctx, g.input, 1, 0, 2, 3));  // [8, 4096, F]
    ggml_tensor * y  = ggml_mul_mat(ctx, g.mix, xp);                            // [1, 4096, F]
    y                = ggml_reshape_2d(ctx, y, (int64_t) c.hidden_dim, F);      // [4096, F]

    // im2col wants time-fastest; the mix leaves us channel-fastest. The conv puts
    // it back to channel-fastest, which is what the gather (rows = TIME) and the
    // DiT's condition layout both want (design note D).
    y = ggml_cont(ctx, ggml_transpose(ctx, y));                       // [F, 4096]
    y = mm3_cond_conv1d(ctx, w.proj_w, w.proj_b, y, (int) c.padding);  // [2048, F]

    return ggml_get_rows(ctx, y, g.in_idx);  // [2048, L]
}

static bool mm3_cond_ensure_graph(const MM3Model & m, MM3CondGraph * g, int64_t F, std::string * err) {
    if (g->graph && g->graph_F == F) {
        return true;
    }
    mm3_cond_free_graph(g);

    const MM3CondConfig & c = m.synth_cfg.cond;
    const int64_t         L = mm3_cond_latent_length(c, F);

    const size_t ctx_bytes =
        ggml_tensor_overhead() * (MM3_COND_MAX_NODES + 64) + ggml_graph_overhead_custom(MM3_COND_MAX_NODES, false);
    g->gbuf = (uint8_t *) malloc(ctx_bytes);
    if (!g->gbuf) {
        if (err) {
            *err = "out of host memory allocating the condition-encoder graph context";
        }
        return false;
    }
    ggml_init_params ip  = { ctx_bytes, g->gbuf, /*no_alloc*/ true };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        free(g->gbuf);
        g->gbuf = nullptr;
        if (err) {
            *err = "ggml_init failed for the condition-encoder graph context";
        }
        return false;
    }

    g->input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, (int64_t) c.hidden_dim, (int64_t) c.num_layers, F);
    ggml_set_name(g->input, "mm3_cond_in");
    ggml_set_input(g->input);

    g->in_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, L);
    ggml_set_name(g->in_idx, "mm3_cond_resample_idx");
    ggml_set_input(g->in_idx);

    g->output = mm3_cond_build(ctx, m, *g);
    ggml_set_name(g->output, "mm3_cond_out");
    ggml_set_output(g->output);

    g->graph = ggml_new_graph_custom(ctx, MM3_COND_MAX_NODES, false);
    ggml_build_forward_expand(g->graph, g->output);

    ggml_backend_sched_reset(g->sched);
    if (!ggml_backend_sched_alloc_graph(g->sched, g->graph)) {
        ggml_free(ctx);
        free(g->gbuf);
        g->gbuf  = nullptr;
        g->graph = nullptr;
        if (err) {
            *err = "condition-encoder graph allocation failed (out of VRAM?) for F=" + std::to_string(F);
        }
        return false;
    }

    g->gctx          = ctx;
    g->graph_F       = F;
    g->graph_L       = L;
    g->compute_bytes = ggml_backend_sched_get_buffer_size(g->sched, g->backend);

    fprintf(stderr, "[MM3-Cond] Graph: F=%lld -> L=%lld, %d nodes, %d splits, compute buffer %.0f MB\n", (long long) F,
            (long long) L, ggml_graph_n_nodes(g->graph), ggml_backend_sched_get_n_splits(g->sched),
            (double) g->compute_bytes / (1024.0 * 1024.0));
    return true;
}

// ── Public API ──────────────────────────────────────────────────────────────

static MM3CondGraph g_mm3_cond;

// Encode one window of AR hidden states into the flow stage's conditioning.
//
//   hiddens   F * 8 * 4096 floats in torch [1, F, 8*4096] memory order: frame
//             slowest, then LAYER, then feature (note 1). Layer 0 is the LM's
//             last_hidden_state, layers 1..7 the depth decoder's, conditional
//             row only (note 2).
//   F         AR frames in this window (the reference uses 200).
//   out       resized to 2048 * L and filled in torch [1, L, 2048] order —
//             directly usable as mm3_dit_run()'s `cond` (design note D).
//   out_L     the latent length, = int(F * 3.4453125).
//
// Not thread-safe: the caller serialises (mm3-server.h holds g_mm3_mutex).
static bool mm3_cond_encode(const MM3Model & m, const float * hiddens, int64_t F, std::vector<float> & out,
                            int64_t * out_L, std::string * err = nullptr) {
    if (F <= 0 || F > MM3_COND_MAX_FRAMES) {
        if (err) {
            *err = "frames must be in 1.." + std::to_string(MM3_COND_MAX_FRAMES);
        }
        return false;
    }
    if (!mm3_cond_prepare(m, &g_mm3_cond, err)) {
        return false;
    }
    MM3CondGraph * g = &g_mm3_cond;
    if (!mm3_cond_ensure_graph(m, g, F, err)) {
        return false;
    }

    const int64_t L = g->graph_L;

    // Nearest-neighbour source frames, integer arithmetic (note 3).
    std::vector<int32_t> idx((size_t) L);
    for (int64_t i = 0; i < L; i++) {
        int64_t s = i * F / L;
        idx[(size_t) i] = (int32_t) (s < F ? s : F - 1);
    }
    ggml_backend_tensor_set(g->in_idx, idx.data(), 0, idx.size() * sizeof(int32_t));
    ggml_backend_tensor_set(g->input, hiddens, 0, ggml_nbytes(g->input));

    if (ggml_backend_sched_graph_compute(g->sched, g->graph) != GGML_STATUS_SUCCESS) {
        if (err) {
            *err = "condition-encoder graph compute failed";
        }
        return false;
    }

    out.assign((size_t) ((int64_t) m.synth_cfg.cond.out_dim * L), 0.0f);
    ggml_backend_tensor_get(g->output, out.data(), 0, out.size() * sizeof(float));
    if (out_L) {
        *out_L = L;
    }
    return true;
}
