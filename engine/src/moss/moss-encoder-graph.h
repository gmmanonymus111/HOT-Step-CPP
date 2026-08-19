// moss-encoder-graph.h: GGML graph for the MOSS-Music audio tower.
//
//   log-mel [n_mels, T]
//     -> conv stem x3 (k3 s2 p1) + GELU        -> [480, 16, T/8]
//     -> flatten + stem_proj                   -> [1280, L]   (L = T/8)
//     -> + sinusoidal positions (RECOMPUTED)
//     -> 32 pre-norm Whisper encoder layers    (taps at 8/16/24)
//     -> LayerNorm                             -> encoder_out
//     -> SwiGLU adapter                        -> [4096, L]
//     -> 3 SwiGLU deepstack mergers from the taps
//
// Shapes in comments are GGML order (ne0 fastest), so [1280, L] is what numpy
// would call [L, 1280].
//
// The op-for-op source of truth is engine/tools/moss-numpy-ref.py, which scores
// corr = 1.000000 against the fp32 reference dumps. Check this graph with:
//   GGML_BACKEND=CPU moss-ggml-test --fixtures <dir> --models <dir> --component encoder
//
// FIVE THINGS THAT ARE EASY TO GET SUBTLY WRONG
// ----------------------------------------------
// 1. GELU must be ggml_gelu_erf, NOT ggml_gelu. Upstream uses ACT2FN["gelu"],
//    the exact erf form; ggml_gelu is the tanh approximation. Both "work".
// 2. k_proj has NO bias (Whisper convention). Do not add one.
// 3. Positions are RECOMPUTED, never loaded — inv_timescales is persistent=False
//    upstream and absent from the GGUF on purpose.
// 4. The encoder is bidirectional and, at batch 1, unpadded, so the attention
//    mask is all zeros. It is omitted entirely rather than materialised.
// 5. The conv stem uses an explicit F32 im2col. ggml_conv_2d forces its im2col
//    to F16 — the same reason mm3-cond-graph.h and mm3-vocoder-graph.h hand-roll
//    theirs.

#pragma once

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "moss/moss-model.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace moss {

struct EncoderOutput {
    int n_tokens = 0;                                  // L
    std::vector<float> encoder_out;                    // [L * d_model]
    std::vector<std::vector<float>> deepstack_taps;    // K x [L * d_model]
    std::vector<float> adapter_out;                    // [L * lm_embd]
    std::vector<std::vector<float>> merger_out;        // K x [L * lm_embd]
};

namespace enc_detail {

// Conv2d (+bias), square stride/pad, dilation 1.
//   w [KW,KH,IC,OC], x [W,H,IC,N] -> [OW,OH,OC,N]
static ggml_tensor * conv2d(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                            ggml_tensor * x, int stride, int pad) {
    // Explicit F32 im2col: ggml_conv_2d would force F16 here (see note 5).
    ggml_tensor * col = ggml_im2col(ctx, w, x, stride, stride, pad, pad, 1, 1,
                                    /*is_2D*/ true, GGML_TYPE_F32);  // [IC*KH*KW, OW, OH, N]
    ggml_tensor * y = ggml_mul_mat(
        ctx,
        ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1] * w->ne[2], w->ne[3]),
        ggml_reshape_2d(ctx, col, col->ne[0], col->ne[1] * col->ne[2] * col->ne[3]));
    // y: [OC, OW*OH*N] -> [OC, OW, OH, N] -> [OW, OH, OC, N]
    y = ggml_reshape_4d(ctx, y, w->ne[3], col->ne[1], col->ne[2], col->ne[3]);
    y = ggml_cont(ctx, ggml_permute(ctx, y, 2, 0, 1, 3));
    if (b) {
        y = ggml_add(ctx, y, ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1));
    }
    return y;
}

// LayerNorm over ne0 (mean + variance), then scale and shift. NOT RMSNorm --
// the LM half uses RMSNorm, the audio tower does not.
static ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w,
                                ggml_tensor * b, float eps) {
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, w);
    return ggml_add(ctx, x, b);
}

// down(silu(gate(h)) * up(h)); no biases anywhere in these.
static ggml_tensor * swiglu(ggml_context * ctx, const SwiGLU & s, ggml_tensor * h) {
    ggml_tensor * g = ggml_mul_mat(ctx, s.gate, h);
    ggml_tensor * u = ggml_mul_mat(ctx, s.up, h);
    return ggml_mul_mat(ctx, s.down, ggml_mul(ctx, ggml_silu(ctx, g), u));
}

}  // namespace enc_detail

// Runs the whole tower for one utterance. `mel` is row-major [n_mels, T] exactly
// as moss::log_mel returns it. Returns false on any allocation failure.
inline bool moss_encode_audio(const AudioTower & m, const float * mel, int n_mels, int T,
                              ggml_backend_t backend, EncoderOutput * out) {
    using namespace enc_detail;
    const AudioHParams & hp = m.hp;
    const int64_t d_model = hp.d_model;
    const int64_t n_head  = hp.n_head;
    const int64_t hd      = d_model / n_head;
    const int64_t n_ds    = (int64_t) hp.deepstack_layers.size();

    if (n_mels != (int) hp.n_mels) {
        fprintf(stderr, "[MOSS] mel has %d bins, model wants %u\n", n_mels, hp.n_mels);
        return false;
    }

    // Three stride-2 convs: ceil(T/2) three times.
    auto half = [](int64_t v) { return (v + 1) / 2; };
    const int64_t L = half(half(half((int64_t) T)));

    // Node budget. Every reshape/permute/cont/view counts, so a pre-norm block is
    // nearer 45 nodes than the ~20 arithmetic ops suggest. Overshoot deliberately:
    // unused slots cost only a pointer each, while undershooting trips
    // GGML_ASSERT(cgraph->n_nodes < cgraph->size) at build time.
    const size_t n_nodes = (size_t) (256 + 96 * hp.n_layer + 32 * (n_ds + 1));
    const size_t ctx_size = ggml_tensor_overhead() * (n_nodes + 64) + ggml_graph_overhead_custom(n_nodes, false);
    ggml_init_params ip = { ctx_size, nullptr, /*no_alloc*/ true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        return false;
    }
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, n_nodes, false);

    // ---- inputs ----------------------------------------------------------
    // mel row-major [n_mels, T] is ne=[T, n_mels] in ggml order; the conv wants
    // [W=T, H=n_mels, C=1, N=1].
    ggml_tensor * in_mel = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, T, n_mels, 1, 1);
    ggml_set_name(in_mel, "mel");
    ggml_set_input(in_mel);

    ggml_tensor * in_pos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, L);
    ggml_set_name(in_pos, "pos");
    ggml_set_input(in_pos);

    // ---- conv stem -------------------------------------------------------
    ggml_tensor * x = in_mel;
    for (int i = 0; i < 3; ++i) {
        x = conv2d(ctx, m.conv_w[i], m.conv_b[i], x, 2, 1);
        x = ggml_gelu_erf(ctx, x);
    }
    // x: [L, 16, 480, 1]. torch does permute(0,3,1,2).flatten(2) on [B,C,F,T],
    // i.e. out[l][c*16 + f] = x[c][f][l] -- F fastest within the pair, so the
    // destination axis order is (F, C, L).
    x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3));   // [16, 480, L, 1]
    ggml_tensor * h = ggml_reshape_2d(ctx, x, x->ne[0] * x->ne[1], L);  // [7680, L]

    h = ggml_add(ctx, ggml_mul_mat(ctx, m.stem_w, h), m.stem_b);        // [1280, L]
    h = ggml_add(ctx, h, in_pos);

    // ---- 32 pre-norm encoder layers --------------------------------------
    std::vector<ggml_tensor *> taps(n_ds, nullptr);
    for (uint32_t il = 0; il < hp.n_layer; ++il) {
        const AudioBlock & b = m.blocks[il];

        ggml_tensor * res = h;
        ggml_tensor * cur = layer_norm(ctx, h, b.attn_norm_w, b.attn_norm_b, hp.layer_norm_eps);

        ggml_tensor * q = ggml_add(ctx, ggml_mul_mat(ctx, b.q_w, cur), b.q_b);
        q = ggml_scale(ctx, q, 1.0f / sqrtf((float) hd));
        ggml_tensor * k = ggml_mul_mat(ctx, b.k_w, cur);              // no bias, by design
        ggml_tensor * v = ggml_add(ctx, ggml_mul_mat(ctx, b.v_w, cur), b.v_b);

        q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, q, hd, n_head, L), 0, 2, 1, 3));
        k = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, k, hd, n_head, L), 0, 2, 1, 3));
        // v transposed to [L, hd, n_head] so the second mul_mat contracts over L.
        v = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, v, hd, n_head, L), 1, 2, 0, 3));

        ggml_tensor * kq = ggml_mul_mat(ctx, k, q);                   // [L, L, n_head]
        kq = ggml_soft_max(ctx, kq);                                  // no mask: see note 4
        ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);                 // [hd, L, n_head]
        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));     // [hd, n_head, L]
        cur = ggml_reshape_2d(ctx, kqv, d_model, L);
        cur = ggml_add(ctx, ggml_mul_mat(ctx, b.o_w, cur), b.o_b);
        h = ggml_add(ctx, res, cur);

        res = h;
        cur = layer_norm(ctx, h, b.ffn_norm_w, b.ffn_norm_b, hp.layer_norm_eps);
        cur = ggml_gelu_erf(ctx, ggml_add(ctx, ggml_mul_mat(ctx, b.up_w, cur), b.up_b));
        cur = ggml_add(ctx, ggml_mul_mat(ctx, b.down_w, cur), b.down_b);
        h = ggml_add(ctx, res, cur);

        // Tap AFTER the FFN residual, before the next block -- matching the
        // reference's output_hidden_states capture.
        for (int64_t kdx = 0; kdx < n_ds; ++kdx) {
            if ((int32_t) il == hp.deepstack_layers[(size_t) kdx]) {
                taps[(size_t) kdx] = h;
                // ggml_set_output is REQUIRED, not cosmetic: without it the
                // allocator reuses this buffer as soon as the tap's last consumer
                // has run, and the readback returns whatever landed there next.
                // The failure is deceptive -- the SwiGLU fed BY the tap still
                // scores 1.0000000 because it consumed the correct values before
                // the overwrite; only the tap itself reads back as noise.
                ggml_set_output(h);
                ggml_build_forward_expand(gf, h);
            }
        }
    }

    ggml_tensor * enc = layer_norm(ctx, h, m.norm_w, m.norm_b, hp.layer_norm_eps);
    ggml_set_name(enc, "encoder_out");
    ggml_set_output(enc);   // see the note on the taps above
    ggml_build_forward_expand(gf, enc);

    ggml_tensor * ad = swiglu(ctx, m.adapter, enc);
    ggml_set_name(ad, "adapter_out");
    ggml_set_output(ad);
    ggml_build_forward_expand(gf, ad);

    std::vector<ggml_tensor *> mergers(n_ds, nullptr);
    for (int64_t kdx = 0; kdx < n_ds; ++kdx) {
        if (!taps[(size_t) kdx]) {
            fprintf(stderr, "[MOSS] deepstack tap %lld never captured\n", (long long) kdx);
            ggml_free(ctx);
            return false;
        }
        mergers[(size_t) kdx] = swiglu(ctx, m.deepstack[(size_t) kdx], taps[(size_t) kdx]);
        ggml_set_output(mergers[(size_t) kdx]);
        ggml_build_forward_expand(gf, mergers[(size_t) kdx]);
    }

    // ---- allocate, upload, run -------------------------------------------
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        fprintf(stderr, "[MOSS] graph allocation failed\n");
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(in_mel, mel, 0, (size_t) n_mels * T * sizeof(float));

    // Sinusoids, recomputed (note 3):
    //   inv = exp(-log(max_ts)/(d/2 - 1) * arange(d/2))
    //   pos = concat(sin(t*inv), cos(t*inv))
    {
        const int64_t half_d = d_model / 2;
        const double log_inc = std::log((double) hp.max_timescale) / (double) (half_d - 1);
        std::vector<float> pos((size_t) d_model * L);
        for (int64_t t = 0; t < L; ++t) {
            for (int64_t i = 0; i < half_d; ++i) {
                const double s = (double) t * std::exp(-log_inc * (double) i);
                pos[(size_t) t * d_model + i]            = (float) std::sin(s);
                pos[(size_t) t * d_model + half_d + i]   = (float) std::cos(s);
            }
        }
        ggml_backend_tensor_set(in_pos, pos.data(), 0, pos.size() * sizeof(float));
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[MOSS] graph compute failed\n");
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    auto fetch = [](ggml_tensor * t, std::vector<float> & dst) {
        dst.resize((size_t) ggml_nelements(t));
        ggml_backend_tensor_get(t, dst.data(), 0, dst.size() * sizeof(float));
    };

    out->n_tokens = (int) L;
    fetch(enc, out->encoder_out);
    fetch(ad, out->adapter_out);
    out->deepstack_taps.assign((size_t) n_ds, {});
    out->merger_out.assign((size_t) n_ds, {});
    for (int64_t kdx = 0; kdx < n_ds; ++kdx) {
        fetch(taps[(size_t) kdx], out->deepstack_taps[(size_t) kdx]);
        fetch(mergers[(size_t) kdx], out->merger_out[(size_t) kdx]);
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return true;
}

}  // namespace moss
