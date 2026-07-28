#pragma once
// lm-fsq.h — reference-conformant FSQ quantization for the LM trainer.
//
// WHY THIS FILE EXISTS
// --------------------
// engine/src/fsq-tok.h::fsq_encode_index() implements the OLD
// vector_quantize_pytorch symmetry-preserving bound:
//
//     level = floor((L-1) * (tanh(z) + 1) / 2 + 0.5)
//
// The checkpoints we train against were tokenized by
// vector_quantize_pytorch >= 1.27.21 (Side-Step ships 1.28.0), where
// `ResidualFSQ.__init__` hard-wires `preserve_symmetry=True` AND
// `bound_hard_clamp=True` (residual_fsq.py:63,94-95,116-123). That turns the
// forward path into (residual_fsq.py:191-195 + finite_scalar_quantization.py
// symmetry_preserving_bound/_scale_and_shift):
//
//     c     = 1 + 1/(L-1)                       # soft_clamp_input_value
//     y     = tanh(z / c) * c                   # ResidualFSQ soft clamp
//     w     = clamp(y, -1, 1)                   # hard_clamp replaces tanh
//     level = floor((L-1) * (w + 1) / 2 + 0.5)
//     index = sum_d level_d * basis_d,  basis = exclusive cumprod(levels)
//
// The two agree only near z = 0; on real tokenizer projections they differ on
// ~60% of frames (always by +-1 level). Verified against a PyTorch dump of the
// tokenizer's own pre-quantization projections: this formula reproduces the
// reference indices 13046/13046 = 100.00%, the engine's formula 40.19%.
//
// The engine's inference path (fsq-tok.h / fsq-detok.h) is deliberately NOT
// touched here: changing it would change every existing generation. This header
// is used ONLY by the train-lm extract stage, which must produce codes that
// match what the LM was pretrained on.

#include "fsq-tok.h"  // TokGGML, FSQ_NDIMS, FSQ_LEVELS, qwen3_* helpers

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

// ─── reference-conformant scalar quantizer ──────────────────────────────────
//
// raw_vals: the `quantizer.project_in` output for one 5 Hz group (FSQ_NDIMS
// floats, pre-quantization). Returns the flat codebook index.
static int fsq_encode_index_ref(const float * raw_vals) {
    int index = 0;
    int mult  = 1;
    for (int d = 0; d < FSQ_NDIMS; d++) {
        const int   L = FSQ_LEVELS[d];
        const float Lm1 = (float) (L - 1);

        // ResidualFSQ soft clamp: x = tanh(x / c) * c   with c = 1 + 1/(L-1)
        const float c = 1.0f + 1.0f / Lm1;
        float       y = tanhf(raw_vals[d] / c) * c;

        // FSQ(bound_hard_clamp=True): tanh is replaced by a hard clamp
        if (y < -1.0f) {
            y = -1.0f;
        }
        if (y > 1.0f) {
            y = 1.0f;
        }

        // symmetry_preserving_bound + _scale_and_shift collapse to this level
        int code = (int) floorf(Lm1 * (y + 1.0f) / 2.0f + 0.5f);
        if (code < 0) {
            code = 0;
        }
        if (code >= L) {
            code = L - 1;
        }

        index += code * mult;
        mult *= L;
    }
    return index;
}

// ─── tokenizer forward + reference quantization ─────────────────────────────
//
// Same graph as fsq-tok.h::tok_ggml_encode() (which we cannot reuse directly:
// it swallows the pre-quantization floats and returns only indices, and that
// file is shared with the inference path). Only the final float -> level step
// differs. `raw_out`, when non-NULL, receives T_5Hz * FSQ_NDIMS floats — the
// pre-quantization projections, for debugging/verification.
//
// latents:  [T_25Hz, 64] time-major
// codes_out: caller-allocated, at least (T_25Hz + 4) / 5 ints
// Returns T_5Hz, or -1 on error.
static int lm_tok_encode_ref(TokGGML *     m,
                             const float * latents,
                             int           T_25Hz,
                             int *         codes_out,
                             const float * silence_latent,
                             float *       raw_out = NULL) {
    const int P = 5;                   // pool_window_size
    const int S = 6;                   // 1 CLS + 5 patches
    const int H = m->cfg.hidden_size;  // 2048

    const int pad      = (P - (T_25Hz % P)) % P;
    const int T_padded = T_25Hz + pad;
    const int T_5Hz    = T_padded / P;

    std::vector<float> input((size_t) T_padded * 64);
    memcpy(input.data(), latents, (size_t) T_25Hz * 64 * sizeof(float));
    if (pad > 0) {
        memcpy(input.data() + (size_t) T_25Hz * 64, silence_latent, (size_t) pad * 64 * sizeof(float));
    }

    size_t    ctx_size = ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(4096, false);
    uint8_t * ctx_buf  = (uint8_t *) malloc(ctx_size);
    if (!ctx_buf) {
        fprintf(stderr, "[FSQ-Tok] OOM allocating graph context (%zu bytes)\n", ctx_size);
        return -1;
    }
    struct ggml_init_params gparams = { ctx_size, ctx_buf, true };
    struct ggml_context *   ctx     = ggml_init(gparams);

    struct ggml_tensor * tok_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, P);
    ggml_set_name(tok_in, "tok_in");
    ggml_set_input(tok_in);

    struct ggml_tensor * projected = ggml_mul_mat(ctx, m->proj_w, tok_in);
    projected                      = ggml_add(ctx, projected, qwen3_f32(ctx, ggml_reshape_2d(ctx, m->proj_b, H, 1)));

    struct ggml_tensor * embedded = ggml_mul_mat(ctx, m->embed_w, projected);
    embedded                      = ggml_add(ctx, embedded, qwen3_f32(ctx, ggml_reshape_2d(ctx, m->embed_b, H, 1)));

    struct ggml_tensor * cls    = ggml_reshape_2d(ctx, qwen3_f32(ctx, m->special_tok), H, 1);
    struct ggml_tensor * hidden = ggml_concat(ctx, cls, embedded, 1);

    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    ggml_set_name(positions, "tok_pos");
    ggml_set_input(positions);

    hidden = qwen3_build_layers(ctx, m->cfg, m->layers, m->norm, hidden, positions, NULL, S, m->use_flash_attn);

    struct ggml_tensor * cls_out = ggml_view_2d(ctx, hidden, H, 1, hidden->nb[1], 0);

    struct ggml_tensor * fsq_vals = ggml_mul_mat(ctx, m->fsq_in_w, cls_out);
    fsq_vals = ggml_add(ctx, fsq_vals, qwen3_f32(ctx, ggml_reshape_2d(ctx, m->fsq_in_b, FSQ_NDIMS, 1)));
    ggml_set_name(fsq_vals, "tok_out");
    ggml_set_output(fsq_vals);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(gf, fsq_vals);

    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) {
        fprintf(stderr, "[Tok] FATAL: graph alloc failed\n");
        ggml_free(ctx);
        free(ctx_buf);
        return -1;
    }

    int32_t pos_data[6] = { 0, 1, 2, 3, 4, 5 };

    struct ggml_tensor * t_in  = ggml_graph_get_tensor(gf, "tok_in");
    struct ggml_tensor * t_out = ggml_graph_get_tensor(gf, "tok_out");
    struct ggml_tensor * t_pos = ggml_graph_get_tensor(gf, "tok_pos");

    float fsq_buf[FSQ_NDIMS];
    for (int g = 0; g < T_5Hz; g++) {
        ggml_backend_tensor_set(t_pos, pos_data, 0, S * sizeof(int32_t));
        ggml_backend_tensor_set(t_in, input.data() + (size_t) g * P * 64, 0, (size_t) P * 64 * sizeof(float));

        ggml_backend_sched_graph_compute(m->sched, gf);

        ggml_backend_tensor_get(t_out, fsq_buf, 0, FSQ_NDIMS * sizeof(float));
        if (raw_out) {
            memcpy(raw_out + (size_t) g * FSQ_NDIMS, fsq_buf, FSQ_NDIMS * sizeof(float));
        }
        codes_out[g] = fsq_encode_index_ref(fsq_buf);
    }

    fprintf(stderr, "[Tok] Tokenized: %d frames -> %d codes (%.1fs @ 5Hz, reference FSQ)\n", T_25Hz, T_5Hz,
            (float) T_5Hz / 5.0f);

    ggml_backend_sched_reset(m->sched);
    ggml_free(ctx);
    free(ctx_buf);
    return T_5Hz;
}
