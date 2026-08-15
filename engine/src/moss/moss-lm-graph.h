// moss-lm-graph.h: MOSS-Music LM (Qwen3-8B) prefill, with audio splice + deepstack.
//
// SCOPE: a single PREFILL pass over T positions, batch 1, no KV cache. That is
// exactly what the audio->LM seam needs to be proven against the reference's
// `logits_last` fixture, and it is the riskiest remaining join in the port
// (deepstack uses two different layer indexings that both start at 0). The
// cached decode loop is a separate, later step -- model it on
// minimax/mm3-lm-graph.h, whose LM is architecturally IDENTICAL to this one.
//
// ── The two audio joins ──────────────────────────────────────────────────────
//
// 1. SPLICE. inputs_embeds = token_embd[ids], with the rows at <|AUDIO|>
//    (id 151654) REPLACED by the audio adapter output. Done here as
//    emb*text_mask + audio_vals rather than a scatter, because token_embd is
//    q8_0 and dequantising host-side to patch rows would be worse in every way.
//
// 2. DEEPSTACK. The three merger outputs are ADDED into the first three LM
//    layers -- at the audio rows only -- after each block's output. Note the two
//    indexings: mergers come from ENCODER layers 8/16/24, and land in LM layers
//    0/1/2. Both start at 0 and neither is the other.
//
// Both join tensors are built host-side (zeros at text rows) and uploaded, which
// keeps the graph free of scatter ops and makes the masking obvious on
// inspection. At 4096 x ~500 f32 that is ~8 MB per tensor -- irrelevant next to
// the 8.7 GB of weights.

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

namespace lm_detail {

static ggml_tensor * rms(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}

}  // namespace lm_detail

// Runs one prefill and returns the logits at the LAST position.
//
//   ids          [T]      token ids, with hp.audio_token_id at every audio slot
//   audio_vals   [H*T]    adapter output at audio rows, 0 elsewhere (row-major, H fastest)
//   merger_vals  K x [H*T] deepstack merger output at audio rows, 0 elsewhere
//   text_mask    [T]      1.0 at text rows, 0.0 at audio rows
inline bool moss_lm_prefill_logits(const LmModel & m, const std::vector<int32_t> & ids,
                                   const std::vector<float> & audio_vals,
                                   const std::vector<std::vector<float>> & merger_vals,
                                   const std::vector<float> & text_mask,
                                   ggml_backend_t backend, std::vector<float> * out_logits) {
    using namespace lm_detail;
    const LmHParams & hp = m.hp;
    const int64_t T   = (int64_t) ids.size();
    const int64_t H   = (int64_t) hp.n_embd;
    const int64_t D   = (int64_t) hp.head_dim;
    const int64_t Nh  = (int64_t) hp.n_head;
    const int64_t Nkv = (int64_t) hp.n_head_kv;
    const int64_t n_inject = (int64_t) merger_vals.size();

    if ((int64_t) audio_vals.size() != H * T || (int64_t) text_mask.size() != T) {
        fprintf(stderr, "[MOSS] prefill: join tensors have the wrong size\n");
        return false;
    }

    // ~60 nodes per block once reshapes/permutes/conts are counted, plus the
    // splice, the injections and the head. Overshoot: see the encoder graph note.
    const size_t n_nodes = (size_t) (512 + 128 * hp.n_layer);
    const size_t ctx_size = ggml_tensor_overhead() * (n_nodes + 64) +
                            ggml_graph_overhead_custom(n_nodes, false);
    ggml_init_params ip = { ctx_size, nullptr, /*no_alloc*/ true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        return false;
    }
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, n_nodes, false);

    ggml_tensor * in_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_input(in_ids);
    ggml_tensor * in_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_input(in_pos);
    ggml_tensor * in_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, T);   // text mask
    ggml_set_input(in_mask);
    ggml_tensor * in_audio = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
    ggml_set_input(in_audio);
    std::vector<ggml_tensor *> in_merge((size_t) n_inject, nullptr);
    for (int64_t k = 0; k < n_inject; ++k) {
        in_merge[(size_t) k] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
        ggml_set_input(in_merge[(size_t) k]);
    }
    // Causal mask, [T_kv, T_q], padded to GGML_KQ_MASK_PAD by convention.
    ggml_tensor * in_kq = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, T);
    ggml_set_input(in_kq);

    // ---- splice ----------------------------------------------------------
    ggml_tensor * h = ggml_get_rows(ctx, m.token_embd, in_ids);   // [H, T]
    h = ggml_add(ctx, ggml_mul(ctx, h, in_mask), in_audio);

    const float scale = 1.0f / sqrtf((float) D);
    for (uint32_t il = 0; il < hp.n_layer; ++il) {
        const LmLayer & w = m.layers[il];

        ggml_tensor * n = rms(ctx, h, w.attn_norm, hp.rms_eps);
        ggml_tensor * q = ggml_mul_mat(ctx, w.attn_q, n);
        ggml_tensor * k = ggml_mul_mat(ctx, w.attn_k, n);
        ggml_tensor * v = ggml_mul_mat(ctx, w.attn_v, n);

        q = ggml_reshape_3d(ctx, q, D, Nh, T);
        k = ggml_reshape_3d(ctx, k, D, Nkv, T);
        v = ggml_reshape_3d(ctx, v, D, Nkv, T);

        // Qwen3 per-head RMSNorm over the head dim, BEFORE RoPE.
        q = ggml_mul(ctx, ggml_rms_norm(ctx, q, hp.rms_eps), w.attn_q_norm);
        k = ggml_mul(ctx, ggml_rms_norm(ctx, k, hp.rms_eps), w.attn_k_norm);

        q = ggml_rope_ext(ctx, q, in_pos, NULL, (int) D, GGML_ROPE_TYPE_NEOX, 0,
                          hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, in_pos, NULL, (int) D, GGML_ROPE_TYPE_NEOX, 0,
                          hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // [D, head, pos] -> [D, pos, head] for the attention contraction.
        ggml_tensor * qp = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [D, T, Nh]
        ggml_tensor * kp = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  // [D, T, Nkv]
        ggml_tensor * vp = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));  // [D, T, Nkv]

        // GQA falls out of ggml's mul_mat broadcast: query head h reads KV head
        // h / (Nh/Nkv). No repeat needed.
        ggml_tensor * kq = ggml_mul_mat(ctx, kp, qp);                 // [T, T, Nh]
        kq = ggml_soft_max_ext(ctx, kq, in_kq, scale, 0.0f);
        ggml_tensor * vt = ggml_cont(ctx, ggml_transpose(ctx, vp));   // [T, D, Nkv]
        ggml_tensor * o = ggml_mul_mat(ctx, vt, kq);                  // [D, T, Nh]
        o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));         // [D, Nh, T]
        o = ggml_reshape_2d(ctx, o, H, T);
        h = ggml_add(ctx, h, ggml_mul_mat(ctx, w.attn_output, o));

        ggml_tensor * n2 = rms(ctx, h, w.ffn_norm, hp.rms_eps);
        ggml_tensor * gate = ggml_silu(ctx, ggml_mul_mat(ctx, w.ffn_gate, n2));
        ggml_tensor * up = ggml_mul_mat(ctx, w.ffn_up, n2);
        h = ggml_add(ctx, h, ggml_mul_mat(ctx, w.ffn_down, ggml_mul(ctx, gate, up)));

        // Deepstack: merger k lands in LM layer k. The values are already zero
        // at text rows, so a plain add is the masking.
        if ((int64_t) il < n_inject) {
            h = ggml_add(ctx, h, in_merge[(size_t) il]);
        }
    }

    h = rms(ctx, h, m.output_norm, hp.rms_eps);
    // Only the final position matters for the fixture check.
    ggml_tensor * last = ggml_view_2d(ctx, h, H, 1, h->nb[1], (size_t) (T - 1) * h->nb[1]);
    ggml_tensor * logits = ggml_mul_mat(ctx, m.output, ggml_cont(ctx, last));  // [n_vocab, 1]
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        fprintf(stderr, "[MOSS] LM graph allocation failed\n");
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(in_ids, ids.data(), 0, ids.size() * sizeof(int32_t));
    {
        std::vector<int32_t> pos((size_t) T);
        for (int64_t i = 0; i < T; ++i) {
            pos[(size_t) i] = (int32_t) i;
        }
        ggml_backend_tensor_set(in_pos, pos.data(), 0, pos.size() * sizeof(int32_t));
    }
    ggml_backend_tensor_set(in_mask, text_mask.data(), 0, text_mask.size() * sizeof(float));
    ggml_backend_tensor_set(in_audio, audio_vals.data(), 0, audio_vals.size() * sizeof(float));
    for (int64_t k = 0; k < n_inject; ++k) {
        ggml_backend_tensor_set(in_merge[(size_t) k], merger_vals[(size_t) k].data(), 0,
                                merger_vals[(size_t) k].size() * sizeof(float));
    }
    {
        std::vector<float> kq((size_t) T * T, 0.0f);
        for (int64_t i = 0; i < T; ++i) {        // query
            for (int64_t j = 0; j < T; ++j) {    // key
                kq[(size_t) i * T + j] = (j <= i) ? 0.0f : -INFINITY;
            }
        }
        ggml_backend_tensor_set(in_kq, kq.data(), 0, kq.size() * sizeof(float));
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[MOSS] LM graph compute failed\n");
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    out_logits->resize((size_t) hp.n_vocab);
    ggml_backend_tensor_get(logits, out_logits->data(), 0, out_logits->size() * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return true;
}

}  // namespace moss
