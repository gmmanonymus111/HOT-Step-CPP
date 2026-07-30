#pragma once
// qwen3-lora.h: minimal runtime-LoRA structs + graph helper for Qwen3 models.
//
// Kept deliberately tiny so qwen3-enc.h can include it without pulling in
// safetensors/json parsing.  The loader lives in lm-adapter.h.
//
// Runtime application (never merged into base weights, so it works on any
// base quantization):  y = W@x + B@(scale * (A@x))
//
// A: [H, r] (ggml ne0=H, ne1=r), B: [r, out] (ne0=r, ne1=out) — both loaded
// straight from PEFT lora_A/lora_B safetensors layouts with no transpose.
// scale = (lora_alpha / r) * user_scale, precomputed at load.
//
// Local HOT-Step feature — not upstream acestep.cpp.

#include "ggml.h"

#define QWEN3_LORA_MAX_LAYERS 64  // must cover QW3LM_MAX_LAYERS

enum QwLoraSlot {
    QW_LORA_Q = 0,
    QW_LORA_K,
    QW_LORA_V,
    QW_LORA_O,
    QW_LORA_GATE,
    QW_LORA_UP,
    QW_LORA_DOWN,
    QW_LORA_NSLOTS,
};

struct QwLoraPair {
    struct ggml_tensor * A     = nullptr;  // [H, r]
    struct ggml_tensor * B     = nullptr;  // [r, out]
    float                scale = 1.0f;     // (alpha/r) * user_scale

    // ─── LoKr (2026-07-30) ───────────────────────────────────────────────────
    //
    // Mutually exclusive with A/B: a site is either a LoRA or a LoKr, never
    // both. Populated by the TRAINER (train/lm-lokr.h) and, once it learns the
    // format, by the inference loader (lm-adapter.h) — which is the point of
    // hanging them here rather than on a trainer-private struct. One kron
    // implementation, one place to get the ggml indexing right, both paths.
    //
    // ggml layouts, matching what the DiT writes and adapter-merge.h reads:
    //   w1   [in_m, out_l]      w2   [in_n, out_k]        (monolithic)
    //   w2_a [dim,  out_k]      w2_b [in_n, dim]          (factorized)
    struct ggml_tensor * w1    = nullptr;
    struct ggml_tensor * w2    = nullptr;
    struct ggml_tensor * w2_a  = nullptr;
    struct ggml_tensor * w2_b  = nullptr;
    int64_t              in_m = 0, in_n = 0, out_l = 0, out_k = 0;
    float                lokr_scale = 1.0f;  // alpha / dim

    bool has_lokr() const { return w1 && (w2 || (w2_a && w2_b)); }
};

// LoKr delta: y += kron(w1, w2) . x, contracted factor-by-factor so the full
// [out, in] delta is never materialized.
//
// THE TOKEN AXIS MUST NOT REACH ne2 OF THESE MUL_MATS. Both contractions have a
// 2-D trainable factor as src0; if src1 carries the token count in ne2, ggml
// emits the weight gradient as out_prod(src1, grad) with dst->ne[2] == S, and
// ggml-cuda's out_prod takes its `dps2 > 1` fallback — one cublasSgemm PER
// TOKEN (out-prod.cu:96-108). On the DiT that made LoKr training 16.4x slower
// than the same run with a LoRA before it was found (2026-07-30).
//
// [in_n, in_m, S] and [in_n, in_m*S] are the SAME BYTES, so folding the token
// axis into the column count is a pure reshape and leaves ne2 == 1. The 3-D
// form is restored only around the permutes, which genuinely need it.
static inline struct ggml_tensor * qwen3_lokr_delta(struct ggml_context * ctx,
                                                    const QwLoraPair *    p,
                                                    struct ggml_tensor *  x,
                                                    struct ggml_tensor *  y) {
    struct ggml_tensor * xc = ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);
    const int64_t        S  = ggml_nelements(xc) / xc->ne[0];

    struct ggml_tensor * X2 = ggml_reshape_2d(ctx, xc, p->in_n, p->in_m * S);
    struct ggml_tensor * T1 = p->w2 ? ggml_mul_mat(ctx, p->w2, X2)
                                    : ggml_mul_mat(ctx, p->w2_a, ggml_mul_mat(ctx, p->w2_b, X2));
    struct ggml_tensor * T13 = ggml_reshape_3d(ctx, T1, p->out_k, p->in_m, S);
    struct ggml_tensor * T1p = ggml_cont(ctx, ggml_permute(ctx, T13, 1, 0, 2, 3));
    struct ggml_tensor * T1f = ggml_reshape_2d(ctx, T1p, p->in_m, p->out_k * S);
    struct ggml_tensor * w1s = ggml_scale(ctx, p->w1, p->lokr_scale);
    struct ggml_tensor * T2  = ggml_mul_mat(ctx, w1s, T1f);
    struct ggml_tensor * T23 = ggml_reshape_3d(ctx, T2, p->out_l, p->out_k, S);
    struct ggml_tensor * T2p = ggml_cont(ctx, ggml_permute(ctx, T23, 1, 0, 2, 3));
    return ggml_add(ctx, y, ggml_reshape_4d(ctx, T2p, y->ne[0], y->ne[1], y->ne[2], y->ne[3]));
}

struct QwLoraLayer {
    QwLoraPair p[QW_LORA_NSLOTS];
};

// Set while loading a Qwen3 model whose layers will carry LoRA slots:
// disables QKV / gate-up weight fusion so each projection stays individually
// addressable (mirrors the DiT runtime-adapter fusion skip, dit.h:420).
inline bool g_qwen3_load_no_fuse = false;

// y = W@x (+ LoRA delta when the pair is populated).
static inline struct ggml_tensor * qwen3_linear_lora(struct ggml_context * ctx,
                                                     struct ggml_tensor *  w,
                                                     const QwLoraPair *    p,
                                                     struct ggml_tensor *  x) {
    struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    if (p && p->A && p->B) {
        struct ggml_tensor * t = ggml_mul_mat(ctx, p->A, x);   // [r, S]
        t = ggml_scale(ctx, t, p->scale);                      // cheapest on the rank-r side
        y = ggml_add(ctx, y, ggml_mul_mat(ctx, p->B, t));      // [out, S]
    } else if (p && p->has_lokr()) {
        y = qwen3_lokr_delta(ctx, p, x, y);
    }
    return y;
}

// Convenience: fetch a slot pair from an optional per-layer slot table.
static inline const QwLoraPair * qwen3_lora_slot(const QwLoraLayer * ll, QwLoraSlot s) {
    return ll ? &ll->p[s] : nullptr;
}
