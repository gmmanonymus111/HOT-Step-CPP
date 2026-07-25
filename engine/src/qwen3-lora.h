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
};

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
    }
    return y;
}

// Convenience: fetch a slot pair from an optional per-layer slot table.
static inline const QwLoraPair * qwen3_lora_slot(const QwLoraLayer * ll, QwLoraSlot s) {
    return ll ? &ll->p[s] : nullptr;
}
