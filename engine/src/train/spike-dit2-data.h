#pragma once
// spike-dit2-data.h — ROUND 2 DiT spike: real preprocess-cache data + the
// rectified-flow objective of docs/plans/2026-07-27-training-studio-design.md §4.5.
//
// EVIDENCE CODE, NOT PRODUCT CODE.
//
// What lives here:
//   * D2Sample          — one preprocessed song read straight off the cache
//                         (target_latents / context_latents / encoder_hidden_states
//                         / encoder_attention_mask), no resampling, no synthesis.
//   * d2_crop()         — a contiguous latent window, patch-aligned.
//   * d2_flow_target()  — x1 ~ N(0,1) seeded, t ~ logit-normal(mu,sigma),
//                         xt = t*x1 + (1-t)*x0, v = x1 - x0.  This is exactly the
//                         sampler's convention, verified against the shipped
//                         solvers: euler.lua does xt <- xt - vt*(t_curr - t_prev)
//                         and hot-step-sampler.h:1277 does x0_hat = xt - vt*t,
//                         i.e. x(t) = t*noise + (1-t)*data and dx/dt = x1 - x0.
//   * d2_stub_adapter() — a 2-tensor ZERO LoRA used only to make dit_ggml_load()
//                         take its skip_fusion branch (see spike-dit2-graph.h).
//   * mask builders     — the real sliding-window self-attention mask
//                         (acestep.sliding_window = 128 on this base) and the
//                         real cross-attention encoder-padding mask.

#include "safetensors.h"
#include "train/lm-common.h"  // LmRng
#include "train/st-write.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ─── one preprocessed song ──────────────────────────────────────────────────

struct D2Sample {
    std::string        path;
    int                T     = 0;  // latent frames
    int                Oc    = 0;  // target_latents channels (64)
    int                Cc    = 0;  // context_latents channels (128)
    int                enc_S = 0;
    int                enc_H = 0;
    std::vector<float> lat;       // [T, Oc]   frame-major
    std::vector<float> ctxl;      // [T, Cc]   frame-major
    std::vector<float> enc;       // [enc_S, enc_H]
    std::vector<float> enc_mask;  // [enc_S] 1 = real, 0 = padding
};

static const float * d2_st_f32(const STFile & st, const char * name, const STEntry ** out_e) {
    const STEntry * e = st_find(st, name);
    if (!e || e->dtype != "F32") {
        return nullptr;
    }
    *out_e = e;
    return (const float *) st_data(st, *e);
}

static bool d2_load_sample(const char * path, D2Sample * s) {
    STFile st;
    if (!st_open(&st, path)) {
        fprintf(stderr, "[D2] FATAL: cannot open %s\n", path);
        return false;
    }
    const STEntry *e_lat = nullptr, *e_ctx = nullptr, *e_enc = nullptr, *e_em = nullptr;
    const float *  lat = d2_st_f32(st, "target_latents", &e_lat);
    const float *  ctxl = d2_st_f32(st, "context_latents", &e_ctx);
    const float *  enc = d2_st_f32(st, "encoder_hidden_states", &e_enc);
    const float *  em  = d2_st_f32(st, "encoder_attention_mask", &e_em);
    if (!lat || !ctxl || !enc) {
        fprintf(stderr, "[D2] FATAL: %s is missing an F32 target_latents/context_latents/encoder_hidden_states\n",
                path);
        st_close(&st);
        return false;
    }
    s->path  = path;
    s->T     = (int) e_lat->shape[0];
    s->Oc    = (int) e_lat->shape[1];
    s->Cc    = (int) e_ctx->shape[1];
    s->enc_S = (int) e_enc->shape[0];
    s->enc_H = (int) e_enc->shape[1];
    if ((int) e_ctx->shape[0] != s->T) {
        fprintf(stderr, "[D2] FATAL: context_latents frames %d != target_latents frames %d\n",
                (int) e_ctx->shape[0], s->T);
        st_close(&st);
        return false;
    }
    s->lat.assign(lat, lat + (size_t) s->T * s->Oc);
    s->ctxl.assign(ctxl, ctxl + (size_t) s->T * s->Cc);
    s->enc.assign(enc, enc + (size_t) s->enc_S * s->enc_H);
    s->enc_mask.assign((size_t) s->enc_S, 1.0f);
    if (em && (int) e_em->shape[0] == s->enc_S) {
        s->enc_mask.assign(em, em + s->enc_S);
    }
    st_close(&st);
    int enc_real = 0;
    for (size_t i = 0; i < s->enc_mask.size(); i++) {
        if (s->enc_mask[i] > 0.5f) {
            enc_real++;
        }
    }
    fprintf(stderr,
            "[D2] sample %s\n[D2]   target_latents [%d,%d]  context_latents [%d,%d]  enc [%d,%d]  enc_real=%d\n", path,
            s->T, s->Oc, s->T, s->Cc, s->enc_S, s->enc_H, enc_real);
    return true;
}

// ─── rectified-flow objective (design §4.5) ─────────────────────────────────

struct D2Flow {
    float              t = 0.0f;
    std::vector<float> xt;  // [crop, Oc] frame-major
    std::vector<float> v;   // [crop, Oc] frame-major  (= x1 - x0)
};

// logit-normal timestep: t = sigmoid(mu + sigma * z), z ~ N(0,1).
static inline float d2_logit_normal_t(LmRng * rng, float mu, float sigma) {
    const float z = lm_rng_normal(rng);
    const float u = mu + sigma * z;
    return 1.0f / (1.0f + expf(-u));
}

// x0 = the real latent crop; x1 = seeded gaussian; xt = t*x1 + (1-t)*x0; v = x1 - x0.
static void d2_flow_target(const float * x0, size_t n, float t, LmRng * rng, D2Flow * out) {
    out->t = t;
    out->xt.resize(n);
    out->v.resize(n);
    for (size_t i = 0; i < n; i++) {
        const float x1 = lm_rng_normal(rng);
        out->xt[i]     = t * x1 + (1.0f - t) * x0[i];
        out->v[i]      = x1 - x0[i];
    }
}

// ─── attention masks (exactly what the sampler uploads) ─────────────────────

// Self-attention sliding window, F16, [S(kv), S(q)] — hot-step-sampler.h:266-287.
static void d2_sa_mask(int S, int win, std::vector<uint16_t> * out) {
    out->assign((size_t) S * S, 0);
    for (int qi = 0; qi < S; qi++) {
        for (int ki = 0; ki < S; ki++) {
            const int  dist   = (qi > ki) ? (qi - ki) : (ki - qi);
            const bool in_win = (win <= 0) || (S <= win) || (dist <= win);
            (*out)[(size_t) qi * S + ki] = ggml_fp32_to_fp16(in_win ? 0.0f : -INFINITY);
        }
    }
}

// Cross-attention encoder-padding mask, F16, [enc_S(kv), S(q)].
static void d2_ca_mask(int enc_S, int S, const std::vector<float> & enc_mask, std::vector<uint16_t> * out) {
    out->assign((size_t) enc_S * S, 0);
    for (int qi = 0; qi < S; qi++) {
        for (int ki = 0; ki < enc_S; ki++) {
            const bool real = enc_mask.empty() || enc_mask[(size_t) ki] > 0.5f;
            (*out)[(size_t) qi * enc_S + ki] = ggml_fp32_to_fp16(real ? 0.0f : -INFINITY);
        }
    }
}

// ─── stub adapter (unfused-load key) ────────────────────────────────────────
//
// dit_ggml_load() only takes its `skip_fusion` branch when adapter_path is
// non-null AND adapter_mode is runtime/runtime_lowrank/merge (dit.h:552-555),
// and it then HARD-FAILS the load if the adapter merges/loads no tensors
// (dit.h:762-766 / 834-845).  A real trainer would have this problem too, so
// the spike solves it the cheapest honest way: write a 2-tensor, rank-1,
// all-ZERO LoRA covering one real projection.  It loads, contributes a delta
// that is identically zero, and the training graph never references m->lora
// (it is freed immediately after the load).  Equivalence class: EXACT — a zero
// delta changes no forward value; only the fusion decision is affected.
static bool d2_stub_adapter(const std::string & path, int64_t in_feat, int64_t out_feat) {
    std::vector<float> a((size_t) in_feat, 0.0f);   // lora_A [rank=1, in]
    std::vector<float> b((size_t) out_feat, 0.0f);  // lora_B [out, rank=1]
    std::vector<STWTensor> ts;
    ts.push_back({ "layers.0.self_attn.q_proj.lora_A.weight", { 1, in_feat }, a.data() });
    ts.push_back({ "layers.0.self_attn.q_proj.lora_B.weight", { out_feat, 1 }, b.data() });
    std::vector<std::pair<std::string, std::string>> md;
    md.push_back({ "producer", "ace-train spike dit2 (unfused-load key, zero delta)" });
    return st_write_file(path.c_str(), ts, md, STW_F32);
}
