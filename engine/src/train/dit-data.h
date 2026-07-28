#pragma once
// dit-data.h — `ace-train train-dit` sample cache, crop sampler and the
// rectified-flow objective (plan §3.1).
//
//   DitSample            one preprocessed song, host-resident for the whole run
//   dit_scan_samples()   preprocess_meta.json order -> per-song safetensors
//   dit_load_channel_stats()  channel_stats.json -> inverse-std weights, mean 1
//   dit_sample_crop()    random patch-aligned window (Side-Step parity)
//   dit_sample_t()       logit-normal timestep with an interval-expert window
//   dit_flow_target()    x1 ~ N(0,1), xt = t*x1 + (1-t)*x0, v = x1 - x0   (D15)
//   dit_sa_mask/dit_ca_mask   byte-for-byte the sampler's masks
//   dit_flow_snr_w()     the §3.5 timestep weight
//
// docs/plans/2026-07-28-dit-trainer-implementation.md §3.1 / §3.5

#include "model-registry.h"  // registry_list_dir
#include "safetensors.h"
#include "train/lm-common.h"  // LmRng, pm_* (via preprocess-io.h), lm_log
#include "yyjson.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ─── one preprocessed song ──────────────────────────────────────────────────

struct DitSample {
    std::string        path, id;
    int                T = 0, Oc = 0, Cc = 0;  // frames, 64, 128
    int                enc_S = 0, enc_H = 0;
    int                enc_S_genre = 0;
    std::vector<float> lat, ctxl, enc, enc_mask;    // frame-major
    std::vector<float> enc_genre, enc_mask_genre;   // empty when absent
};

struct DitChannelStats {  // channel_stats.json
    bool               ok = false;
    std::vector<float> weight;  // [64], mean == 1
};

struct DitCrop {
    int start = 0, len = 0;
};

// ─── safetensors helpers ────────────────────────────────────────────────────

static const float * dit_st_f32(const STFile & st, const char * name, const STEntry ** out_e) {
    const STEntry * e = st_find(st, name);
    if (!e || e->dtype != "F32") {
        return nullptr;
    }
    *out_e = e;
    return (const float *) st_data(st, *e);
}

// One cached song. Returns false (with `why`) for anything malformed; the caller
// turns that into a `warn` and skips the song.
static bool dit_load_sample(const std::string & path, DitSample * s, std::string * why) {
    STFile st;
    if (!st_open(&st, path.c_str())) {
        *why = "cannot open";
        return false;
    }
    const STEntry *e_lat = nullptr, *e_ctx = nullptr, *e_enc = nullptr, *e_em = nullptr;
    const float *  lat  = dit_st_f32(st, "target_latents", &e_lat);
    const float *  ctxl = dit_st_f32(st, "context_latents", &e_ctx);
    const float *  enc  = dit_st_f32(st, "encoder_hidden_states", &e_enc);
    const float *  em   = dit_st_f32(st, "encoder_attention_mask", &e_em);
    if (!lat || !ctxl || !enc || e_lat->n_dims != 2 || e_ctx->n_dims != 2 || e_enc->n_dims != 2) {
        *why = "missing or mistyped target_latents/context_latents/encoder_hidden_states";
        st_close(&st);
        return false;
    }
    s->path  = path;
    s->T     = (int) e_lat->shape[0];
    s->Oc    = (int) e_lat->shape[1];
    s->Cc    = (int) e_ctx->shape[1];
    s->enc_S = (int) e_enc->shape[0];
    s->enc_H = (int) e_enc->shape[1];
    if ((int) e_ctx->shape[0] != s->T || s->T <= 0 || s->Oc <= 0 || s->Cc <= 0 || s->enc_S <= 0 || s->enc_H <= 0) {
        *why = "context/target frame mismatch or empty tensor";
        st_close(&st);
        return false;
    }
    s->lat.assign(lat, lat + (size_t) s->T * (size_t) s->Oc);
    s->ctxl.assign(ctxl, ctxl + (size_t) s->T * (size_t) s->Cc);
    s->enc.assign(enc, enc + (size_t) s->enc_S * (size_t) s->enc_H);
    s->enc_mask.assign((size_t) s->enc_S, 1.0f);
    if (em && e_em->n_dims == 1 && (int) e_em->shape[0] == s->enc_S) {
        s->enc_mask.assign(em, em + s->enc_S);
    }

    // genre-conditioned encoder states (preprocess writes these when the dataset
    // has a genre ratio) — optional, used by --genre-ratio (D14).
    const STEntry *e_g = nullptr, *e_gm = nullptr;
    const float *  eg  = dit_st_f32(st, "encoder_hidden_states_genre", &e_g);
    const float *  egm = dit_st_f32(st, "encoder_attention_mask_genre", &e_gm);
    if (eg && e_g->n_dims == 2 && (int) e_g->shape[1] == s->enc_H) {
        s->enc_S_genre = (int) e_g->shape[0];
        s->enc_genre.assign(eg, eg + (size_t) s->enc_S_genre * (size_t) s->enc_H);
        s->enc_mask_genre.assign((size_t) s->enc_S_genre, 1.0f);
        if (egm && e_gm->n_dims == 1 && (int) e_gm->shape[0] == s->enc_S_genre) {
            s->enc_mask_genre.assign(egm, egm + s->enc_S_genre);
        }
    }
    st_close(&st);
    return true;
}

// `dit_path` recorded by preprocess for this variant (§4.2's base-match rule).
static std::string dit_meta_dit_path(const char * tensors_dir) {
    const std::string p = lm_join(tensors_dir, "preprocess_meta.json");
    FILE *            f = fopen(p.c_str(), "rb");
    if (!f) {
        return std::string();
    }
    std::string buf;
    char        tmp[8192];
    size_t      n;
    while ((n = fread(tmp, 1, sizeof(tmp), f)) > 0) {
        buf.append(tmp, n);
    }
    fclose(f);
    yyjson_doc * doc = yyjson_read(buf.data(), buf.size(), 0);
    if (!doc) {
        return std::string();
    }
    yyjson_val *      root = yyjson_doc_get_root(doc);
    const std::string out  = (root && yyjson_is_obj(root)) ? pm_js_str(root, "dit_path") : std::string();
    yyjson_doc_free(doc);
    return out;
}

// Scan the variant directory. preprocess_meta.json supplies the order and the
// per-song file names; a variant without it falls back to a directory listing so
// a hand-assembled cache still trains.
static bool dit_scan_samples(const char * tensors_dir, int limit, std::vector<DitSample> * out, std::string * err) {
    out->clear();
    std::vector<std::pair<std::string, std::string>> files;  // (file, id)

    const std::string meta_path = lm_join(tensors_dir, "preprocess_meta.json");
    FILE *            f         = fopen(meta_path.c_str(), "rb");
    if (f) {
        std::string buf;
        char        tmp[8192];
        size_t      n;
        while ((n = fread(tmp, 1, sizeof(tmp), f)) > 0) {
            buf.append(tmp, n);
        }
        fclose(f);
        yyjson_doc * doc = yyjson_read(buf.data(), buf.size(), 0);
        if (doc) {
            yyjson_val * root = yyjson_doc_get_root(doc);
            yyjson_val * arr  = (root && yyjson_is_obj(root)) ? yyjson_obj_get(root, "samples") : NULL;
            if (arr && yyjson_is_arr(arr)) {
                size_t       idx, mx;
                yyjson_val * it;
                yyjson_arr_foreach(arr, idx, mx, it) {
                    if (!yyjson_is_obj(it)) {
                        continue;
                    }
                    if (!pm_js_bool(it, "ok", false)) {
                        continue;
                    }
                    const std::string file = pm_js_str(it, "file");
                    if (!file.empty()) {
                        files.push_back({ file, pm_js_str(it, "id") });
                    }
                }
            }
            yyjson_doc_free(doc);
        }
    }
    if (files.empty()) {
        std::vector<std::string> names;
        registry_list_dir(tensors_dir, &names);
        std::sort(names.begin(), names.end());
        for (size_t i = 0; i < names.size(); i++) {
            const std::string & nm = names[i];
            if (nm.size() > 12 && nm.compare(nm.size() - 12, 12, ".safetensors") == 0) {
                files.push_back({ nm, nm.substr(0, nm.size() - 12) });
            }
        }
    }
    if (files.empty()) {
        *err = std::string("no cached songs in ") + tensors_dir;
        return false;
    }
    if (limit > 0 && (int) files.size() > limit) {
        files.resize((size_t) limit);
    }

    for (size_t i = 0; i < files.size(); i++) {
        DitSample   s;
        std::string why;
        const std::string path = lm_join(tensors_dir, files[i].first);
        if (!dit_load_sample(path, &s, &why)) {
            lm_log("warn", "SKIP " + files[i].first + ": " + why);
            continue;
        }
        s.id = files[i].second;
        out->push_back(std::move(s));
    }
    if (out->empty()) {
        *err = std::string("every cached song in ") + tensors_dir + " was unusable";
        return false;
    }
    return true;
}

// channel_std[64] -> w = 1/max(std,1e-6), then w /= mean(w) (mean exactly 1).
// Missing/short file => ok=false and channel balancing is disabled.
static bool dit_load_channel_stats(const char * tensors_dir, DitChannelStats * out) {
    out->ok = false;
    out->weight.clear();
    const std::string p = lm_join(tensors_dir, "channel_stats.json");
    FILE *            f = fopen(p.c_str(), "rb");
    if (!f) {
        return false;
    }
    std::string buf;
    char        tmp[8192];
    size_t      n;
    while ((n = fread(tmp, 1, sizeof(tmp), f)) > 0) {
        buf.append(tmp, n);
    }
    fclose(f);
    yyjson_doc * doc = yyjson_read(buf.data(), buf.size(), 0);
    if (!doc) {
        return false;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    yyjson_val * arr  = (root && yyjson_is_obj(root)) ? yyjson_obj_get(root, "channel_std") : NULL;
    if (arr && yyjson_is_arr(arr)) {
        size_t       idx, mx;
        yyjson_val * it;
        yyjson_arr_foreach(arr, idx, mx, it) {
            if (yyjson_is_num(it)) {
                const double sd = yyjson_get_num(it);
                out->weight.push_back(1.0f / std::max(1e-6f, (float) sd));
            }
        }
    }
    yyjson_doc_free(doc);
    if (out->weight.empty()) {
        out->weight.clear();
        return false;
    }
    double mean = 0.0;
    for (size_t i = 0; i < out->weight.size(); i++) {
        mean += (double) out->weight[i];
    }
    mean /= (double) out->weight.size();
    if (!(mean > 0.0)) {
        out->weight.clear();
        return false;
    }
    for (size_t i = 0; i < out->weight.size(); i++) {
        out->weight[i] = (float) ((double) out->weight[i] / mean);
    }
    out->ok = true;
    return true;
}

// ─── crop sampler (D10) ─────────────────────────────────────────────────────

// len = min(crop, T) rounded DOWN to a multiple of `patch`; start uniform over
// the patch-aligned starts in [0, T - len].
//
// DEVIATION (E1, documented in the handoff): §3.1 says "start = uniform(0, T-len)
// rounded down to a multiple of patch". That composition is NOT uniform over the
// aligned starts — the top bucket is short — and would fail T10's own ±3σ
// uniformity assertion. Drawing uniformly over the (T-len)/patch + 1 aligned
// starts is the same support with an exactly uniform distribution.
static DitCrop dit_sample_crop(LmRng * rng, int T, int crop, int patch) {
    DitCrop c;
    if (patch < 1) {
        patch = 1;
    }
    int len = (crop > 0 && crop < T) ? crop : T;
    len -= len % patch;
    if (len < patch) {
        len = std::min(T - (T % patch), patch);
    }
    const int n_starts = (T - len) / patch + 1;
    c.len              = len;
    c.start            = patch * (int) lm_rng_below(rng, (uint64_t) std::max(1, n_starts));
    GGML_ASSERT(c.len > 0 && c.len <= T);
    GGML_ASSERT(c.start >= 0 && c.start + c.len <= T);
    return c;
}

// ─── timestep (D12) ─────────────────────────────────────────────────────────

// t = sigmoid(mu + sigma*z), clamped to [1e-5, 1-1e-5]; rejection-resampled
// (max 64 tries, then clamped) into the interval-expert window [t_min, t_max].
static float dit_sample_t(LmRng * rng, float mu, float sigma, float t_min, float t_max) {
    float t = 0.5f;
    for (int i = 0; i < 64; i++) {
        const float z = lm_rng_normal(rng);
        t             = 1.0f / (1.0f + expf(-(mu + sigma * z)));
        if (t < 1e-5f) {
            t = 1e-5f;
        }
        if (t > 1.0f - 1e-5f) {
            t = 1.0f - 1e-5f;
        }
        if (t >= t_min && t <= t_max) {
            return t;
        }
    }
    return std::min(std::max(t, t_min), t_max);
}

// ─── rectified-flow target (D15) ────────────────────────────────────────────
//
// ONE rng stream, drawn in index order. This exact ordering is the verifier's
// E[v^2] = 1.672071 fingerprint (T9) — do not reorder it.
static void dit_flow_target(const float * x0, size_t n, float t, LmRng * rng, std::vector<float> * xt,
                            std::vector<float> * v) {
    xt->resize(n);
    v->resize(n);
    for (size_t i = 0; i < n; i++) {
        const float x1 = lm_rng_normal(rng);
        (*xt)[i]       = t * x1 + (1.0f - t) * x0[i];
        (*v)[i]        = x1 - x0[i];
    }
}

// ─── attention masks (byte-for-byte the sampler's) ──────────────────────────

// Self-attention sliding window, F16, [S(kv), S(q)].
static void dit_sa_mask(int S, int win, std::vector<uint16_t> * out) {
    out->assign((size_t) S * (size_t) S, 0);
    for (int qi = 0; qi < S; qi++) {
        for (int ki = 0; ki < S; ki++) {
            const int  dist   = (qi > ki) ? (qi - ki) : (ki - qi);
            const bool in_win = (win <= 0) || (S <= win) || (dist <= win);
            (*out)[(size_t) qi * (size_t) S + (size_t) ki] = ggml_fp32_to_fp16(in_win ? 0.0f : -INFINITY);
        }
    }
}

// Cross-attention encoder-padding mask, F16, [enc_S(kv), S(q)].
static void dit_ca_mask(int enc_S, int S, const std::vector<float> & m, std::vector<uint16_t> * out) {
    out->assign((size_t) enc_S * (size_t) S, 0);
    for (int qi = 0; qi < S; qi++) {
        for (int ki = 0; ki < enc_S; ki++) {
            const bool real = m.empty() || m[(size_t) ki] > 0.5f;
            (*out)[(size_t) qi * (size_t) enc_S + (size_t) ki] = ggml_fp32_to_fp16(real ? 0.0f : -INFINITY);
        }
    }
}

// ─── flow_snr timestep weight (§3.5) ────────────────────────────────────────
//
// w(t) = ((1-t)^t_bias) / (t * (1-t)), clamped to snr_gamma.
// Side-Step clamps t to [1e-4, 1-1e-4] before the division (lora_module.py:579).
static float dit_flow_snr_w(float t, float t_bias, float snr_gamma) {
    float tf = t;
    if (tf < 1e-4f) {
        tf = 1e-4f;
    }
    if (tf > 1.0f - 1e-4f) {
        tf = 1.0f - 1e-4f;
    }
    float w = powf(1.0f - tf, t_bias) / (tf * (1.0f - tf));
    if (w > snr_gamma) {
        w = snr_gamma;
    }
    return w;
}
