#pragma once
// lm-common.h — shared small utilities for `ace-train train-lm`.
//
// Scope (plan §1.1 / §3.2 / §3.7):
//   * stmd_read()      — read ONLY the __metadata__ object of a safetensors
//                        file (engine/src/safetensors.h deliberately skips it)
//   * sha1_16()        — 16 hex chars of SHA-1, used for the `_tok` identity
//   * LmRng            — deterministic xoshiro256** + Fisher-Yates shuffle
//   * lm_json_escape() — JSON string escaping for the JSONL stream
//   * lm_vram_*_mb()   — CUDA device memory probes
//   * lm_size_label_from_config() — "0.6B" / "1.7B" / "4B" / "custom"
//
// docs/plans/2026-07-27-lm-trainer-implementation.md
//
// NOTE: this header (and every other lm-*.h) is only ever included from
// engine/tools/ace-train.cpp. `jl()` / `json_escape()` are defined there, but
// AFTER the include, so they are forward-declared below rather than duplicated
// (plan §3.1: "Reuse the existing g_jsonl / jl() / json_escape()").

#include "config-json.h"           // Qwen3LMConfig
#include "train/preprocess-io.h"   // pm_* path/json helpers, stw_replace_file
#include "yyjson.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// ─── JSONL plumbing (defined in ace-train.cpp) ──────────────────────────────

static void jl(const char * fmt, ...);

// ─── JSON escaping ──────────────────────────────────────────────────────────

static std::string lm_json_escape(const std::string & s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (size_t i = 0; i < s.size(); i++) {
        const unsigned char c = (unsigned char) s[i];
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\b': o += "\\b";  break;
            case '\f': o += "\\f";  break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if (c < 0x20) {
                    char b[8];
                    snprintf(b, sizeof(b), "\\u%04x", (unsigned) c);
                    o += b;
                } else {
                    o.push_back((char) c);
                }
        }
    }
    return o;
}

// One `log` event (§2.2). level ∈ info|warn|error.
static void lm_log(const char * level, const std::string & msg) {
    jl("{\"type\":\"log\",\"level\":\"%s\",\"message\":\"%s\"}", level, lm_json_escape(msg).c_str());
    fprintf(stderr, "[train-lm] %s: %s\n", level, msg.c_str());
}

// One `fatal` event (§2.2). Always the last line before a non-zero exit.
static void lm_fatal(const char * reason, const std::string & msg, const std::string & extra = std::string()) {
    jl("{\"type\":\"fatal\",\"reason\":\"%s\",\"message\":\"%s\"%s}", reason, lm_json_escape(msg).c_str(),
       extra.c_str());
    fprintf(stderr, "[train-lm] FATAL (%s): %s\n", reason, msg.c_str());
}

// ─── deterministic RNG (xoshiro256** seeded through splitmix64) ─────────────

struct LmRng {
    uint64_t s[4];
};

static inline uint64_t lm_splitmix64(uint64_t * x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ull);
    z          = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z          = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static inline void lm_rng_seed(LmRng * r, uint64_t seed) {
    uint64_t x = seed ? seed : 0x123456789ABCDEFull;
    for (int i = 0; i < 4; i++) {
        r->s[i] = lm_splitmix64(&x);
    }
}

static inline uint64_t lm_rng_next(LmRng * r) {
    const uint64_t s1     = r->s[1];
    const uint64_t result = ((s1 * 5ull) << 7 | (s1 * 5ull) >> 57) * 9ull;
    const uint64_t t      = s1 << 17;
    r->s[2] ^= r->s[0];
    r->s[3] ^= s1;
    r->s[1] ^= r->s[2];
    r->s[0] ^= r->s[3];
    r->s[2] ^= t;
    r->s[3] = (r->s[3] << 45) | (r->s[3] >> 19);
    return result;
}

// Uniform in [0, n) without modulo bias.
static inline uint64_t lm_rng_below(LmRng * r, uint64_t n) {
    if (n <= 1) {
        return 0;
    }
    const uint64_t limit = UINT64_MAX - (UINT64_MAX % n);
    uint64_t       v;
    do {
        v = lm_rng_next(r);
    } while (v >= limit);
    return v % n;
}

static inline float lm_rng_uniform(LmRng * r) {  // (0,1)
    return (float) ((lm_rng_next(r) >> 11) + 1) / (float) (1ull << 53);
}

static inline float lm_rng_normal(LmRng * r) {
    const float u1 = lm_rng_uniform(r);
    const float u2 = lm_rng_uniform(r);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
}

static inline void lm_rng_fill_normal(LmRng * r, std::vector<float> & v, float sigma) {
    for (size_t i = 0; i < v.size(); i++) {
        v[i] = lm_rng_normal(r) * sigma;
    }
}

// In-place Fisher-Yates.
static inline void lm_rng_shuffle(LmRng * r, std::vector<int> & v) {
    for (size_t i = v.size(); i > 1; i--) {
        const size_t j = (size_t) lm_rng_below(r, (uint64_t) i);
        std::swap(v[i - 1], v[j]);
    }
}

// ─── SHA-1 (only used for the FSQ-tokenizer identity string) ────────────────

struct LmSha1 {
    uint32_t h[5];
    uint64_t len;
    uint8_t  buf[64];
    size_t   buf_len;
};

static inline uint32_t lm_sha1_rol(uint32_t v, int b) {
    return (v << b) | (v >> (32 - b));
}

static void lm_sha1_block(LmSha1 * c, const uint8_t * p) {
    uint32_t w[80];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t) p[i * 4] << 24) | ((uint32_t) p[i * 4 + 1] << 16) | ((uint32_t) p[i * 4 + 2] << 8) |
               (uint32_t) p[i * 4 + 3];
    }
    for (int i = 16; i < 80; i++) {
        w[i] = lm_sha1_rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    }
    uint32_t a = c->h[0], b = c->h[1], d = c->h[2], e = c->h[3], f = c->h[4];
    for (int i = 0; i < 80; i++) {
        uint32_t k, t;
        if (i < 20) {
            t = (b & d) | ((~b) & e);
            k = 0x5A827999u;
        } else if (i < 40) {
            t = b ^ d ^ e;
            k = 0x6ED9EBA1u;
        } else if (i < 60) {
            t = (b & d) | (b & e) | (d & e);
            k = 0x8F1BBCDCu;
        } else {
            t = b ^ d ^ e;
            k = 0xCA62C1D6u;
        }
        const uint32_t tmp = lm_sha1_rol(a, 5) + t + f + k + w[i];
        f                  = e;
        e                  = d;
        d                  = lm_sha1_rol(b, 30);
        b                  = a;
        a                  = tmp;
    }
    c->h[0] += a;
    c->h[1] += b;
    c->h[2] += d;
    c->h[3] += e;
    c->h[4] += f;
}

static void lm_sha1_init(LmSha1 * c) {
    c->h[0]   = 0x67452301u;
    c->h[1]   = 0xEFCDAB89u;
    c->h[2]   = 0x98BADCFEu;
    c->h[3]   = 0x10325476u;
    c->h[4]   = 0xC3D2E1F0u;
    c->len    = 0;
    c->buf_len = 0;
}

static void lm_sha1_update(LmSha1 * c, const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;
    c->len += (uint64_t) n * 8ull;
    while (n > 0) {
        const size_t take = std::min(n, (size_t) 64 - c->buf_len);
        memcpy(c->buf + c->buf_len, p, take);
        c->buf_len += take;
        p += take;
        n -= take;
        if (c->buf_len == 64) {
            lm_sha1_block(c, c->buf);
            c->buf_len = 0;
        }
    }
}

static std::string lm_sha1_hex(LmSha1 * c) {
    const uint64_t bits = c->len;
    const uint8_t  pad  = 0x80;
    lm_sha1_update(c, &pad, 1);
    const uint8_t z = 0;
    while (c->buf_len != 56) {
        lm_sha1_update(c, &z, 1);
    }
    uint8_t lenbe[8];
    for (int i = 0; i < 8; i++) {
        lenbe[i] = (uint8_t) ((bits >> (56 - 8 * i)) & 0xFFu);
    }
    c->len = 0;  // the length bytes themselves are not counted
    memcpy(c->buf + 56, lenbe, 8);
    lm_sha1_block(c, c->buf);
    c->buf_len = 0;

    char out[41];
    for (int i = 0; i < 5; i++) {
        snprintf(out + i * 8, 9, "%08x", c->h[i]);
    }
    return std::string(out, 40);
}

// First 16 hex chars of SHA-1(s).
static std::string sha1_16(const std::string & s) {
    LmSha1 c;
    lm_sha1_init(&c);
    lm_sha1_update(&c, s.data(), s.size());
    return lm_sha1_hex(&c).substr(0, 16);
}

// Bump whenever the extract stage's float -> code mapping changes: it is folded
// into `_tok`, so every cached lm_codes.jsonl row is invalidated and re-encoded.
//   v1 — fsq-tok.h::fsq_encode_index (old vqp symmetry-preserving bound)
//   v2 — reference-conformant encoder (vqp >= 1.27.21 ResidualFSQ: soft clamp +
//        hard clamp), 100% match vs the PyTorch reference. Originally lived in
//        train/lm-fsq.h::fsq_encode_index_ref; that file is gone — the same
//        formula is now the ENGINE-WIDE encoder in src/fsq-quant.h, so extract
//        calls the shared tok_ggml_encode and codes stay bit-identical to v2.
#define LM_FSQ_ENCODER_VERSION "fsq-v2"

// _tok = sha1(dit_path + "\0" + size + "\0" + mtime_ms + "\0" + encoder)[0:16]  (§2.3)
static std::string lm_tok_identity(const std::string & dit_path) {
    long long bytes = 0, mtime = 0;
    pm_stat_file(dit_path, &bytes, &mtime);
    char b[64];
    snprintf(b, sizeof(b), "%lld", bytes);
    std::string s = dit_path;
    s.push_back('\0');
    s += b;
    s.push_back('\0');
    snprintf(b, sizeof(b), "%lld", mtime);
    s += b;
    s.push_back('\0');
    s += LM_FSQ_ENCODER_VERSION;
    return sha1_16(s);
}

// ─── safetensors __metadata__ reader (L14) ──────────────────────────────────
//
// engine/src/safetensors.h skips __metadata__ on purpose (safetensors.h:141),
// and we do not edit it. This reads just the header object.
static bool stmd_read(const char * path, std::map<std::string, std::string> * out) {
    out->clear();
    FILE * f = fopen(path, "rb");
    if (!f) {
        return false;
    }
    uint8_t hl[8];
    if (fread(hl, 1, 8, f) != 8) {
        fclose(f);
        return false;
    }
    uint64_t hdr_len = 0;
    for (int i = 0; i < 8; i++) {
        hdr_len |= (uint64_t) hl[i] << (8 * i);
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return false;
    }
    const long fsz = ftell(f);
    if (hdr_len == 0 || hdr_len > (uint64_t) 64 * 1024 * 1024 || fsz <= 8 || hdr_len > (uint64_t) (fsz - 8)) {
        fclose(f);
        return false;
    }
    std::vector<char> hdr((size_t) hdr_len);
    if (fseek(f, 8, SEEK_SET) != 0 || fread(hdr.data(), 1, hdr.size(), f) != hdr.size()) {
        fclose(f);
        return false;
    }
    fclose(f);

    yyjson_doc * doc = yyjson_read(hdr.data(), hdr.size(), 0);
    if (!doc) {
        return false;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    yyjson_val * md   = (root && yyjson_is_obj(root)) ? yyjson_obj_get(root, "__metadata__") : NULL;
    if (md && yyjson_is_obj(md)) {
        yyjson_obj_iter it;
        yyjson_obj_iter_init(md, &it);
        yyjson_val * k;
        while ((k = yyjson_obj_iter_next(&it)) != NULL) {
            yyjson_val * v = yyjson_obj_iter_get_val(k);
            if (yyjson_is_str(k) && v && yyjson_is_str(v)) {
                (*out)[yyjson_get_str(k)] = yyjson_get_str(v);
            }
        }
    }
    yyjson_doc_free(doc);
    return true;  // true with an empty map when there is no __metadata__
}

static inline std::string lm_md_get(const std::map<std::string, std::string> & md, const char * key) {
    auto it = md.find(key);
    return it == md.end() ? std::string() : it->second;
}

// Apply the dataset's custom tag exactly the way preprocess-run.h:192-204 does.
// The training caption MUST carry the trigger word — §2.3's own example
// ("electriccallboy, This track is …") and Side-Step's abba lm_codes.jsonl
// ("abba, This track is …") both do.
static std::string lm_apply_tag(const std::string & text, const std::string & tag, const std::string & position) {
    if (tag.empty()) {
        return text;
    }
    const std::string pos = position.empty() ? std::string("prepend") : position;
    if (pos == "prepend") {
        return text.empty() ? tag : tag + ", " + text;
    }
    if (pos == "append") {
        return text.empty() ? tag : text + ", " + tag;
    }
    return text;  // "replace": tag is not applied here (verbatim Side-Step behaviour)
}

// ─── VRAM probes ────────────────────────────────────────────────────────────

static inline void lm_vram_query(ggml_backend_t backend, size_t * free_b, size_t * total_b) {
    *free_b  = 0;
    *total_b = 0;
    ggml_backend_dev_t dev = backend ? ggml_backend_get_device(backend) : nullptr;
    if (!dev) {
        return;
    }
    ggml_backend_dev_memory(dev, free_b, total_b);
}

static inline size_t lm_vram_used_mb(ggml_backend_t backend) {
    size_t f = 0, t = 0;
    lm_vram_query(backend, &f, &t);
    return t == 0 ? 0 : (t - f) / (1024 * 1024);
}

static inline size_t lm_vram_free_mb(ggml_backend_t backend) {
    size_t f = 0, t = 0;
    lm_vram_query(backend, &f, &t);
    return f / (1024 * 1024);
}

static inline size_t lm_vram_total_mb(ggml_backend_t backend) {
    size_t f = 0, t = 0;
    lm_vram_query(backend, &f, &t);
    return t / (1024 * 1024);
}

// ─── model-size label ───────────────────────────────────────────────────────

// Label written into adapter_config / lm_train_log and used for the 4B refusal.
static std::string lm_size_label_from_config(const Qwen3LMConfig & c) {
    if (c.hidden_size <= 1024 && c.n_layers <= 28) {
        return "0.6B";
    }
    if (c.hidden_size <= 2048 && c.n_layers <= 28) {
        return "1.7B";
    }
    if (c.hidden_size >= 2560 || c.n_layers >= 36) {
        return "4B";
    }
    return "custom";
}

// ─── low-VRAM defaults (4B plan §2.1 / D6) ──────────────────────────────────

// `g` is legal iff it divides n_heads AND every block still owns >= 1 kv head
// (g * Nkv must be a whole multiple of Nh, so the GQA broadcast factor Nh/Nkv
// survives the split). g == 0 means "off".
static inline bool lm_ckpt_head_block_ok(const Qwen3LMConfig & c, int g) {
    if (g == 0) {
        return true;
    }
    if (g < 0 || g >= c.n_heads || c.n_heads % g != 0) {
        return false;
    }
    return ((int64_t) g * (int64_t) c.n_kv_heads) % (int64_t) c.n_heads == 0;
}

// Default heads-per-attention-block when --low-vram is active:
//   n_heads <= 16 -> 0 (off), else 8.
static inline int lm_ckpt_default_head_block(const Qwen3LMConfig & c) {
    if (c.n_heads <= 16) {
        return 0;
    }
    return lm_ckpt_head_block_ok(c, 8) ? 8 : 0;
}

// ─── misc ───────────────────────────────────────────────────────────────────

// Write `text` to `path` via tmp + stw_replace_file (never leaves a partial file).
static bool lm_write_atomic(const std::string & path, const std::string & text) {
    return pm_write_atomic(path, text);
}

static inline std::string lm_join(const std::string & dir, const std::string & leaf) {
    if (dir.empty()) {
        return leaf;
    }
    const char last = dir[dir.size() - 1];
    return (last == '/' || last == '\\') ? dir + leaf : dir + "/" + leaf;
}

// "%.1f" with a '.' decimal separator regardless of locale (milestone dir names).
static inline std::string lm_fmt1(double v) {
    char b[64];
    snprintf(b, sizeof(b), "%.1f", v);
    for (size_t i = 0; b[i]; i++) {
        if (b[i] == ',') {
            b[i] = '.';
        }
    }
    return std::string(b);
}
