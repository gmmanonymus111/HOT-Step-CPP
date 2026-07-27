#pragma once
// spike-common.h — Phase-0 training spike shared helpers.
//
// EVIDENCE CODE, NOT PRODUCT CODE. Everything here exists to produce
// trustworthy measurements about whether GGML can train HOT-Step graphs.

#include "backend.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-opt.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ─── deterministic RNG ──────────────────────────────────────────────────────

struct SpikeRng {
    uint64_t s;
};

static inline uint64_t srng_next(SpikeRng * r) {
    uint64_t x = r->s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    r->s = x;
    return x;
}

static inline float srng_uniform(SpikeRng * r) {  // (0,1)
    return (float) ((srng_next(r) >> 11) + 1) / (float) (1ull << 53);
}

static inline float srng_normal(SpikeRng * r) {
    float u1 = srng_uniform(r);
    float u2 = srng_uniform(r);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
}

static inline void srng_fill_normal(SpikeRng * r, std::vector<float> & v, float sigma) {
    for (size_t i = 0; i < v.size(); i++) {
        v[i] = srng_normal(r) * sigma;
    }
}

// ─── tensor helpers ─────────────────────────────────────────────────────────

static inline void spike_set(ggml_tensor * t, const std::vector<float> & v) {
    GGML_ASSERT((int64_t) v.size() == ggml_nelements(t));
    ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
}

static inline std::vector<float> spike_get(ggml_tensor * t) {
    std::vector<float> v((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    return v;
}

static inline double spike_l2(const std::vector<float> & v) {
    double s = 0;
    for (float x : v) {
        s += (double) x * (double) x;
    }
    return sqrt(s);
}

static inline bool spike_all_finite(const std::vector<float> & v) {
    for (float x : v) {
        if (!std::isfinite(x)) {
            return false;
        }
    }
    return true;
}

// ─── VRAM probe (CUDA-only; falls back to 0) ────────────────────────────────

static inline size_t spike_vram_used_mb(ggml_backend_t backend) {
    ggml_backend_dev_t dev = backend ? ggml_backend_get_device(backend) : nullptr;
    if (!dev) {
        return 0;
    }
    size_t f = 0, t = 0;
    ggml_backend_dev_memory(dev, &f, &t);
    if (t == 0) {
        return 0;
    }
    return (t - f) / (1024 * 1024);
}
