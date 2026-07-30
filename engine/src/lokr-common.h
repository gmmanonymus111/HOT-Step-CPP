#pragma once
// lokr-common.h — the LyCORIS LoKr shape rules, shared by every LoKr consumer.
//
// These two functions decide the ENTIRE layout of a LoKr adapter: how a base
// dimension splits into kron factors, and whether w2 stays monolithic or is
// itself factorized. They therefore have to be identical everywhere, because a
// disagreement between the trainer and a loader is not a crash — it is an
// adapter that loads and quietly computes something else.
//
// Extracted from train/dit-adapter-lokr.h (2026-07-30) when the LM trainer
// needed the same rules. Pure functions, no ggml, no model types.

#include <algorithm>
#include <cstdint>

// LyCORIS `factorization(dimension, factor)`: split `dimension` into m * n with
// m <= n, preferring the largest m that does not exceed `factor`. factor <= 0
// means "no cap" (LyCORIS's -1), which lands on the most balanced split.
static inline void lokr_factorization(int64_t dimension, int factor, int64_t * out_m, int64_t * out_n) {
    if (factor > 0 && (dimension % (int64_t) factor) == 0) {
        int64_t m = (int64_t) factor, n = dimension / (int64_t) factor;
        if (m > n) {
            std::swap(m, n);
        }
        *out_m = m;
        *out_n = n;
        return;
    }
    const int64_t cap    = (factor < 0) ? dimension : (int64_t) factor;
    int64_t       m      = 1, n = dimension;
    const int64_t length = m + n;
    while (m < n) {
        int64_t new_m = m + 1;
        while (dimension % new_m != 0) {
            new_m++;
        }
        const int64_t new_n = dimension / new_m;
        if (new_m + new_n > length || new_m > cap) {
            break;
        }
        m = new_m;
        n = new_n;
    }
    if (m > n) {
        std::swap(m, n);
    }
    *out_m = m;
    *out_n = n;
}

// LyCORIS keeps w2 monolithic unless the factorized form is strictly cheaper.
static inline bool lokr_w2_mono(int dim, int64_t out_k, int64_t in_n) {
    return !((double) dim < (double) std::max(out_k, in_n) / 2.0);
}
