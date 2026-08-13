#pragma once
// minimax/mm3-sample.h — the MiniMax-Music3 top-k multinomial sampler.
//
// HOT-Step file (does not exist upstream). Included by minimax/mm3-depth-graph.h
// and minimax/mm3-ar-loop.h, both of which reach tools/hot-step-server.cpp only
// through the single minimax/mm3-server.h hook include.
//
// SCOPE: one function. Both AR-stage samplers — the global LM's semantic code and
// the depth decoder's seven acoustic codes — call the SAME reference helper
// (`_sample_top_k` in the diffusers `encoders.py` at commit dafe3733), so it lives
// once, here, instead of twice in two graph files.
//
// ── The reference, line by line ──────────────────────────────────────────────
//
//   values    = nan_to_num(logits.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
//   top_k     = min(50, values.shape[-1])
//   threshold = topk(values, top_k).values[..., -1]        # the k-th LARGEST
//   values    = values.masked_fill(values < threshold, -inf)
//   probs     = nan_to_num(softmax(values), nan=0.0)
//   probs     = probs / probs.sum().clamp_min(1e-12)
//   return multinomial(probs, 1, generator)
//
// Three details that are easy to lose:
//
// 1. THERE IS NO TEMPERATURE. The softmax is over the raw (guided) logits. The
//    checkpoint's recipe exposes exactly two AR sampling knobs — cfg 1.5 and
//    top-k 50 — and both are fixed metadata (`mm3.ar.*`), not user parameters.
//
// 2. `-inf` BECOMES `-1e9` BEFORE THE TOP-K, NOT AFTER. That matters only when
//    fewer than k entries are finite: then the k-th largest is -1e9, nothing is
//    filtered, and the -1e9 entries still contribute exp(-1e9 - max) = 0. So the
//    result is the same as filtering them — but reproducing the order keeps the
//    degenerate case from diverging.
//
// 3. THE COMPARISON IS STRICTLY LESS. Values EQUAL to the threshold survive, so a
//    tie at the boundary keeps more than k candidates. This is observable in the
//    fixtures: `lm_i*_guided_logits` has 50..55 finite entries, not exactly 50.
//
// ── What cannot be reproduced, and what we do instead ────────────────────────
//
// `torch.multinomial` with a torch Generator is not reproducible outside torch —
// not the Mersenne draw, not the CDF walk order. So a seeded C++ run does NOT
// reproduce the reference's code sequence, and never can. Parity is therefore
// validated on LOGITS with the sampler bypassed (`forced_*`), exactly as the depth
// decoder increment did. Here the draw is a plain inverse-CDF walk in ascending
// index order over a std::mt19937_64 — deterministic for a given seed, which is
// what a user-facing `seed` parameter actually needs to mean.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <vector>

// Sample one index from `logits[0, n)` under the reference recipe above.
//
//   scratch   optional caller-owned buffer, reused across calls to keep the AR
//             loop allocation-free; may be null.
//   returns   an index into `logits`, always in [0, n). Never negative: a fully
//             degenerate distribution (all zero mass) falls back to the argmax.
static int64_t mm3_sample_top_k(const float * logits, int64_t n, int top_k, std::mt19937_64 & rng,
                                std::vector<float> * scratch = nullptr) {
    if (n <= 0) {
        return 0;
    }
    std::vector<float>   local;
    std::vector<float> & v = scratch ? *scratch : local;
    v.resize((size_t) n);

    // nan_to_num, and the argmax fallback in the same sweep.
    int64_t arg_best = 0;
    float   val_best = -INFINITY;
    for (int64_t i = 0; i < n; i++) {
        float x = logits[i];
        if (std::isnan(x)) {
            x = -1e9f;
        } else if (std::isinf(x)) {
            x = x > 0.0f ? 1e9f : -1e9f;
        }
        v[(size_t) i] = x;
        if (x > val_best) {
            val_best = x;
            arg_best = i;
        }
    }

    int64_t k = top_k > 0 ? (int64_t) top_k : n;
    if (k > n) {
        k = n;
    }

    // The k-th largest. nth_element over a copy: the sampler runs 25 times a
    // second over 16385 candidates, so an O(n) selection matters and a full sort
    // does not pay for itself.
    float threshold = -INFINITY;
    if (k < n) {
        std::vector<float> sel(v);
        std::nth_element(sel.begin(), sel.begin() + (size_t) (k - 1), sel.end(), std::greater<float>());
        threshold = sel[(size_t) (k - 1)];
    }

    // softmax over the survivors, shifted by their max for stability.
    float max_v = -INFINITY;
    for (int64_t i = 0; i < n; i++) {
        if (v[(size_t) i] >= threshold && v[(size_t) i] > max_v) {
            max_v = v[(size_t) i];
        }
    }
    if (!std::isfinite(max_v)) {
        return arg_best;
    }

    double sum = 0.0;
    for (int64_t i = 0; i < n; i++) {
        if (v[(size_t) i] >= threshold) {
            const double p = std::exp((double) v[(size_t) i] - (double) max_v);
            sum += p;
            v[(size_t) i] = (float) p;
        } else {
            v[(size_t) i] = 0.0f;
        }
    }
    if (!(sum > 0.0)) {
        return arg_best;
    }

    // Inverse-CDF walk in ascending index order.
    const double u   = std::uniform_real_distribution<double>(0.0, 1.0)(rng) * sum;
    double       acc = 0.0;
    for (int64_t i = 0; i < n; i++) {
        acc += (double) v[(size_t) i];
        if (acc > u) {
            return i;
        }
    }
    return arg_best;
}
