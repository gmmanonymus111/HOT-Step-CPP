#pragma once
//
// concept-extract.h — CAA / AUSteer concept extraction (TADA arXiv 2602.11910)
//
// Builds a concept direction from paired forward passes. No gradients, no
// dataset, no optimizer: run the sampler twice per pair (positive prompt vs
// negative prompt, everything else identical), tap the frame-averaged
// cross-attention output at every (layer, step), and accumulate
//
//     CAA:     v[l,s] = (1/N) * sum_i ( h+[l,s] - h-[l,s] )
//     AUSteer: beta[l,s,d] = (1/N) * sum_i sign( h+[l,s,d] - h-[l,s,d] )
//
// beta is the sign-agreement score in [-1,1]: how consistently a dimension moves
// the same way across pairs. AUSteer keeps the global top-s dimensions by |beta|
// and zeroes the rest. It costs one extra accumulator to collect, so it is
// gathered from day one even while the engine only consumes `vec`.
//
// The (step -> t) axis comes from the recorded evaluation times, NOT from step
// indices — see docs/plans/caa-activation-steering.md.
//
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "concept-steer.h"
#include "ggml.h"
#include "gguf.h"

// ---------------------------------------------------------------------------
// Running accumulator over pairs.
// ---------------------------------------------------------------------------
struct ConceptAccum {
    int n_layers = 0;
    int hidden   = 0;
    int n_steps  = 0;  // evaluations recorded per run (must match across pairs)
    int n_pairs  = 0;

    std::vector<double> sum_diff;  // [n_layers * n_steps * hidden]
    std::vector<double> sum_sign;  // same shape
    std::vector<float>  t_values;  // [n_steps], from the first accepted pair

    // Layout helper: the on-disk tensor is [hidden, n_steps, n_layers], i.e.
    // index = (l * n_steps + s) * hidden + d. The tap sink hands us one
    // [n_layers * hidden] block per step, so the transpose happens here.
    size_t idx(int l, int s, int d) const {
        return ((size_t) l * n_steps + s) * hidden + d;
    }

    void init(int L, int H, int S, const std::vector<float> & ts) {
        n_layers = L;
        hidden   = H;
        n_steps  = S;
        n_pairs  = 0;
        sum_diff.assign((size_t) L * S * H, 0.0);
        sum_sign.assign((size_t) L * S * H, 0.0);
        t_values = ts;
    }

    // Fold one (positive, negative) pair. Both sinks must have recorded the same
    // number of evaluations; a mismatch means the two runs diverged (different
    // step caching / CFG cutoff behaviour) and the pair is REJECTED rather than
    // silently truncated — a misaligned pair would inject noise into every layer.
    bool add_pair(const ConceptTapSink & pos, const ConceptTapSink & neg) {
        if ((int) pos.steps.size() != n_steps || (int) neg.steps.size() != n_steps) {
            fprintf(stderr,
                    "[Concept] WARNING: pair rejected — recorded %zu/%zu evaluations, expected %d. "
                    "Disable step caching / CFG cutoff for extraction.\n",
                    pos.steps.size(), neg.steps.size(), n_steps);
            return false;
        }
        for (int s = 0; s < n_steps; s++) {
            const float * p = pos.steps[s].data();
            const float * q = neg.steps[s].data();
            for (int l = 0; l < n_layers; l++) {
                const size_t src = (size_t) l * hidden;
                for (int d = 0; d < hidden; d++) {
                    const double diff = (double) p[src + d] - (double) q[src + d];
                    const size_t o    = idx(l, s, d);
                    sum_diff[o] += diff;
                    sum_sign[o] += (diff > 0.0) ? 1.0 : (diff < 0.0 ? -1.0 : 0.0);
                }
            }
        }
        n_pairs++;
        return true;
    }

    // Mean-difference vector, ready to write.
    std::vector<float> mean_vec() const {
        std::vector<float> v(sum_diff.size(), 0.0f);
        if (n_pairs <= 0) return v;
        const double inv = 1.0 / (double) n_pairs;
        for (size_t i = 0; i < sum_diff.size(); i++) v[i] = (float) (sum_diff[i] * inv);
        return v;
    }

    std::vector<float> mean_beta() const {
        std::vector<float> b(sum_sign.size(), 0.0f);
        if (n_pairs <= 0) return b;
        const double inv = 1.0 / (double) n_pairs;
        for (size_t i = 0; i < sum_sign.size(); i++) b[i] = (float) (sum_sign[i] * inv);
        return b;
    }
};

// ---------------------------------------------------------------------------
// Provenance recorded into the concept file.
// ---------------------------------------------------------------------------
struct ConceptMeta {
    std::string name;
    std::string target = "dit";
    std::string method = "caa";
    std::string base_name;
    std::string pos_prompt;
    std::string neg_prompt;
    std::string target_class;
};

// Write a concept GGUF. Layout and keys are documented in concept-steer.h /
// the design doc; concept_steer_load is the reader and the two must agree.
static bool concept_write_gguf(const char * path, const ConceptMeta & meta, const ConceptAccum & acc) {
    if (acc.n_pairs <= 0) {
        fprintf(stderr, "[Concept] ERROR: refusing to write %s — no pairs accumulated\n", path);
        return false;
    }

    struct gguf_context * gf = gguf_init_empty();
    if (!gf) {
        fprintf(stderr, "[Concept] ERROR: gguf_init_empty failed\n");
        return false;
    }

    gguf_set_val_str(gf, "general.architecture", "concept-steer");
    gguf_set_val_str(gf, "concept.name", meta.name.c_str());
    gguf_set_val_str(gf, "concept.target", meta.target.c_str());
    gguf_set_val_str(gf, "concept.method", meta.method.c_str());
    gguf_set_val_str(gf, "concept.base_name", meta.base_name.c_str());
    gguf_set_val_str(gf, "concept.pos_prompt", meta.pos_prompt.c_str());
    gguf_set_val_str(gf, "concept.neg_prompt", meta.neg_prompt.c_str());
    gguf_set_val_str(gf, "concept.target_class", meta.target_class.c_str());
    gguf_set_val_u32(gf, "concept.n_layers", (uint32_t) acc.n_layers);
    gguf_set_val_u32(gf, "concept.hidden", (uint32_t) acc.hidden);
    gguf_set_val_u32(gf, "concept.n_pairs", (uint32_t) acc.n_pairs);
    if (!acc.t_values.empty()) {
        gguf_set_arr_data(gf, "concept.t_schedule", GGUF_TYPE_FLOAT32, acc.t_values.data(),
                          acc.t_values.size());
    }

    // Tensor data lives in a throwaway ggml context; gguf copies it on write.
    const size_t n_el = (size_t) acc.hidden * acc.n_steps * acc.n_layers;
    struct ggml_init_params ip = {
        /*.mem_size   =*/2 * (n_el * sizeof(float) + ggml_tensor_overhead()) + (1u << 20),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/false,
    };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        fprintf(stderr, "[Concept] ERROR: ggml ctx alloc failed for %zu elements\n", n_el);
        gguf_free(gf);
        return false;
    }

    std::vector<float> vec  = acc.mean_vec();
    std::vector<float> beta = acc.mean_beta();

    struct ggml_tensor * tv = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, acc.hidden, acc.n_steps, acc.n_layers);
    ggml_set_name(tv, "vec");
    memcpy(tv->data, vec.data(), n_el * sizeof(float));
    gguf_add_tensor(gf, tv);

    struct ggml_tensor * tb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, acc.hidden, acc.n_steps, acc.n_layers);
    ggml_set_name(tb, "beta");
    memcpy(tb->data, beta.data(), n_el * sizeof(float));
    gguf_add_tensor(gf, tb);

    const bool ok = gguf_write_to_file(gf, path, /*only_meta=*/false);
    ggml_free(ctx);
    gguf_free(gf);

    if (!ok) {
        fprintf(stderr, "[Concept] ERROR: failed to write %s\n", path);
        return false;
    }

    // Per-layer localization report.
    //
    // |v| (raw L2 of the mean-difference) is reported for reference but is NOT a
    // localization signal: activation magnitude grows with depth in transformers,
    // so a raw norm largely re-measures depth. Measured on the first real run
    // (2026-07-31), |v| and mean|beta| ranked almost oppositely — L02 was near
    // the bottom on |v| and 2nd from top on mean|beta|.
    //
    // mean|beta| is the honest signal: sign agreement is scale-free, in [0,1],
    // with a noise floor of sqrt(2/(pi*N)) for N pairs. Layers meaningfully above
    // that floor are the ones carrying the concept.
    const double noise_floor = acc.n_pairs > 0 ? std::sqrt(2.0 / (3.14159265358979 * acc.n_pairs)) : 0.0;

    fprintf(stderr, "[Concept] Wrote %s (%d pairs, %dL x %d steps x %dH)\n", path, acc.n_pairs, acc.n_layers,
            acc.n_steps, acc.hidden);
    fprintf(stderr, "[Concept] Per-layer localization (sign-agreement noise floor = %.3f at N=%d):\n",
            noise_floor, acc.n_pairs);
    fprintf(stderr, "[Concept]   layer      |v|   mean|B|   excess\n");

    int    best_l = 0;
    double best_b = -1.0;
    for (int l = 0; l < acc.n_layers; l++) {
        double tot_v = 0.0;
        double tot_b = 0.0;
        for (int s = 0; s < acc.n_steps; s++) {
            double n2 = 0.0;
            for (int d = 0; d < acc.hidden; d++) {
                const size_t o = acc.idx(l, s, d);
                const float  x = vec[o];
                n2 += (double) x * x;
                tot_b += std::fabs((double) beta[o]);
            }
            tot_v += std::sqrt(n2);
        }
        const double mv = tot_v / (double) acc.n_steps;
        const double mb = tot_b / ((double) acc.n_steps * acc.hidden);
        if (mb > best_b) {
            best_b = mb;
            best_l = l;
        }
        fprintf(stderr, "[Concept]   L%02d %9.3f   %7.4f   %+.4f\n", l, mv, mb, mb - noise_floor);
    }
    fprintf(stderr, "[Concept] Most discriminative layer: L%02d (mean|B|=%.4f)\n", best_l, best_b);
    if (best_b < noise_floor * 1.5) {
        fprintf(stderr,
                "[Concept] WARNING: no layer clears 1.5x the noise floor — this concept did not "
                "separate. Try more pairs, or more sharply opposed prompts.\n");
    }
    return true;
}
