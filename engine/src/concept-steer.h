#pragma once
//
// concept-steer.h — CAA activation steering ("Concept Studio")
//
// Implements the TADA method (Staniszewski et al., arXiv 2602.11910) natively:
// a concept direction is added to a layer's output activations,
//
//     h'_l = h_l + alpha * v_c[l, t]
//
// where v_c is a per-(layer, timestep) vector extracted by contrastive activation
// addition — the mean difference of frame-averaged activations over N prompt pairs.
//
// Two hook targets, four model shapes:
//
//   target "dit"  cross-attention output, pre-residual   std 24L/2048, XL 32L/2560
//   target "lm"   decoder layer output, post-MLP         1.7B 28L/2048, 4B 36L/2560
//
// Unlike adapters, steering vectors are NOT weights: they never enter the ModelKey,
// never trigger a merge or reload, and alpha is free to change between generations.
// The DiT graph is rebuilt per sampling run (hot-step-sampler.h), so topology may
// depend on whether steering is active — but NOT on which layers are steered (see
// ConceptSteerRuntime::resolve; unsteered layers get an all-zero vector instead).
//
// Design notes + provenance: docs/plans/caa-activation-steering.md
//
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gguf-weights.h"
#include "hot-step-params.h"

// ---------------------------------------------------------------------------
// One loaded concept file.
// ---------------------------------------------------------------------------
struct ConceptSteer {
    std::string name;    // "aggressive"
    std::string target;  // "dit" | "lm"
    std::string method;  // "caa" | "austeer"
    std::string base_name;

    int n_layers = 0;
    int hidden   = 0;
    int n_steps  = 0;  // vectors sampled at n_steps points along the schedule

    float alpha = 1.0f;  // user strength for this generation

    // t values the vectors were extracted at, DESCENDING (1 = noise -> 0 = clean),
    // matching flow-matching convention. Size n_steps. Empty => uniform over [1,0].
    std::vector<float> t_schedule;

    // [hidden * n_steps * n_layers], layer-major: vec[(l * n_steps + s) * hidden + h]
    std::vector<float> vec;

    // Active layer indices. Empty => every layer that has a non-zero vector.
    std::vector<int> layers;

    bool layer_active(int l) const {
        if (layers.empty()) {
            return true;
        }
        return std::find(layers.begin(), layers.end(), l) != layers.end();
    }

    // Interpolate this concept's vector for layer `l` at flow-matching time `t`,
    // accumulating alpha * v into `out` (size `hidden`).
    //
    // t_schedule is stored descending, so we search for the bracketing pair and
    // lerp. Storing against t rather than step index is deliberate: extraction and
    // playback run different step counts on shifted schedules that are wildly
    // nonuniform in t. Indexing by step number reproduces the adapter gain-curve
    // bug where a shift-3 schedule starved a late-window adapter to 3 tail steps.
    void accumulate(int l, float t, float * out) const {
        if (n_steps <= 0 || hidden <= 0 || !layer_active(l)) {
            return;
        }
        const float * base = vec.data() + (size_t) l * n_steps * hidden;

        if (n_steps == 1) {
            for (int h = 0; h < hidden; h++) {
                out[h] += alpha * base[h];
            }
            return;
        }

        // Locate t in the descending schedule.
        int   i0 = 0, i1 = 1;
        float w  = 0.0f;
        if (t_schedule.empty()) {
            // Uniform descending [1 .. 0] over n_steps points.
            float pos = (1.0f - t) * (float) (n_steps - 1);
            pos       = std::min(std::max(pos, 0.0f), (float) (n_steps - 1));
            i0        = (int) pos;
            i1        = std::min(i0 + 1, n_steps - 1);
            w         = pos - (float) i0;
        } else {
            i0 = n_steps - 2;
            for (int i = 0; i + 1 < n_steps; i++) {
                // descending: t_schedule[i] >= t >= t_schedule[i+1]
                if (t >= t_schedule[i + 1]) {
                    i0 = i;
                    break;
                }
            }
            i1        = i0 + 1;
            float den = t_schedule[i0] - t_schedule[i1];
            w         = (den != 0.0f) ? (t_schedule[i0] - t) / den : 0.0f;
            w         = std::min(std::max(w, 0.0f), 1.0f);
        }

        const float * v0 = base + (size_t) i0 * hidden;
        const float * v1 = base + (size_t) i1 * hidden;
        for (int h = 0; h < hidden; h++) {
            out[h] += alpha * (v0[h] + w * (v1[h] - v0[h]));
        }
    }
};

// ---------------------------------------------------------------------------
// All concepts active for one target on one generation.
//
// Several concepts stack by simple summation into a single per-layer vector, so
// the graph carries ONE input tensor per layer regardless of stack depth — the
// same trick the runtime adapter path uses for delta-sum.
// ---------------------------------------------------------------------------
struct ConceptSteerRuntime {
    std::vector<ConceptSteer> concepts;
    int                       n_layers = 0;
    int                       hidden   = 0;

    // Host scratch, [n_layers * hidden], refilled at every model evaluation.
    std::vector<float> scratch;

    bool active() const { return !concepts.empty() && n_layers > 0 && hidden > 0; }

    void reset() {
        concepts.clear();
        n_layers = 0;
        hidden   = 0;
        scratch.clear();
    }

    // Fill `scratch` with the summed alpha*v for every layer at time `t`.
    // Layers no concept steers are left at zero — the graph still adds them, which
    // keeps topology independent of the layer selection.
    void resolve(float t) {
        scratch.assign((size_t) n_layers * hidden, 0.0f);
        for (const ConceptSteer & c : concepts) {
            for (int l = 0; l < n_layers; l++) {
                c.accumulate(l, t, scratch.data() + (size_t) l * hidden);
            }
        }
    }

    const float * layer_ptr(int l) const { return scratch.data() + (size_t) l * hidden; }
};

// Extra graph-node/tensor budget steering needs: one input tensor + one add node
// per layer. The graph_cap formula is duplicated between dit-graph.h (graph
// build) and hot-step-sampler.h (ctx sizing) and the two MUST agree — a
// mismatch overflows the graph exactly like the per-section adapter node-budget
// bug (e09d6a3). Both call this.
static inline size_t concept_steer_graph_nodes(const ConceptSteerRuntime & s) {
    return s.active() ? (size_t) s.n_layers * 2 : 0;
}

// ---------------------------------------------------------------------------
// Activation tap — the extraction (Phase B) side.
//
// CAA needs mean_frames(h_l) at every (layer, diffusion step). Reading the raw
// [H, S, N] cross-attn output back to the host would be ~10 MB per layer per
// step (XL, S~1000) — around 10 GB per run — so the mean is taken ON THE GPU
// and only [H] floats per (layer, step) come back.
//
// Single-worker-thread global, exactly like g_hotstep_params.
// ---------------------------------------------------------------------------
struct ConceptTapSink {
    bool recording = false;  // set by the extraction driver around a run
    int  n_layers  = 0;
    int  hidden    = 0;

    // Captured frame-means, appended in evaluation order:
    //   steps[k] holds [n_layers * hidden] for the k-th recorded evaluation
    std::vector<std::vector<float>> steps;
    std::vector<float>              t_values;  // flow-matching t per recorded evaluation

    // Arm before a run. Dimensions are filled in by the sampler on the first
    // push, so the extraction driver never needs to reach into the model.
    void arm() {
        recording = true;
        n_layers  = 0;
        hidden    = 0;
        steps.clear();
        t_values.clear();
        act_norms.clear();
    }
    void disarm() { recording = false; }

    // Per-layer L2 norm of mean_frames(h) for each recorded evaluation, i.e. the
    // magnitude of what is already there. Used to express alpha as a FRACTION of
    // local activation magnitude rather than an absolute number: |v| spans 4.85
    // (L01) to 132 (L31) on the XL DiT, so a raw alpha means something different
    // at every layer and for every concept.
    std::vector<std::vector<float>> act_norms;  // [eval][n_layers]

    // Only the conditional (positive-prompt) pass is meaningful for CAA; the
    // sampler records exactly one evaluation per computed diffusion step.
    void push(int L, int H, float t, const float * data) {
        n_layers = L;
        hidden   = H;
        t_values.push_back(t);
        steps.emplace_back(data, data + (size_t) L * H);

        std::vector<float> norms((size_t) L, 0.0f);
        for (int l = 0; l < L; l++) {
            double n2 = 0.0;
            for (int h = 0; h < H; h++) {
                const double x = data[(size_t) l * H + h];
                n2 += x * x;
            }
            norms[(size_t) l] = (float) std::sqrt(n2);
        }
        act_norms.push_back(std::move(norms));
    }
};

inline ConceptTapSink g_concept_tap;

// transpose + cont + mean = 3 nodes per layer.
static inline size_t concept_tap_graph_nodes(int n_layers) {
    return g_concept_tap.recording ? (size_t) n_layers * 3 : 0;
}

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

// Load a concept GGUF. `want_target` is "dit" or "lm"; `exp_layers`/`exp_hidden`
// come from the loaded model and are checked hard.
//
// The compat check is NOT optional. Our four targets pair up on hidden size
// (std DiT and LM-1.7B are both 2048; XL DiT and LM-4B are both 2560), so a
// dimension match alone proves nothing about which network a concept came from.
// Loading a DiT concept into the LM would run silently and produce garbage.
static bool concept_steer_load(const char * path,
                               const char * want_target,
                               int          exp_layers,
                               int          exp_hidden,
                               float        alpha,
                               ConceptSteer * out) {
    GGUFModel gf;
    if (!gf_load(&gf, path)) {
        fprintf(stderr, "[Concept] ERROR: cannot open %s\n", path);
        return false;
    }

    out->name      = gf_get_str(gf, "concept.name");
    out->target    = gf_get_str(gf, "concept.target");
    out->method    = gf_get_str(gf, "concept.method");
    out->base_name = gf_get_str(gf, "concept.base_name");
    out->n_layers  = (int) gf_get_u32(gf, "concept.n_layers");
    out->hidden    = (int) gf_get_u32(gf, "concept.hidden");
    out->alpha     = alpha;

    if (out->target != want_target) {
        fprintf(stderr, "[Concept] ERROR: %s targets '%s', expected '%s' — refusing\n",
                path, out->target.c_str(), want_target);
        gf_close(&gf);
        return false;
    }
    if (out->n_layers != exp_layers || out->hidden != exp_hidden) {
        fprintf(stderr,
                "[Concept] ERROR: %s is %dL/%dH but the loaded %s is %dL/%dH — "
                "concepts are scale-specific, re-extract for this model\n",
                path, out->n_layers, out->hidden, want_target, exp_layers, exp_hidden);
        gf_close(&gf);
        return false;
    }

    int64_t vi = gguf_find_tensor(gf.gguf, "vec");
    if (vi < 0) {
        fprintf(stderr, "[Concept] ERROR: %s has no 'vec' tensor\n", path);
        gf_close(&gf);
        return false;
    }

    // vec is [hidden, n_steps, n_layers] f32.
    struct ggml_tensor * meta = ggml_get_tensor(gf.meta, "vec");
    if (!meta || meta->type != GGML_TYPE_F32) {
        fprintf(stderr, "[Concept] ERROR: %s 'vec' must be F32\n", path);
        gf_close(&gf);
        return false;
    }
    if ((int) meta->ne[0] != out->hidden || (int) meta->ne[2] != out->n_layers) {
        fprintf(stderr, "[Concept] ERROR: %s 'vec' shape [%lld,%lld,%lld] disagrees with header %dH/%dL\n",
                path, (long long) meta->ne[0], (long long) meta->ne[1], (long long) meta->ne[2],
                out->hidden, out->n_layers);
        gf_close(&gf);
        return false;
    }
    out->n_steps = (int) meta->ne[1];

    size_t       n     = (size_t) out->hidden * out->n_steps * out->n_layers;
    size_t       off   = gguf_get_tensor_offset(gf.gguf, vi);
    const float * data = (const float *) (gf.mapping + gf.data_offset + off);
    out->vec.assign(data, data + n);

    // Optional t schedule.
    int64_t ti = gguf_find_key(gf.gguf, "concept.t_schedule");
    if (ti >= 0 && gguf_get_arr_type(gf.gguf, ti) == GGUF_TYPE_FLOAT32) {
        size_t        ns = gguf_get_arr_n(gf.gguf, ti);
        const float * ts = (const float *) gguf_get_arr_data(gf.gguf, ti);
        if ((int) ns == out->n_steps) {
            out->t_schedule.assign(ts, ts + ns);
        } else {
            fprintf(stderr, "[Concept] WARNING: %s t_schedule has %zu entries but vec has %d steps — ignoring\n",
                    path, ns, out->n_steps);
        }
    }

    fprintf(stderr, "[Concept] Loaded '%s' (%s, %s): %dL x %d steps x %dH, alpha=%.2f%s\n",
            out->name.empty() ? path : out->name.c_str(), out->target.c_str(),
            out->method.empty() ? "caa" : out->method.c_str(), out->n_layers, out->n_steps,
            out->hidden, alpha, out->t_schedule.empty() ? " (uniform t)" : "");

    gf_close(&gf);
    return true;
}

// Resolve this generation's sideband concept requests for one target into `rt`.
//
// Called by the sampler (DiT) / LM forward BEFORE the graph is built — not by the
// model loader. Concepts are per-request activations, so binding them at load
// time would wrongly imply they belong in the ModelKey.
//
// A concept that fails to load is SKIPPED with a loud error, never silently
// treated as loaded. Steering stays off entirely if nothing loads, so the user
// hears the unsteered base rather than a half-applied stack.
static void concept_steer_prepare(ConceptSteerRuntime * rt,
                                  const char *          target,
                                  int                   n_layers,
                                  int                   hidden) {
    rt->reset();
    rt->n_layers = n_layers;
    rt->hidden   = hidden;

    for (const HotStepConcept & req : g_hotstep_params.concepts) {
        if (req.target != target) {
            continue;
        }
        if (req.alpha == 0.0f) {
            fprintf(stderr, "[Concept] '%s' alpha=0 — skipping\n", req.name.c_str());
            continue;
        }
        ConceptSteer cs;
        if (!concept_steer_load(req.path.c_str(), target, n_layers, hidden, req.alpha, &cs)) {
            continue;  // error already printed
        }
        if (!req.layers.empty()) {
            cs.layers = req.layers;
        }
        if (!req.name.empty()) {
            cs.name = req.name;
        }
        rt->concepts.push_back(std::move(cs));
    }

    if (rt->concepts.empty()) {
        rt->n_layers = 0;
        rt->hidden   = 0;
        return;
    }

    fprintf(stderr, "[Concept] Steering %s: %zu concept(s) over %d layers\n", target, rt->concepts.size(), n_layers);
    for (const ConceptSteer & c : rt->concepts) {
        if (c.layers.empty()) {
            fprintf(stderr, "[Concept]   %s alpha=%+.2f (all layers)\n", c.name.c_str(), c.alpha);
        } else {
            std::string ls;
            for (size_t i = 0; i < c.layers.size(); i++) {
                ls += (i ? "," : "") + std::to_string(c.layers[i]);
            }
            fprintf(stderr, "[Concept]   %s alpha=%+.2f layers=[%s]\n", c.name.c_str(), c.alpha, ls.c_str());
        }
    }
}
