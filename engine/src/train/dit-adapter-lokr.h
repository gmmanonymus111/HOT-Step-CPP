#pragma once
// dit-adapter-lokr.h — the LoKR parameterization behind the D22 seam
// (docs/plans/2026-07-28-lokr-dit-training.md §2).
//
// LyCORIS parity, exactly:
//   (out_l, out_k) = factorization(out, factor)   (a, c) in LyCORIS terms
//   (in_m, in_n)   = factorization(in,  factor)   (b, d)
//   torch w1 [out_l, in_m], w2 [out_k, in_n]      dW = kron(w1, w2) * scale
//   dW[l*out_k + k, m*in_n + n] = scale * w1[l,m] * w2[k,n]
//
// ggml stores the transpose of the torch row-major view, so w1 lives as
// [ne0=in_m, ne1=out_l] and w2 as [ne0=in_n, ne1=out_k] — the same convention
// adapter-merge.h's LoKr path builds its kron from.
//
// dW is NEVER materialized in the training graph (K7): a per-site 6144x2048 F32
// delta is 50 MB that the backward pass would have to retain. apply() contracts
// against x directly (§2.3) and its retained intermediates scale with the
// activations instead.

#include "train/dit-adapter.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// ─── LyCORIS factorization (lycoris/functional/general.py) ──────────────────
//
// The `while (m < n)` loop guard is load-bearing: without it the sweep walks
// past the crossover to m = dimension, n = 1 and then spins forever looking for
// a divisor above `dimension`. The plan's §1 pseudocode drops it; LyCORIS has
// it, and (2048, -1) -> (32, 64) only comes out with it.
static void dit_lokr_factorization(int64_t dimension, int factor, int64_t * out_m, int64_t * out_n) {
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
static inline bool dit_lokr_w2_mono(int dim, int64_t out_k, int64_t in_n) {
    return !((double) dim < (double) std::max(out_k, in_n) / 2.0);
}

// K3: HOT-Step's loaders read only `lokr_w1`, so w1 is always monolithic. This
// reports whether LyCORIS WOULD have factorized it, so the override can warn.
static inline bool dit_lokr_w1_would_factor(int dim, bool decompose_both, int64_t out_l, int64_t in_m) {
    return decompose_both && ((double) dim < (double) std::max(out_l, in_m) / 2.0);
}

// ─── per-site state ─────────────────────────────────────────────────────────

struct DitLokrSite {
    ggml_tensor * w1   = nullptr;  // ggml [in_m, out_l]
    ggml_tensor * w2   = nullptr;  // ggml [in_n, out_k]   monolithic
    ggml_tensor * w2_a = nullptr;  // ggml [dim,  out_k]   factorized
    ggml_tensor * w2_b = nullptr;  // ggml [in_n, dim]
    int64_t       out_l = 0, out_k = 0, in_m = 0, in_n = 0;
    bool          mono  = true;
    float         scale = 1.0f, alpha_eff = 0.0f;
};

// ─── LoKR ───────────────────────────────────────────────────────────────────

struct DitAdapterLoKr final : DitAdapter {
    ggml_context *                                   ctx = nullptr;
    ggml_backend_buffer_t                            buf = nullptr;
    std::vector<std::array<DitLokrSite, DIT_NSITES>> layers;  // indexed by (layer - lo)
    std::vector<ggml_tensor *>                       par;
    int    lo = 0, hi = 0, dim = 0, factor = 0, n_sites = DIT_NSITES_ATTN;
    float  alpha           = 0.0f;
    bool   decompose_both  = true;
    size_t n_params        = 0;

    const std::vector<ggml_tensor *> & params() const override { return par; }
    size_t                             nParams() const override { return n_params; }
    int                                nSites() const override { return n_sites; }
    const char *                       typeName() const override { return "lokr"; }

    void free() override {
        if (buf) {
            ggml_backend_buffer_free(buf);
            buf = nullptr;
        }
        if (ctx) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        layers.clear();
        par.clear();
        n_params = 0;
    }

    const DitLokrSite * site(int layer, int s) const {
        if (layer < lo || layer >= hi || s < 0 || s >= n_sites) {
            return nullptr;
        }
        const DitLokrSite & k = layers[(size_t) (layer - lo)][(size_t) s];
        return (k.w1 && (k.w2 || (k.w2_a && k.w2_b))) ? &k : nullptr;
    }

    bool init(DiTGGML * m, ggml_backend_t backend, int lo_, int hi_, const DitAdapterCfg & cfg,
              std::string * err) override {
        lo             = lo_;
        hi             = hi_;
        dim            = cfg.lokr_dim;
        factor         = cfg.lokr_factor;
        alpha          = cfg.lokr_alpha;
        decompose_both = cfg.lokr_decompose_both;
        n_sites        = cfg.target_mlp ? DIT_NSITES : DIT_NSITES_ATTN;
        layers.assign((size_t) (hi - lo), std::array<DitLokrSite, DIT_NSITES>{});

        const int n_ten = (hi - lo) * n_sites * 3;
        {
            ggml_init_params p = { (size_t) (n_ten + 8) * ggml_tensor_overhead(), nullptr, true };
            ctx                = ggml_init(p);
        }
        if (!ctx) {
            *err = "cannot create the LoKR context";
            return false;
        }

        int w1_forced = 0, alpha_forced = 0, n_mono = 0, n_fact = 0;
        for (int l = lo; l < hi; l++) {
            for (int s = 0; s < n_sites; s++) {
                ggml_tensor * w = dit_site_weight(&m->layers[l], s);
                if (!w) {
                    char b[128];
                    snprintf(b, sizeof(b), "layer %d site %s has no weight", l, dit_site_peft(s));
                    *err = b;
                    return false;
                }
                DitLokrSite & k = layers[(size_t) (l - lo)][(size_t) s];
                dit_lokr_factorization(w->ne[1], factor, &k.out_l, &k.out_k);
                dit_lokr_factorization(w->ne[0], factor, &k.in_m, &k.in_n);
                k.mono = dit_lokr_w2_mono(dim, k.out_k, k.in_n);
                if (dit_lokr_w1_would_factor(dim, decompose_both, k.out_l, k.in_m)) {
                    w1_forced++;  // K3: monolithic anyway, the loaders read no w1_a/w1_b
                }
                // K6: LyCORIS forces alpha = dim whenever BOTH factors are
                // monolithic, which makes the in-graph scale exactly 1.
                float a_eff = (alpha == 0.0f) ? (float) dim : alpha;
                if (k.mono) {
                    if (a_eff != (float) dim) {
                        alpha_forced++;
                    }
                    a_eff = (float) dim;
                }
                k.alpha_eff = a_eff;
                k.scale     = a_eff / (float) dim;
                if (k.mono) {
                    n_mono++;
                } else {
                    n_fact++;
                }

                char nm[128];
                k.w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k.in_m, k.out_l);
                snprintf(nm, sizeof(nm), "L%d.%s.lokr_w1", l, dit_site_peft(s));
                ggml_set_name(k.w1, nm);
                ggml_set_param(k.w1);
                par.push_back(k.w1);
                if (k.mono) {
                    k.w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k.in_n, k.out_k);
                    snprintf(nm, sizeof(nm), "L%d.%s.lokr_w2", l, dit_site_peft(s));
                    ggml_set_name(k.w2, nm);
                    ggml_set_param(k.w2);
                    par.push_back(k.w2);
                } else {
                    k.w2_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, k.out_k);
                    k.w2_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k.in_n, dim);
                    snprintf(nm, sizeof(nm), "L%d.%s.lokr_w2_a", l, dit_site_peft(s));
                    ggml_set_name(k.w2_a, nm);
                    snprintf(nm, sizeof(nm), "L%d.%s.lokr_w2_b", l, dit_site_peft(s));
                    ggml_set_name(k.w2_b, nm);
                    ggml_set_param(k.w2_a);
                    ggml_set_param(k.w2_b);
                    par.push_back(k.w2_a);
                    par.push_back(k.w2_b);
                }

                // Exportability is CHECKED, not assumed: kron(w1, w2) has to land
                // exactly on the base weight, or the adapter is unloadable.
                if (k.out_l * k.out_k != w->ne[1] || k.in_m * k.in_n != w->ne[0]) {
                    char b[240];
                    snprintf(b, sizeof(b),
                             "layer %d site %s does not factorize: base[%lld,%lld] but kron(w1[%lld,%lld], "
                             "w2[%lld,%lld]) = [%lld,%lld]",
                             l, dit_site_peft(s), (long long) w->ne[0], (long long) w->ne[1], (long long) k.out_l,
                             (long long) k.in_m, (long long) k.out_k, (long long) k.in_n,
                             (long long) (k.in_m * k.in_n), (long long) (k.out_l * k.out_k));
                    *err = b;
                    return false;
                }
            }
        }

        // LoKR tensors live in their OWN backend buffer, never the mirror's:
        // ggml_opt_step_adamw writes into them and the mirror is USAGE_WEIGHTS.
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) {
            *err = "LoKR parameter buffer allocation failed";
            return false;
        }

        // Init (§1): monolithic w2 -> 0, factorized w2_a kaiming / w2_b -> 0, w1
        // kaiming_uniform(a=sqrt(5)) == U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
        // Drawn in param order so the stream is reproducible from the seed alone.
        LmRng rng;
        lm_rng_seed(&rng, cfg.seed);
        std::vector<float> vals;
        for (int l = lo; l < hi; l++) {
            for (int s = 0; s < n_sites; s++) {
                DitLokrSite & k = layers[(size_t) (l - lo)][(size_t) s];
                fill_kaiming(&rng, k.w1, &vals);
                if (k.mono) {
                    fill_zero(&rng, k.w2, cfg.b_sigma, &vals);
                } else {
                    fill_kaiming(&rng, k.w2_a, &vals);
                    fill_zero(&rng, k.w2_b, cfg.b_sigma, &vals);
                }
            }
        }
        for (size_t i = 0; i < par.size(); i++) {
            n_params += (size_t) ggml_nelements(par[i]);
        }

        if (w1_forced > 0) {
            char b[224];
            snprintf(b, sizeof(b),
                     "LoKR dim %d would factorize w1 on %d site(s); forced monolithic — the engine's adapter "
                     "loaders read lokr_w1 only (K3)",
                     dim, w1_forced);
            lm_log("warn", b);
        }
        if (alpha_forced > 0) {
            char b[224];
            snprintf(b, sizeof(b),
                     "LoKR alpha %.4g overridden to dim %d on %d monolithic site(s) — LyCORIS forces scale 1 when "
                     "both factors are monolithic (K6)",
                     (double) alpha, dim, alpha_forced);
            lm_log("info", b);
        }
        fprintf(stderr, "[train-dit] LoKR: dim %d factor %d, %d monolithic / %d factorized w2, %zu params\n", dim,
                factor, n_mono, n_fact, n_params);
        return true;
    }

    // §2.3 kron-matvec. x [in, S]; the delta is contracted factor-by-factor:
    //   X3  = reshape(x)                     [in_n, in_m, S]     (n fastest, per
    //                                        row-major col index m*in_n + n)
    //   T1  = w2 . X3                        [out_k, in_m, S]
    //   T2  = (scale*w1) . swap(T1)          [out_l, out_k, S]
    //   d   = reshape(swap(T2))              [out_l*out_k, S]    (k fastest, per
    //                                        row index l*out_k + k)
    // ggml_permute(t, 1, 0, 2, 3) sends src axis0 to position 1 and axis1 to
    // position 0 — the ne0/ne1 swap both steps need. LK3 is the arbiter.
    ggml_tensor * apply(ggml_context * ctx_g, ggml_tensor * w, int layer, int s, ggml_tensor * x) const override {
        ggml_tensor *       y = ggml_mul_mat(ctx_g, w, x);
        const DitLokrSite * k = site(layer, s);
        if (!k) {
            return y;
        }
        ggml_tensor * xc = ggml_is_contiguous(x) ? x : ggml_cont(ctx_g, x);
        const int64_t S  = ggml_nelements(xc) / xc->ne[0];
        ggml_tensor * X3 = ggml_reshape_3d(ctx_g, xc, k->in_n, k->in_m, S);
        ggml_tensor * T1 = k->mono ? ggml_mul_mat(ctx_g, k->w2, X3)
                                   : ggml_mul_mat(ctx_g, k->w2_a, ggml_mul_mat(ctx_g, k->w2_b, X3));
        ggml_tensor * T1p   = ggml_cont(ctx_g, ggml_permute(ctx_g, T1, 1, 0, 2, 3));
        ggml_tensor * w1s   = ggml_scale(ctx_g, k->w1, k->scale);  // alpha/dim on the tiny side
        ggml_tensor * T2    = ggml_mul_mat(ctx_g, w1s, T1p);
        ggml_tensor * T2p   = ggml_cont(ctx_g, ggml_permute(ctx_g, T2, 1, 0, 2, 3));
        ggml_tensor * delta = ggml_reshape_4d(ctx_g, T2p, y->ne[0], y->ne[1], y->ne[2], y->ne[3]);
        return ggml_add(ctx_g, y, delta);
    }

    // §2.4: <dir>/lokr_weights.safetensors, LyCORIS key layout. No
    // adapter_config.json — that is PEFT-only and the LoKr loader never reads it.
    bool exportDir(const char * dir, const DitExportMeta & meta, DitExportResult * res,
                   std::string * err) const override {
        const std::string d(dir);
        if (!pm_mkdir_p(d)) {
            *err = "cannot create " + d;
            return false;
        }

        std::vector<STWTensor>          tensors;
        std::vector<std::vector<float>> store;
        tensors.reserve((size_t) (hi - lo) * (size_t) n_sites * 4);
        store.reserve(tensors.capacity());

        for (int l = lo; l < hi; l++) {
            for (int s = 0; s < n_sites; s++) {
                const DitLokrSite & k    = layers[(size_t) (l - lo)][(size_t) s];
                const std::string   stem = lokr_key_stem(l, s);

                store.push_back(std::vector<float>(1, k.alpha_eff));
                STWTensor sa;
                sa.name  = stem + ".alpha";
                sa.shape = { 1 };
                sa.data  = nullptr;
                tensors.push_back(sa);

                ggml_tensor * ts[3]  = { k.w1, k.mono ? k.w2 : k.w2_a, k.mono ? nullptr : k.w2_b };
                const char *  sfx[3] = { "lokr_w1", k.mono ? "lokr_w2" : "lokr_w2_a", "lokr_w2_b" };
                for (int i = 0; i < 3 && ts[i]; i++) {
                    store.push_back(std::vector<float>((size_t) ggml_nelements(ts[i])));
                    std::vector<float> & v = store.back();
                    ggml_backend_tensor_get(ts[i], v.data(), 0, v.size() * sizeof(float));
                    STWTensor st;
                    st.name  = stem + "." + sfx[i];
                    st.shape = { ts[i]->ne[1], ts[i]->ne[0] };  // row-major (rows, cols)
                    st.data  = nullptr;
                    tensors.push_back(st);
                }
            }
        }
        // `store` reallocates as it grows; re-point once it has stopped moving.
        GGML_ASSERT(store.size() == tensors.size());
        for (size_t i = 0; i < tensors.size(); i++) {
            tensors[i].data = store[i].data();
        }

        std::vector<std::pair<std::string, std::string>> md;
        md.push_back({ "format", "lycoris" });
        md.push_back({ "algo", "lokr" });
        md.push_back({ "producer", meta.producer });
        md.push_back({ "hot_step_dit_trainer", "v1" });
        // linear_dim is NOT optional: adapter-merge.h skips every monolithic
        // module whose lokr_config lacks it, with only a warning.
        md.push_back({ "lokr_config", lokr_config_json() });
        if (!meta.trigger.empty()) {
            md.push_back({ "hot_step_trigger", meta.trigger });
            md.push_back({ "hot_step_trigger_position",
                           meta.trigger_position.empty() ? std::string("prepend") : meta.trigger_position });
            md.push_back({ "modelspec.trigger_phrase", meta.trigger });
        }

        const std::string sf = lm_join(d, "lokr_weights.safetensors");
        if (!st_write_file(sf.c_str(), tensors, md, STW_F32)) {
            *err = "cannot write " + sf;
            return false;
        }
        if (res) {
            long long bytes = 0;
            pm_stat_file(sf, &bytes, NULL);
            res->tensors = (int) tensors.size();
            res->bytes   = bytes;
        }
        return true;
    }

    // "lycoris_layers_<L>_<peft site with dots flattened>" — the reverse of
    // lokr_build_reverse_map() in adapter-merge.h.
    static std::string lokr_key_stem(int l, int s) {
        std::string site_key(dit_site_peft(s));
        for (size_t i = 0; i < site_key.size(); i++) {
            if (site_key[i] == '.') {
                site_key[i] = '_';
            }
        }
        return "lycoris_layers_" + std::to_string(l) + "_" + site_key;
    }

    std::string lokr_config_json() const {
        // §2.4: linear_alpha is alpha_eff — the value that was actually TRAINED,
        // after K6's "both factors monolithic forces alpha = dim" override — not
        // the configured alpha. Read off a real site so the two can never drift;
        // the per-module `.alpha` tensors stay authoritative for the loaders.
        float a_eff = (alpha == 0.0f) ? (float) dim : alpha;
        if (!layers.empty()) {
            a_eff = layers[0][0].alpha_eff;
        }
        char b[224];
        snprintf(b, sizeof(b),
                 "{\"linear_dim\":%d,\"linear_alpha\":%.9g,\"factor\":%d,\"decompose_both\":%s,\"target_mlp\":%s}",
                 dim, (double) a_eff, factor,
                 decompose_both ? "true" : "false", n_sites == DIT_NSITES ? "true" : "false");
        return std::string(b);
    }

    static void fill_kaiming(LmRng * rng, ggml_tensor * t, std::vector<float> * v) {
        const size_t n = (size_t) ggml_nelements(t);
        const float  b = 1.0f / sqrtf((float) t->ne[0]);  // ggml ne0 == torch fan_in
        v->assign(n, 0.0f);
        for (size_t i = 0; i < n; i++) {
            (*v)[i] = (2.0f * lm_rng_uniform(rng) - 1.0f) * b;
        }
        ggml_backend_tensor_set(t, v->data(), 0, n * sizeof(float));
    }

    // dW = 0 at step 0. `sigma` > 0 only for the finite-difference gate, which
    // needs a non-zero w2 or every dL/dw1 is trivially zero.
    static void fill_zero(LmRng * rng, ggml_tensor * t, float sigma, std::vector<float> * v) {
        const size_t n = (size_t) ggml_nelements(t);
        v->assign(n, 0.0f);
        if (sigma > 0.0f) {
            lm_rng_fill_normal(rng, *v, sigma);
        }
        ggml_backend_tensor_set(t, v->data(), 0, n * sizeof(float));
    }
};

// Expected parameter count for the cross-check in dit-train-run.h.
static size_t dit_lokr_expected_params(const DiTGGMLConfig & c, int lo, int hi, int lokr_dim, int lokr_factor,
                                       bool target_mlp) {
    const int64_t H = c.hidden_size, Q = (int64_t) c.n_heads * c.head_dim, KV = (int64_t) c.n_kv_heads * c.head_dim,
                  F = c.intermediate_size;
    const int64_t sites[DIT_NSITES][2] = {
        { H, Q },  { H, KV }, { H, KV }, { Q, H },  // self q/k/v/o   {in, out}
        { H, Q },  { H, KV }, { H, KV }, { Q, H },  // cross q/k/v/o
        { H, F },  { H, F },  { F, H },             // gate / up / down
    };
    const int n_sites   = target_mlp ? DIT_NSITES : DIT_NSITES_ATTN;
    int64_t   per_layer = 0;
    for (int s = 0; s < n_sites; s++) {
        int64_t out_l, out_k, in_m, in_n;
        dit_lokr_factorization(sites[s][1], lokr_factor, &out_l, &out_k);
        dit_lokr_factorization(sites[s][0], lokr_factor, &in_m, &in_n);
        per_layer += out_l * in_m;
        per_layer += dit_lokr_w2_mono(lokr_dim, out_k, in_n) ? out_k * in_n
                                                             : out_k * (int64_t) lokr_dim + (int64_t) lokr_dim * in_n;
    }
    return (size_t) (per_layer * (int64_t) (hi - lo));
}
