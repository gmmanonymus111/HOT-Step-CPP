#pragma once
// minimax/mm3-lm-adapter.h — runtime LoRA adapters for the MM3 global LM.
//
// HOT-Step file (does not exist upstream). Loads a PEFT-format LM LoRA
// (SimpleTuner `--minimax_music_train_component=language_model` checkpoints:
// keys `language_model.model.layers.N.{self_attn.{q,k,v,o}_proj,mlp.{gate,up,
// down}_proj}.lora_{A,B}.weight`) and applies it as RUNTIME low-rank deltas
// inside mm3-lm-graph.h — the base weights are never modified, so per-group
// scales are live per generation with no reload.
//
// Why runtime deltas and not a load-time merge: the whole point (validated by
// the 2026-08-20 ablation grid on the alk3 r256 adapter) is DIALING groups at
// render time — attention carries the plan/genre, the MLPs carry vocal
// identity AND the fidelity damage, and attention 1.0 / MLP 0.5 was the ear
// winner. A merge would freeze one setting into 17 GB of resident weights.
//
// Scale model (multiplicative, all default 1.0):
//     effective(module, layer) = global
//                              * (attn | mlp)          by module kind
//                              * (early | mid | late)  by layer/12
// PEFT's own alpha/rank scaling: SimpleTuner LM checkpoints carry NO alpha
// tensors and train with alpha == rank, so the baked base scale is 1.0. If a
// per-module `.alpha` scalar IS present (comfy-style exports), it is honoured
// as alpha/rank.
//
// Cost: r256 f16 adapter ≈ 1.5 GB resident, streamed per AR token on top of
// the 17.2 GB base — expect ≈ +9 % on the LM step at rank 256, proportionally
// less at lower ranks. VRAM and step cost both scale linearly with rank.
//
// Scales are baked into cached graphs as constants; mm3-lm-graph.h owns an
// `adapter_epoch` and rebuilds its slots when (adapter, scales) change. That
// is once per generation, not per frame — rebuild cost is graph construction
// only.

#include "backend.h"
#include "safetensors.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

#define MM3_LM_ADAPTER_LAYERS  36
#define MM3_LM_ADAPTER_MODULES 7

enum MM3LmAdapterModule {
    MM3_LM_ADAPTER_Q = 0,
    MM3_LM_ADAPTER_K,
    MM3_LM_ADAPTER_V,
    MM3_LM_ADAPTER_O,
    MM3_LM_ADAPTER_GATE,
    MM3_LM_ADAPTER_UP,
    MM3_LM_ADAPTER_DOWN,
};

static const char * MM3_LM_ADAPTER_MODULE_KEY[MM3_LM_ADAPTER_MODULES] = {
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj",    "mlp.up_proj",      "mlp.down_proj",
};

struct MM3LmAdapterScales {
    float global = 1.0f;
    float attn   = 1.0f;
    float mlp    = 1.0f;
    float early  = 1.0f;  // layers 0..11
    float mid    = 1.0f;  // layers 12..23
    float late   = 1.0f;  // layers 24..35

    bool operator==(const MM3LmAdapterScales & o) const {
        return global == o.global && attn == o.attn && mlp == o.mlp && early == o.early && mid == o.mid &&
               late == o.late;
    }

    bool operator!=(const MM3LmAdapterScales & o) const { return !(*this == o); }
};

struct MM3LmAdapterPair {
    ggml_tensor * a          = nullptr;  // [in, r]  (f16)
    ggml_tensor * b          = nullptr;  // [r, out] (f16)
    float         base_scale = 1.0f;     // alpha/rank when an alpha scalar exists, else 1
};

struct MM3LmAdapter {
    std::string      path;
    int64_t          mtime = 0;
    int              rank  = 0;
    MM3LmAdapterPair mods[MM3_LM_ADAPTER_LAYERS][MM3_LM_ADAPTER_MODULES];
    int              n_loaded = 0;

    ggml_context *        ctx         = nullptr;
    ggml_backend_buffer_t buf         = nullptr;
    BackendPair           bp          = {};
    bool                  backend_ref = false;

    // effective scale for one module instance under the request's dials
    float effective(int layer, int module, const MM3LmAdapterScales & s) const {
        const MM3LmAdapterPair & p = mods[layer][module];
        float                    v = s.global * p.base_scale;
        v *= (module <= MM3_LM_ADAPTER_O) ? s.attn : s.mlp;
        v *= (layer < 12) ? s.early : (layer < 24) ? s.mid : s.late;
        return v;
    }
};

static void mm3_lm_adapter_free(MM3LmAdapter * ad) {
    if (!ad) {
        return;
    }
    if (ad->buf) {
        ggml_backend_buffer_free(ad->buf);
    }
    if (ad->ctx) {
        ggml_free(ad->ctx);
    }
    if (ad->backend_ref) {
        backend_release(ad->bp.backend, ad->bp.cpu_backend);
    }
    delete ad;
}

// Accept keys with or without the `language_model.` prefix, and PEFT's
// `.default.` infix (present after PeftModel round-trips).
static bool mm3_lm_adapter_parse_key(const std::string & key, int * layer, int * module, bool * is_a) {
    const char * s = key.c_str();
    if (strncmp(s, "language_model.", 15) == 0) {
        s += 15;
    }
    if (strncmp(s, "base_model.model.", 17) == 0) {
        s += 17;
    }
    if (strncmp(s, "model.layers.", 13) != 0) {
        return false;
    }
    s += 13;
    char * end   = nullptr;
    long   lyr   = strtol(s, &end, 10);
    if (end == s || *end != '.' || lyr < 0 || lyr >= MM3_LM_ADAPTER_LAYERS) {
        return false;
    }
    s = end + 1;
    int mod = -1;
    for (int m = 0; m < MM3_LM_ADAPTER_MODULES; m++) {
        size_t n = strlen(MM3_LM_ADAPTER_MODULE_KEY[m]);
        if (strncmp(s, MM3_LM_ADAPTER_MODULE_KEY[m], n) == 0 && s[n] == '.') {
            mod = m;
            s += n + 1;
            break;
        }
    }
    if (mod < 0) {
        return false;
    }
    // remainder: lora_A.weight / lora_B.weight, optionally lora_A.default.weight
    if (strncmp(s, "lora_A.", 7) == 0) {
        *is_a = true;
    } else if (strncmp(s, "lora_B.", 7) == 0) {
        *is_a = false;
    } else {
        return false;
    }
    *layer  = (int) lyr;
    *module = mod;
    return true;
}

// Load a PEFT LM LoRA. Acquires its own backend reference (same shared pool
// as every other module). Returns nullptr with a message on any structural
// problem — a half-loaded adapter is worse than none (the LM-echo whitelist
// lesson: silently dropping modules changes what the adapter IS).
static MM3LmAdapter * mm3_lm_adapter_load(const char * path, std::string * err) {
    STFile st;
    if (!st_open(&st, path)) {
        if (err) {
            *err = std::string("cannot open adapter: ") + path;
        }
        return nullptr;
    }

    struct stat sb {};

    stat(path, &sb);

    MM3LmAdapter * ad = new MM3LmAdapter();
    ad->path          = path;
    ad->mtime         = (int64_t) sb.st_mtime;
    ad->bp            = backend_init("MM3-LM-Adapter");
    ad->backend_ref   = true;
    ggml_backend_t backend = ad->bp.backend ? ad->bp.backend : ad->bp.cpu_backend;

    // Pass 1: count matched pairs so the ggml context can be sized exactly.
    int matched = 0;
    for (const STEntry & e : st.entries) {
        int  layer, module;
        bool is_a;
        if (mm3_lm_adapter_parse_key(e.name, &layer, &module, &is_a)) {
            matched++;
        }
    }
    if (matched == 0) {
        st_close(&st);
        if (err) {
            *err = "no language_model LoRA keys found (is this a DiT adapter?)";
        }
        mm3_lm_adapter_free(ad);
        return nullptr;
    }

    ggml_init_params ip = {
        /*mem_size   =*/(size_t) (matched + 2) * ggml_tensor_overhead(),
        /*mem_buffer =*/nullptr,
        /*no_alloc   =*/true,
    };
    ad->ctx = ggml_init(ip);

    // Pass 2: create tensors (f16, torch row-major maps directly onto ggml
    // [ne0 = innermost]): A [r, in] -> ggml [in, r]; B [out, r] -> ggml [r, out].
    for (const STEntry & e : st.entries) {
        int  layer, module;
        bool is_a;
        if (!mm3_lm_adapter_parse_key(e.name, &layer, &module, &is_a)) {
            continue;
        }
        if (e.n_dims != 2) {
            if (err) {
                *err = "unexpected LoRA tensor rank on " + e.name;
            }
            st_close(&st);
            mm3_lm_adapter_free(ad);
            return nullptr;
        }
        // torch shape[0] = rows (r for A, out for B), shape[1] = cols
        const int64_t ne0 = e.shape[1];  // innermost
        const int64_t ne1 = e.shape[0];
        ggml_tensor * t   = ggml_new_tensor_2d(ad->ctx, GGML_TYPE_F16, ne0, ne1);
        ggml_set_name(t, e.name.c_str());
        MM3LmAdapterPair & p = ad->mods[layer][module];
        (is_a ? p.a : p.b) = t;
        if (is_a) {
            ad->rank = (int) ne1;
        }
    }

    ad->buf = ggml_backend_alloc_ctx_tensors(ad->ctx, backend);
    if (!ad->buf) {
        st_close(&st);
        if (err) {
            *err = "adapter VRAM allocation failed";
        }
        mm3_lm_adapter_free(ad);
        return nullptr;
    }

    // Pass 3: upload, converting any dtype to f16 via f32.
    std::vector<float>      f32;
    std::vector<ggml_fp16_t> f16;
    for (const STEntry & e : st.entries) {
        int  layer, module;
        bool is_a;
        if (!mm3_lm_adapter_parse_key(e.name, &layer, &module, &is_a)) {
            continue;
        }
        MM3LmAdapterPair & p = ad->mods[layer][module];
        ggml_tensor *      t = is_a ? p.a : p.b;
        const int64_t      n = ggml_nelements(t);
        f32.resize((size_t) n);
        if (!adapter_to_f32(st_data(st, e), f32.data(), n, e.dtype)) {
            st_close(&st);
            if (err) {
                *err = "unsupported dtype " + e.dtype + " on " + e.name;
            }
            mm3_lm_adapter_free(ad);
            return nullptr;
        }
        f16.resize((size_t) n);
        ggml_fp32_to_fp16_row(f32.data(), f16.data(), n);
        ggml_backend_tensor_set(t, f16.data(), 0, (size_t) n * sizeof(ggml_fp16_t));
    }
    st_close(&st);

    // Pass 4: validate pairing + count. Every module with an A must have a B.
    for (int l = 0; l < MM3_LM_ADAPTER_LAYERS; l++) {
        for (int m = 0; m < MM3_LM_ADAPTER_MODULES; m++) {
            MM3LmAdapterPair & p = ad->mods[l][m];
            if ((p.a == nullptr) != (p.b == nullptr)) {
                if (err) {
                    *err = "unpaired lora_A/lora_B at layer " + std::to_string(l);
                }
                mm3_lm_adapter_free(ad);
                return nullptr;
            }
            if (p.a) {
                ad->n_loaded++;
            }
        }
    }
    fprintf(stderr, "[MM3] LM adapter loaded: %s (%d modules, rank %d, %.1f MB)\n", path, ad->n_loaded, ad->rank,
            (double) ggml_backend_buffer_get_size(ad->buf) / 1e6);
    return ad;
}
