#pragma once
// lm-adapter.h: runtime LoRA loading for the 5Hz planner LM.
//
// Loads a PEFT LoRA adapter (directory with adapter_model.safetensors +
// adapter_config.json, or a bare .safetensors file) and stages the A/B
// pairs on the LM's backend.  Applied at graph-build time via
// qwen3_linear_lora() — never merged, so the base LM can be any quant
// (Q8_0/Q5_K/NVFP4/...) with zero requantization loss.  Per-token cost at
// rank 16 is negligible (two rank-r matmuls per adapted projection).
//
// Trained by Side-Step's `sidestep lm-train` (sidestep_engine/lm/), which
// replicates the engine's exact prompt format (engine/src/prompt.h).
//
// PEFT tensor naming handled:
//   base_model.model.model.layers.N.self_attn.q_proj.lora_A.weight
//   (any prefix before "model.layers." is ignored)
// Supported slots: q/k/v/o_proj, gate/up/down_proj.  Anything else
// (embed_tokens, lm_head) is skipped with a warning.
//
// Local HOT-Step feature — not upstream acestep.cpp.

#include "qwen3-lora.h"
#include "safetensors.h"
#include "yyjson.h"

#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct LMLora {
    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    QwLoraLayer           layers[QWEN3_LORA_MAX_LAYERS];
    int                   n_tensors = 0;
    int                   max_layer = -1;
    std::string           path;
    float                 user_scale = 1.0f;
};

// Map a module substring to a slot. Order matters: check longer names first.
static bool lm_adapter_slot_for(const std::string & name, QwLoraSlot * out) {
    struct { const char * pat; QwLoraSlot slot; } map[] = {
        { ".self_attn.q_proj.", QW_LORA_Q },
        { ".self_attn.k_proj.", QW_LORA_K },
        { ".self_attn.v_proj.", QW_LORA_V },
        { ".self_attn.o_proj.", QW_LORA_O },
        { ".mlp.gate_proj.", QW_LORA_GATE },
        { ".mlp.up_proj.", QW_LORA_UP },
        { ".mlp.down_proj.", QW_LORA_DOWN },
    };
    for (auto & m : map) {
        if (name.find(m.pat) != std::string::npos) {
            *out = m.slot;
            return true;
        }
    }
    return false;
}

// Parse "…model.layers.<N>…" → layer index, or -1.
static int lm_adapter_layer_for(const std::string & name) {
    size_t p = name.find("model.layers.");
    if (p == std::string::npos) return -1;
    p += strlen("model.layers.");
    if (p >= name.size() || name[p] < '0' || name[p] > '9') return -1;
    return atoi(name.c_str() + p);
}

// Read lora_alpha and r from adapter_config.json. Returns alpha/r ratio
// factor, or -1 when the config is missing/unparseable (caller falls back
// to per-tensor alpha=r, i.e. factor 1.0, with a warning).
static float lm_adapter_read_alpha_ratio(const std::string & dir) {
    std::string  cfg_path = dir + "/adapter_config.json";
    yyjson_doc * doc      = yyjson_read_file(cfg_path.c_str(), 0, NULL, NULL);
    if (!doc) return -1.0f;
    yyjson_val * root  = yyjson_doc_get_root(doc);
    double       alpha = 0.0, r = 0.0;
    if (root && yyjson_is_obj(root)) {
        yyjson_val * a = yyjson_obj_get(root, "lora_alpha");
        yyjson_val * rv = yyjson_obj_get(root, "r");
        if (a && yyjson_is_num(a)) alpha = yyjson_get_num(a);
        if (rv && yyjson_is_num(rv)) r = yyjson_get_num(rv);
    }
    yyjson_doc_free(doc);
    if (alpha > 0.0 && r > 0.0) return (float) (alpha / r);
    return -1.0f;
}

static void lm_adapter_free(LMLora * l) {
    if (!l) return;
    if (l->buf) ggml_backend_buffer_free(l->buf);
    if (l->ctx) ggml_free(l->ctx);
    delete l;
}

// Load a PEFT LoRA for the LM and stage its tensors on `backend`.
// Returns nullptr on failure (caller must treat this as a load failure —
// never fall back silently to the base LM under an adapter-bearing key).
static LMLora * lm_adapter_load(const char * path, float user_scale, ggml_backend_t backend) {
    std::string p = path;
    std::string dir;
    std::string sf_path;
    bool        is_file = p.size() > 12 && p.compare(p.size() - 12, 12, ".safetensors") == 0;
    if (is_file) {
        sf_path = p;
        size_t slash = p.find_last_of("/\\");
        dir = (slash == std::string::npos) ? "." : p.substr(0, slash);
    } else {
        dir = p;
        sf_path = p + "/adapter_model.safetensors";
    }

    STFile st = {};
    if (!st_open(&st, sf_path.c_str())) {
        fprintf(stderr, "[LM-Adapter] FATAL: cannot open %s\n", sf_path.c_str());
        return nullptr;
    }

    float alpha_ratio = lm_adapter_read_alpha_ratio(dir);
    bool  ratio_from_cfg = alpha_ratio > 0.0f;

    LMLora * l = new LMLora();
    l->path = p;
    l->user_scale = user_scale;

    // Pass 1: count usable tensors
    int usable = 0, skipped = 0;
    for (const STEntry & e : st.entries) {
        QwLoraSlot slot;
        int        layer = lm_adapter_layer_for(e.name);
        bool is_ab = e.name.find(".lora_A.") != std::string::npos ||
                     e.name.find(".lora_B.") != std::string::npos;
        if (layer >= 0 && layer < QWEN3_LORA_MAX_LAYERS && is_ab && lm_adapter_slot_for(e.name, &slot)) {
            usable++;
        } else {
            skipped++;
        }
    }
    if (usable == 0) {
        fprintf(stderr, "[LM-Adapter] FATAL: no usable lora_A/lora_B tensors in %s (%d entries)\n",
                sf_path.c_str(), (int) st.entries.size());
        st_close(&st);
        lm_adapter_free(l);
        return nullptr;
    }

    struct ggml_init_params gp = { ggml_tensor_overhead() * (size_t) usable, NULL, true };
    l->ctx = ggml_init(gp);

    // Pass 2: create tensors
    struct Pending { const STEntry * e; struct ggml_tensor * t; };
    std::vector<Pending> pending;
    pending.reserve(usable);
    for (const STEntry & e : st.entries) {
        QwLoraSlot slot;
        int        layer = lm_adapter_layer_for(e.name);
        bool  is_a = e.name.find(".lora_A.") != std::string::npos;
        bool  is_b = e.name.find(".lora_B.") != std::string::npos;
        if (layer < 0 || layer >= QWEN3_LORA_MAX_LAYERS || (!is_a && !is_b) ||
            !lm_adapter_slot_for(e.name, &slot)) {
            continue;
        }
        if (e.n_dims != 2) {
            fprintf(stderr, "[LM-Adapter] FATAL: %s has %d dims (want 2)\n", e.name.c_str(), e.n_dims);
            st_close(&st);
            lm_adapter_free(l);
            return nullptr;
        }
        ggml_type ty = st_ggml_type(e);
        if (ty == GGML_TYPE_COUNT) {
            fprintf(stderr, "[LM-Adapter] FATAL: %s has unsupported dtype %s\n", e.name.c_str(), e.dtype.c_str());
            st_close(&st);
            lm_adapter_free(l);
            return nullptr;
        }
        // torch [rows, cols] row-major -> ggml ne0=cols, ne1=rows
        struct ggml_tensor * t = ggml_new_tensor_2d(l->ctx, ty, e.shape[1], e.shape[0]);
        QwLoraPair & pair = l->layers[layer].p[slot];
        if (is_a) pair.A = t; else pair.B = t;
        if (layer > l->max_layer) l->max_layer = layer;
        pending.push_back({ &e, t });
    }

    l->buf = ggml_backend_alloc_ctx_tensors(l->ctx, backend);
    if (!l->buf) {
        fprintf(stderr, "[LM-Adapter] FATAL: backend alloc failed for %d tensors\n", (int) pending.size());
        st_close(&st);
        lm_adapter_free(l);
        return nullptr;
    }
    for (auto & pd : pending) {
        ggml_backend_tensor_set(pd.t, st_data(st, *pd.e), 0, ggml_nbytes(pd.t));
    }
    l->n_tensors = (int) pending.size();
    st_close(&st);

    // Finalize pairs: both halves present, per-pair scale = ratio * user_scale
    // (ratio from adapter_config.json, else alpha=r fallback -> 1.0).
    if (!ratio_from_cfg) {
        fprintf(stderr, "[LM-Adapter] WARNING: no adapter_config.json next to %s — "
                        "assuming lora_alpha == r (scale factor 1.0)\n", sf_path.c_str());
        alpha_ratio = 1.0f;
    }
    int pairs = 0;
    for (int i = 0; i <= l->max_layer; i++) {
        for (int s = 0; s < QW_LORA_NSLOTS; s++) {
            QwLoraPair & pr = l->layers[i].p[s];
            if ((pr.A == nullptr) != (pr.B == nullptr)) {
                fprintf(stderr, "[LM-Adapter] FATAL: layer %d slot %d has only one of lora_A/lora_B\n", i, s);
                lm_adapter_free(l);
                return nullptr;
            }
            if (pr.A) {
                pr.scale = alpha_ratio * user_scale;
                pairs++;
            }
        }
    }

    fprintf(stderr, "[LM-Adapter] Loaded %s: %d pairs across %d layers, alpha/r=%.3f, user scale=%.2f%s\n",
            p.c_str(), pairs, l->max_layer + 1, alpha_ratio, user_scale,
            skipped ? " (some non-projection tensors skipped)" : "");
    return l;
}
