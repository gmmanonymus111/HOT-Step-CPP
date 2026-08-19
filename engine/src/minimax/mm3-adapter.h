#pragma once
// minimax/mm3-adapter.h — MiniMax-Music3 flow-DiT LoRA merge.
//
// HOT-Step file (does not exist upstream). Included only from inside
// engine/src/minimax/, per the fork's hook doctrine.
//
// SCOPE (increment 1): merge one or more LoRAs into the flow DiT's weights at
// load time, in the seam between mm3_load_rest_tensors() and wctx_alloc() —
// exactly where ACE merges in dit.h. Deliberately NOT in scope yet: LoKr, DoRA,
// basin re-base, and runtime (unmerged) mode. Those exist for ACE and can be
// lifted across once there is an MM3 adapter corpus that needs them.
//
// Everything expensive is REUSED from adapter-merge.h: the safetensors reader,
// alpha handling, and adapter_merge_on_backend()'s single-graph
// upload/dequant/BF16-round/add/re-encode/download pipeline. What is genuinely
// MM3-specific, and all this file really adds, is the KEY MAPPING.
//
// ── The two on-disk formats ────────────────────────────────────────────────
//
// Both come from SimpleTuner (the only MM3 trainer as of 2026-08-14), whose
// converter in helpers/models/minimaxmusic/modular_pipeline.py is the
// authoritative spec this mapping was written against.
//
//   ComfyUI ("lora_format": "comfyui", the shipping default). Prefixed
//   `diffusion_model.` / `model.diffusion_model.`, ComfyUI module naming, and
//   **q/k/v already fused into `.to_qkv`** in q,k,v row order.
//
//     diffusion_model.diffusion_transformer.transformer.layers.7.self_attn.to_qkv.lora_A.weight
//
//   Diffusers/PEFT. Prefixed `transformer.` (or `base_model.model.`), diffusers
//   module naming, **q/k/v as three separate modules**.
//
//     transformer.transformer_blocks.7.attn.to_q.lora_A.weight
//
// Our GGUF fuses qkv at conversion time (convert-mm3.py chunks Comfy's to_qkv
// as q,k,v), so:
//   - the ComfyUI form maps 1:1 onto `dit.blk.N.attn_qkv.weight` — no slicing;
//   - the diffusers form needs its three sub-deltas assembled into one
//     full-height delta, which this file does by concatenating [q|k|v] blocks
//     along ne1 (zero-filling any projection the adapter did not train).
//
// ── The SwiGLU trap ───────────────────────────────────────────────────────
//
// `ffn_in` produces both halves of a SwiGLU. Our GGUF declares
// `mm3.dit.glu_order = "value_gate"`; LyCORIS/diffusers checkpoints may be
// gate-first, and SimpleTuner tracks exactly this as a `swiglu_gate_first` key
// in the safetensors __metadata__ (it swaps the halves itself on ComfyUI
// export, which is why the ComfyUI form is always value-first). A gate-first
// adapter merged without swapping is silently, quietly wrong — the audio still
// renders. So we read the metadata and swap the B rows when it says to.

#include "adapter-merge.h"
#include "safetensors.h"
#include "timer.h"
#include "weight-ctx.h"
#include "weight-source.h"

#include <cctype>
#include <map>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

// Local formatter. Deliberately NOT mm3-model.h's mm3_fmt: that is defined
// partway down that file, and depending on it would make this header's include
// position load-bearing.
static std::string mm3a_fmt(const char * fmt, int a) {
    char buf[128];
    snprintf(buf, sizeof(buf), fmt, a);
    return std::string(buf);
}

// ── Key parsing ─────────────────────────────────────────────────────────────

// Which slot of a fused qkv a key targets. WHOLE means the key covers the
// entire base tensor (everything except a diffusers-form to_q/to_k/to_v).
enum MM3QkvSlot { MM3_QKV_WHOLE = -1, MM3_QKV_Q = 0, MM3_QKV_K = 1, MM3_QKV_V = 2 };

struct MM3LoraTarget {
    std::string gguf_name;                 // empty => not a tensor we merge
    MM3QkvSlot  slot = MM3_QKV_WHOLE;
};

// Strip the LoRA factor suffix. Returns 'A', 'B', 'a' (alpha scalar) or 0.
static char mm3_lora_suffix(const std::string & key, std::string & module_out) {
    static const char * a_sfx[] = { ".lora_A.weight", ".lora.down.weight", ".lora_down.weight" };
    static const char * b_sfx[] = { ".lora_B.weight", ".lora.up.weight", ".lora_up.weight" };
    for (const char * s : a_sfx) {
        size_t n = strlen(s);
        if (key.size() > n && key.compare(key.size() - n, n, s) == 0) {
            module_out = key.substr(0, key.size() - n);
            return 'A';
        }
    }
    for (const char * s : b_sfx) {
        size_t n = strlen(s);
        if (key.size() > n && key.compare(key.size() - n, n, s) == 0) {
            module_out = key.substr(0, key.size() - n);
            return 'B';
        }
    }
    for (const char * s : { ".alpha", ".lora_alpha" }) {
        size_t n = strlen(s);
        if (key.size() > n && key.compare(key.size() - n, n, s) == 0) {
            module_out = key.substr(0, key.size() - n);
            return 'a';
        }
    }
    return 0;
}

static bool mm3_str_starts(const std::string & s, const char * p) {
    size_t n = strlen(p);
    return s.size() >= n && s.compare(0, n, p) == 0;
}

static bool mm3_str_ends(const std::string & s, const char * p) {
    size_t n = strlen(p);
    return s.size() >= n && s.compare(s.size() - n, n, p) == 0;
}

// Parse "<head>layers.<i>.<tail>" / "<head>transformer_blocks.<i>.<tail>".
// Returns false when the module is not per-block.
static bool mm3_split_block(const std::string & mod, const char * infix, int & idx_out, std::string & tail_out) {
    size_t p = mod.find(infix);
    if (p == std::string::npos) {
        return false;
    }
    size_t d = p + strlen(infix);
    size_t e = d;
    while (e < mod.size() && isdigit((unsigned char) mod[e])) {
        e++;
    }
    if (e == d || e >= mod.size() || mod[e] != '.') {
        return false;
    }
    idx_out  = atoi(mod.substr(d, e - d).c_str());
    tail_out = mod.substr(e + 1);
    return true;
}

// Map one adapter module path onto a GGUF tensor name.
//
// Handles both formats described in the file header. Returns an empty
// gguf_name for anything we deliberately do not merge (see the skip list at the
// bottom) — the caller logs those once, rather than per tensor.
static MM3LoraTarget mm3_lora_target(const std::string & raw_key) {
    MM3LoraTarget t;

    std::string m = raw_key;
    for (const char * pfx : { "model.diffusion_model.", "diffusion_model.", "base_model.model.", "transformer." }) {
        if (mm3_str_starts(m, pfx)) {
            m = m.substr(strlen(pfx));
            break;
        }
    }
    // PEFT can wrap modules; ACE hits the same thing (adapter-merge.h).
    for (const char * infix : { ".original_module.", ".base_layer." }) {
        size_t p = m.find(infix);
        if (p != std::string::npos) {
            m = m.substr(0, p) + "." + m.substr(p + strlen(infix));
        }
    }

    int         blk = -1;
    std::string tail;

    // ── ComfyUI / native naming ──
    if (mm3_str_starts(m, "diffusion_transformer.")) {
        if (mm3_split_block(m, "transformer.layers.", blk, tail)) {
            if (tail == "self_attn.to_qkv") { t.gguf_name = mm3a_fmt("dit.blk.%d.attn_qkv.weight",   blk); }
            else if (tail == "self_attn.to_out") { t.gguf_name = mm3a_fmt("dit.blk.%d.attn_output.weight", blk); }
            else if (tail == "ff.ff.0.proj") { t.gguf_name = mm3a_fmt("dit.blk.%d.ffn_in.weight",  blk); }
            else if (tail == "ff.ff.2") { t.gguf_name = mm3a_fmt("dit.blk.%d.ffn_out.weight", blk); }
            return t;
        }
        if (mm3_str_ends(m, "transformer.project_in"))  { t.gguf_name = "dit.proj_in.weight";  return t; }
        if (mm3_str_ends(m, "transformer.project_out")) { t.gguf_name = "dit.proj_out.weight"; return t; }
        return t;  // conv / timestep / anything else: not merged (see skip list)
    }

    // ── Diffusers naming ──
    if (mm3_split_block(m, "transformer_blocks.", blk, tail)) {
        if (tail == "attn.to_q") { t.gguf_name = mm3a_fmt("dit.blk.%d.attn_qkv.weight", blk); t.slot = MM3_QKV_Q; return t; }
        if (tail == "attn.to_k") { t.gguf_name = mm3a_fmt("dit.blk.%d.attn_qkv.weight", blk); t.slot = MM3_QKV_K; return t; }
        if (tail == "attn.to_v") { t.gguf_name = mm3a_fmt("dit.blk.%d.attn_qkv.weight", blk); t.slot = MM3_QKV_V; return t; }
        if (tail == "attn.to_qkv") { t.gguf_name = mm3a_fmt("dit.blk.%d.attn_qkv.weight", blk); return t; }
        if (tail == "attn.to_out.0") { t.gguf_name = mm3a_fmt("dit.blk.%d.attn_output.weight", blk); return t; }
        if (tail == "ff_in")  { t.gguf_name = mm3a_fmt("dit.blk.%d.ffn_in.weight",  blk); return t; }
        if (tail == "ff_out") { t.gguf_name = mm3a_fmt("dit.blk.%d.ffn_out.weight", blk); return t; }
        return t;
    }
    if (m == "proj_in")  { t.gguf_name = "dit.proj_in.weight";  return t; }
    if (m == "proj_out") { t.gguf_name = "dit.proj_out.weight"; return t; }

    return t;
}

// ── SwiGLU orientation ──────────────────────────────────────────────────────

// Read SimpleTuner's `swiglu_gate_first` out of the safetensors __metadata__.
// Returns -1 when the key is absent (which we treat as "matches us", i.e. the
// value-first layout our GGUF declares), 0 for false, 1 for true.
static int mm3_read_swiglu_gate_first(const STFile & st) {
    if (st.data_offset <= 8) {
        return -1;
    }
    yyjson_doc * doc = yyjson_read((const char *) st.mapping + 8, st.data_offset - 8, 0);
    if (!doc) {
        return -1;
    }
    int          out  = -1;
    yyjson_val * meta = yyjson_obj_get(yyjson_doc_get_root(doc), "__metadata__");
    if (meta && yyjson_is_obj(meta)) {
        // SimpleTuner writes it bare or namespaced by the component it applies to.
        for (const char * k : { "swiglu_gate_first", "transformer.swiglu_gate_first" }) {
            yyjson_val * v = yyjson_obj_get(meta, k);
            if (!v) {
                continue;
            }
            if (yyjson_is_bool(v)) {
                out = yyjson_get_bool(v) ? 1 : 0;
            } else if (yyjson_is_str(v)) {
                std::string s = yyjson_get_str(v);
                for (auto & c : s) { c = (char) tolower((unsigned char) c); }
                out = (s == "true" || s == "1") ? 1 : 0;
            } else if (yyjson_is_int(v)) {
                out = yyjson_get_int(v) ? 1 : 0;
            }
            break;
        }
    }
    yyjson_doc_free(doc);
    return out;
}

// ── The merge ───────────────────────────────────────────────────────────────

// One (A, B, alpha) triple bound for a slot of one GGUF tensor.
struct MM3LoraFactor {
    const STEntry * a     = nullptr;
    const STEntry * b     = nullptr;
    float           alpha = 0.0f;   // 0 => default to rank (scaling 1.0)
    MM3QkvSlot      slot  = MM3_QKV_WHOLE;
};

// Merge every LoRA tensor in `st` into `wctx`, which must be a MM3 synth
// WeightCtx that has been filled by mm3_load_rest_tensors() but NOT yet passed
// to wctx_alloc(). `scale` is the user-facing strength.
//
// Returns the number of GGUF tensors actually patched. Zero means the adapter
// matched nothing — the caller decides whether that is fatal.
static int mm3_adapter_merge_st(WeightCtx *         wctx,
                                const GGUFModel &   gf,
                                const STFile &      st,
                                const std::string & cfg_dir,
                                float               scale,
                                ggml_backend_t      backend) {
    WeightSource ws = {};
    ws.is_st        = false;
    ws.gf           = const_cast<GGUFModel *>(&gf);

    const int cfg_alpha    = adapter_read_alpha(cfg_dir.c_str());
    const int gate_first   = mm3_read_swiglu_gate_first(st);
    // Our GGUF is value-first (mm3.dit.glu_order = "value_gate"). Only a
    // gate-first adapter needs its ffn_in output rows swapped.
    const bool swiglu_swap = (gate_first == 1);

    // Group every factor by target GGUF tensor. A fused-qkv tensor can collect
    // up to three (one per projection) from a diffusers-form adapter.
    std::map<std::string, std::vector<MM3LoraFactor>> groups;
    std::map<std::string, float>                      alpha_by_module;
    int                                               unmapped = 0;

    // Pass 1: the baked per-module .alpha scalars.
    for (const auto & e : st.entries) {
        std::string mod;
        if (mm3_lora_suffix(e.name, mod) == 'a' && e.dtype == "F32" && e.n_dims == 0) {
            float v = 0.0f;
            memcpy(&v, st_data(st, e), sizeof(float));
            alpha_by_module[mod] = v;
        }
    }

    // Pass 2: the A/B factors.
    std::map<std::string, const STEntry *> a_by_module, b_by_module;
    for (const auto & e : st.entries) {
        std::string mod;
        char        which = mm3_lora_suffix(e.name, mod);
        if (which == 'A') { a_by_module[mod] = &e; }
        else if (which == 'B') { b_by_module[mod] = &e; }
    }

    for (const auto & kv : a_by_module) {
        const std::string & mod = kv.first;
        auto                bit = b_by_module.find(mod);
        if (bit == b_by_module.end()) {
            fprintf(stderr, "[MM3-Adapter] WARNING: no lora_B for %s, skipping\n", mod.c_str());
            continue;
        }
        MM3LoraTarget t = mm3_lora_target(mod);
        if (t.gguf_name.empty()) {
            unmapped++;
            continue;
        }
        MM3LoraFactor f;
        f.a    = kv.second;
        f.b    = bit->second;
        f.slot = t.slot;
        auto ait = alpha_by_module.find(mod);
        f.alpha  = (ait != alpha_by_module.end()) ? ait->second : (float) cfg_alpha;
        groups[t.gguf_name].push_back(f);
    }

    if (unmapped) {
        fprintf(stderr, "[MM3-Adapter] %d adapter module(s) target tensors we do not merge "
                        "(conv/timestep/norm) — ignored\n", unmapped);
    }

    // pending lookups: by src pointer for adapter_merge_on_backend, and by
    // tensor NAME so a second adapter stacks on the first one's merged result
    // rather than on the pristine GGUF bytes (ACE does the same in dit.h).
    std::unordered_map<const void *, size_t> pending_idx;
    std::unordered_map<std::string, size_t>  pending_by_name;
    pending_idx.reserve(wctx->pending.size());
    for (size_t i = 0; i < wctx->pending.size(); i++) {
        pending_idx[wctx->pending[i].src] = i;
        if (wctx->pending[i].tensor) {
            pending_by_name[ggml_get_name(wctx->pending[i].tensor)] = i;
        }
    }

    int        merged         = 0;
    const bool verify         = getenv("MM3_ADAPTER_VERIFY") != nullptr;
    int        verify_changed = 0;

    for (const auto & kv : groups) {
        const std::string &                gguf_name = kv.first;
        const std::vector<MM3LoraFactor> & factors   = kv.second;

        if (!ws.exists(gguf_name.c_str())) {
            fprintf(stderr, "[MM3-Adapter] WARNING: %s not in the base model, skipping\n", gguf_name.c_str());
            continue;
        }

        enum ggml_type ttype;
        const void *   base_ptr = ws.data(gguf_name.c_str(), ttype);
        int            n_dims;
        int64_t        ne[4];
        ws.shape(gguf_name.c_str(), n_dims, ne);

        // Stack onto the running merged value when a prior adapter already
        // patched this tensor.
        auto pn = pending_by_name.find(gguf_name);
        if (pn != pending_by_name.end()) {
            base_ptr = wctx->pending[pn->second].src;
            if (wctx->pending[pn->second].tensor) {
                ttype = wctx->pending[pn->second].tensor->type;
            }
        }

        const int64_t in_feat  = ne[0];
        const int64_t out_feat = ne[1];
        const bool    is_ffn_in = mm3_str_ends(gguf_name, ".ffn_in.weight");
        const bool    sliced    = factors.size() > 1 || factors[0].slot != MM3_QKV_WHOLE;
        const int64_t sub_out   = sliced ? out_feat / 3 : out_feat;

        if (sliced && out_feat % 3 != 0) {
            fprintf(stderr, "[MM3-Adapter] WARNING: %s has out=%lld, not divisible by 3 for a qkv slice, skipping\n",
                    gguf_name.c_str(), (long long) out_feat);
            continue;
        }

        // Load every factor to F32 up front; the build lambda closes over these.
        struct LoadedFactor {
            std::vector<float> a, b;
            int64_t            rank = 0;
            float              scaling = 1.0f;
            MM3QkvSlot         slot = MM3_QKV_WHOLE;
        };
        std::vector<LoadedFactor> loaded;
        bool                      bad = false;

        for (const MM3LoraFactor & f : factors) {
            // safetensors shapes are PyTorch order: A is [rank, in], B is [out, rank].
            if (f.a->n_dims != 2 || f.b->n_dims != 2) {
                fprintf(stderr, "[MM3-Adapter] WARNING: %s factors are not 2-D, skipping\n", gguf_name.c_str());
                bad = true;
                break;
            }
            LoadedFactor lf;
            lf.rank = f.a->shape[0];
            lf.slot = f.slot;

            const int64_t a_in  = f.a->shape[1];
            const int64_t b_out = f.b->shape[0];
            const int64_t want_out = (f.slot == MM3_QKV_WHOLE) ? out_feat : sub_out;
            if (a_in != in_feat || b_out != want_out || f.b->shape[1] != lf.rank) {
                fprintf(stderr,
                        "[MM3-Adapter] WARNING: %s shape mismatch (A [%lld,%lld], B [%lld,%lld] vs base in=%lld out=%lld), skipping\n",
                        gguf_name.c_str(), (long long) lf.rank, (long long) a_in, (long long) b_out,
                        (long long) f.b->shape[1], (long long) in_feat, (long long) want_out);
                bad = true;
                break;
            }

            lf.a.resize((size_t) (lf.rank * in_feat));
            lf.b.resize((size_t) (want_out * lf.rank));
            if (!adapter_to_f32(st_data(st, *f.a), lf.a.data(), lf.rank * in_feat, f.a->dtype) ||
                !adapter_to_f32(st_data(st, *f.b), lf.b.data(), want_out * lf.rank, f.b->dtype)) {
                fprintf(stderr, "[MM3-Adapter] WARNING: unsupported factor dtype for %s, skipping\n", gguf_name.c_str());
                bad = true;
                break;
            }

            // SwiGLU half swap — see the file header. B rows are the output
            // features, so swapping the two halves re-orders gate/value.
            if (is_ffn_in && swiglu_swap) {
                if (want_out % 2 != 0) {
                    fprintf(stderr, "[MM3-Adapter] WARNING: %s has an odd SwiGLU output, cannot swap halves, skipping\n",
                            gguf_name.c_str());
                    bad = true;
                    break;
                }
                const int64_t      half = want_out / 2;
                std::vector<float> sw((size_t) (want_out * lf.rank));
                memcpy(sw.data(), lf.b.data() + (size_t) (half * lf.rank), (size_t) (half * lf.rank) * sizeof(float));
                memcpy(sw.data() + (size_t) (half * lf.rank), lf.b.data(), (size_t) (half * lf.rank) * sizeof(float));
                lf.b.swap(sw);
            }

            lf.scaling = (f.alpha > 0.0f) ? (f.alpha / (float) lf.rank) : 1.0f;
            loaded.push_back(std::move(lf));
        }
        if (bad) {
            continue;
        }

        // delta = scaling * B @ A, per factor. For a sliced (diffusers-form)
        // qkv the three sub-deltas are concatenated along ne1 in q,k,v order,
        // with a zero block standing in for any projection the adapter did not
        // train — which is what makes a partial q-only LoRA merge correctly.
        auto build = [&](struct ggml_context * ctx) {
            // (tensor, data) pairs. The data pointer is captured EXPLICITLY
            // rather than re-derived at upload time: A is [rank, in] and B is
            // [out, rank], so on a SQUARE weight (attn_output is 2048x2048)
            // they have identical element counts and any size-based
            // disambiguation silently uploads A into B. That bug survived a
            // clean compile and only showed up as a zero-delta adapter changing
            // the audio — which is what MM3_ADAPTER_VERIFY exists to catch.
            std::vector<std::pair<struct ggml_tensor *, const std::vector<float> *>> uploads;
            struct ggml_tensor * blocks[3] = { nullptr, nullptr, nullptr };
            struct ggml_tensor * whole     = nullptr;
            struct ggml_tensor * tzero     = nullptr;

            for (const LoadedFactor & lf : loaded) {
                const int64_t o = (lf.slot == MM3_QKV_WHOLE) ? out_feat : sub_out;

                struct ggml_tensor * ta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_feat, lf.rank);
                struct ggml_tensor * tb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lf.rank, o);
                // PEFT rounds A and B through BF16 before the GEMM; match it
                // on the backend so our merge equals theirs bit for bit.
                struct ggml_tensor * ta_br = ggml_cast(ctx, ggml_cast(ctx, ta, GGML_TYPE_BF16), GGML_TYPE_F32);
                struct ggml_tensor * tb_br = ggml_cast(ctx, ggml_cast(ctx, tb, GGML_TYPE_BF16), GGML_TYPE_F32);
                struct ggml_tensor * ta_t  = ggml_cont(ctx, ggml_transpose(ctx, ta_br));
                struct ggml_tensor * d     = ggml_scale(ctx, ggml_mul_mat(ctx, ta_t, tb_br), lf.scaling);

                uploads.emplace_back(ta, &lf.a);
                uploads.emplace_back(tb, &lf.b);
                if (lf.slot == MM3_QKV_WHOLE) {
                    whole = d;
                } else {
                    blocks[(int) lf.slot] = d;
                }
            }

            struct ggml_tensor * tdelta = whole;
            if (!tdelta) {
                for (int s = 0; s < 3; s++) {
                    if (!blocks[s]) {
                        if (!tzero) {
                            tzero = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_feat, sub_out);
                        }
                        blocks[s] = tzero;
                    }
                }
                tdelta = ggml_concat(ctx, ggml_concat(ctx, blocks[0], blocks[1], 1), blocks[2], 1);
            }

            adapter_delta_build db;
            db.tdelta = tdelta;
            db.upload = [uploads, tzero, in_feat, sub_out]() {
                for (const auto & u : uploads) {
                    const std::vector<float> & src = *u.second;
                    GGML_ASSERT(ggml_nelements(u.first) == (int64_t) src.size());
                    ggml_backend_tensor_set(u.first, src.data(), 0, src.size() * sizeof(float));
                }
                if (tzero) {
                    std::vector<float> zeros((size_t) (in_feat * sub_out), 0.0f);
                    ggml_backend_tensor_set(tzero, zeros.data(), 0, zeros.size() * sizeof(float));
                }
            };
            return db;
        };

        // promote_f32 = false: the MM3 GGUF ships F16 and CUDA can encode
        // F32 -> F16, so the merged weight goes back in its native type and the
        // DiT's VRAM footprint is unchanged. Promoting would cost ~4.8 GB.
        const size_t base_nb = ggml_row_size(ttype, in_feat) * (size_t) out_feat;
        if (!adapter_merge_on_backend(wctx, pending_idx, base_ptr, ttype, in_feat, out_feat,
                                      /*ds=*/nullptr, scale, backend, gguf_name.c_str(), build,
                                      /*promote_f32=*/false)) {
            continue;
        }
        merged++;

        // MM3_ADAPTER_VERIFY=1: byte-compare the merged result against the base
        // it was built from. This is the check that caught the square-weight
        // A/B upload bug above, when a zero-delta adapter changed the audio.
        //
        // EXPECT A NON-ZERO COUNT even when everything is correct, and do not
        // "fix" it: `base + 0` normalises IEEE -0.0 to +0.0, which flips exactly
        // one byte (0x8000 -> 0x0000) per negative zero. Measured on
        // mm3-synth-f16.gguf the differing-byte count equals the -0.0 count
        // EXACTLY, tensor for tensor (attn_qkv 2, attn_output 4, ffn_in 16),
        // and the rendered audio is bit-identical. The real gate is the audio,
        // not this counter; this one localises a failure when the audio moves.
        if (verify) {
            auto vp = pending_idx.find(base_ptr);
            if (vp != pending_idx.end()) {
                const WeightCtx::PendingCopy & pc = wctx->pending[vp->second];
                if (pc.nbytes != base_nb) {
                    fprintf(stderr, "[MM3-Adapter] VERIFY %s: size changed %zu -> %zu (type promoted)\n",
                            gguf_name.c_str(), base_nb, pc.nbytes);
                    verify_changed++;
                } else if (memcmp(pc.src, base_ptr, base_nb) != 0) {
                    size_t ndiff = 0;
                    for (size_t bi = 0; bi < base_nb; bi++) {
                        if (((const uint8_t *) pc.src)[bi] != ((const uint8_t *) base_ptr)[bi]) {
                            ndiff++;
                        }
                    }
                    if (verify_changed < 3) {
                        fprintf(stderr, "[MM3-Adapter] VERIFY %s: %zu/%zu bytes differ\n",
                                gguf_name.c_str(), ndiff, base_nb);
                    }
                    verify_changed++;
                }
            }
        }
    }

    if (verify) {
        fprintf(stderr, "[MM3-Adapter] VERIFY: %d of %d merged tensor(s) differ from their base\n",
                verify_changed, merged);
    }

    return merged;
}

// Resolve an adapter path (PEFT directory or single safetensors file), open it,
// and merge. Returns the number of tensors patched, or -1 when the path could
// not be opened at all.
static int mm3_adapter_merge(WeightCtx *       wctx,
                             const GGUFModel & gf,
                             const char *      path,
                             float             scale,
                             ggml_backend_t    backend) {
    struct stat sb;
    if (stat(path, &sb) != 0) {
        fprintf(stderr, "[MM3-Adapter] path does not exist: %s\n", path);
        return -1;
    }

    std::string sf_path;
    std::string cfg_dir;
    if (S_ISDIR(sb.st_mode)) {
        sf_path = std::string(path) + "/adapter_model.safetensors";
        cfg_dir = path;
        if (stat(sf_path.c_str(), &sb) != 0) {
            fprintf(stderr, "[MM3-Adapter] no adapter_model.safetensors in %s\n", path);
            return -1;
        }
    } else {
        sf_path = path;
        size_t slash = sf_path.find_last_of("/\\");
        cfg_dir      = (slash == std::string::npos) ? "." : sf_path.substr(0, slash);
    }

    STFile st = {};
    if (!st_open(&st, sf_path.c_str())) {
        fprintf(stderr, "[MM3-Adapter] cannot open %s\n", sf_path.c_str());
        return -1;
    }

    Timer     t;
    const int merged = mm3_adapter_merge_st(wctx, gf, st, cfg_dir, scale, backend);
    st_close(&st);

    fprintf(stderr, "[MM3-Adapter] %s: merged %d tensor(s) at scale %.3f in %.1f ms\n",
            sf_path.c_str(), merged, scale, t.ms());
    return merged;
}
