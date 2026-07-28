#pragma once
// spike-dit2-graph.h — ROUND 2 DiT spike: UNFUSED load, F32 weight mirror,
// LoRA bank on separate q/k/v/o, and the full 32-layer trainable forward.
//
// EVIDENCE CODE, NOT PRODUCT CODE.
//
// Round-1 gaps this closes (spike-verdict.md §7b):
//   * the load goes through dit_ggml_load()'s real `skip_fusion` path, so every
//     LoRA site is an individually addressable, PEFT-shaped projection;
//   * all 32 decoder layers, not one block;
//   * the real sliding-window / encoder-padding masks;
//   * a persistent F32 weight mirror (the BF16 buffer is released) instead of
//     in-graph ggml_cast — see the note on d2_build_mirror().
//
// Substitutions, all recorded with their equivalence class:
//   S1  flash-attn OFF -> dit_attn_f32 (soft_max_ext).  EXACT; cost is the
//       retained O(S^2 * Nh) score/softmax tensors (this is the memory story).
//   S2  BF16 base weights -> persistent F32 mirror.     EXACT UPCAST; needed
//       because ggml_out_prod (the backward of mul_mat w.r.t. its ACTIVATION
//       input, ggml.c:6598-6612) is F32-only.
//   S3  fused ggml_swiglu -> ggml_swiglu_split.         EXACT (unfused form).
//   S4  ggml_timestep_embedding front end -> temb/tproj precomputed host-side
//       from the REAL time_embed weights and uploaded as constants.  EXACT:
//       nothing upstream of temb is trainable, so it carries no gradient.

#include "dit-graph.h"
#include "dit.h"
#include "train/lm-common.h"
#include "train/preprocess-io.h"  // pm_mkdir_p / pm_write_atomic
#include "train/st-write.h"

#include <array>
#include <string>
#include <vector>

// ─── F32 weight mirror ──────────────────────────────────────────────────────
//
// Why a mirror and not in-graph ggml_cast: the cast output is out_prod's src0
// and is therefore RETAINED for the backward pass, so a per-step cast keeps the
// full F32 copy alive anyway *and* keeps the BF16 original resident — strictly
// worse than mirroring once and releasing the BF16 buffer.
//
// Two weight classes stay BF16, both provably safe:
//   * layers below `lora_lo` — no parameter beneath them, so no activation
//     gradient ever flows into their mul_mats (src1_needs_grads == false);
//   * cross-attention K/V in EVERY layer — their input is `enc`, which descends
//     only from the (frozen) condition_embedder and the encoder input, so it
//     never carries a gradient.

struct D2Mirror {
    ggml_context *        ctx       = nullptr;
    ggml_backend_buffer_t buf       = nullptr;
    size_t                bytes     = 0;  // total mirror bytes on the device
    size_t                bytes_f32 = 0;  // of which promoted to F32
    int                   n_f32     = 0;
    int                   n_keep    = 0;
};

static void d2_mirror_free(D2Mirror * M) {
    if (M->buf) {
        ggml_backend_buffer_free(M->buf);
    }
    if (M->ctx) {
        ggml_free(M->ctx);
    }
    *M = D2Mirror{};
}

struct D2Slot {
    ggml_tensor ** field;
    bool           promote;
};

static bool d2_build_mirror(DiTGGML * m, int lora_lo, bool mirror_ca_kv, D2Mirror * M, std::string * err) {
    const int L = m->cfg.n_layers;

    std::vector<D2Slot> slots;
    auto add = [&](ggml_tensor ** f, bool promote) {
        if (f && *f) {
            slots.push_back({ f, promote });
        }
    };

    // Globals. proj_in / cond_emb consume graph INPUTS (no gradient), so they
    // never need the F32 promotion; the output head does (its input descends
    // from every trainable layer).
    add(&m->proj_in_w, false);
    add(&m->proj_in_b, false);
    add(&m->cond_emb_w, false);
    add(&m->cond_emb_b, false);
    add(&m->norm_out, true);
    add(&m->out_scale_shift, true);
    add(&m->proj_out_w, true);
    add(&m->proj_out_b, true);
    add(&m->scalar_one, false);

    for (int i = 0; i < L; i++) {
        DiTGGMLLayer & ly = m->layers[i];
        const bool     tr = (i >= lora_lo);  // this layer carries gradients
        add(&ly.self_attn_norm, true);
        add(&ly.sa_q_proj, tr);
        add(&ly.sa_k_proj, tr);
        add(&ly.sa_v_proj, tr);
        add(&ly.sa_q_norm, true);
        add(&ly.sa_k_norm, true);
        add(&ly.sa_o_proj, tr);
        add(&ly.cross_attn_norm, true);
        add(&ly.ca_q_proj, tr);
        add(&ly.ca_k_proj, mirror_ca_kv && tr);
        add(&ly.ca_v_proj, mirror_ca_kv && tr);
        add(&ly.ca_q_norm, true);
        add(&ly.ca_k_norm, true);
        add(&ly.ca_o_proj, tr);
        add(&ly.mlp_norm, true);
        add(&ly.gate_proj, tr);
        add(&ly.up_proj, tr);
        add(&ly.down_proj, tr);
        add(&ly.scale_shift_table, true);
        // Fused slots must be NULL — the unfused load is the whole point.
        if (ly.sa_qkv || ly.sa_qk || ly.ca_qkv || ly.ca_kv || ly.gate_up) {
            char b[160];
            snprintf(b, sizeof(b), "layer %d loaded FUSED (sa_qkv=%d sa_qk=%d ca_qkv=%d ca_kv=%d gate_up=%d)", i,
                     ly.sa_qkv != nullptr, ly.sa_qk != nullptr, ly.ca_qkv != nullptr, ly.ca_kv != nullptr,
                     ly.gate_up != nullptr);
            *err = b;
            return false;
        }
        if (!ly.sa_q_proj || !ly.sa_k_proj || !ly.sa_v_proj || !ly.ca_q_proj || !ly.ca_k_proj || !ly.ca_v_proj ||
            !ly.gate_proj || !ly.up_proj) {
            char b[128];
            snprintf(b, sizeof(b), "layer %d is missing an unfused projection", i);
            *err = b;
            return false;
        }
    }

    {
        ggml_init_params p = { (slots.size() + 16) * ggml_tensor_overhead(), nullptr, true };
        M->ctx             = ggml_init(p);
    }
    if (!M->ctx) {
        *err = "cannot create the mirror context";
        return false;
    }

    std::vector<ggml_tensor *> dsts(slots.size(), nullptr);
    for (size_t i = 0; i < slots.size(); i++) {
        ggml_tensor * s  = *slots[i].field;
        const ggml_type ty = slots[i].promote ? GGML_TYPE_F32 : s->type;
        ggml_tensor *   d  = ggml_new_tensor_4d(M->ctx, ty, s->ne[0], s->ne[1], s->ne[2], s->ne[3]);
        ggml_set_name(d, s->name);
        dsts[i] = d;
        M->bytes += ggml_nbytes(d);
        if (ty == GGML_TYPE_F32 && s->type != GGML_TYPE_F32) {
            M->bytes_f32 += ggml_nbytes(d);
            M->n_f32++;
        } else {
            M->n_keep++;
        }
    }

    M->buf = ggml_backend_alloc_ctx_tensors(M->ctx, m->backend);
    if (!M->buf) {
        char b[192];
        snprintf(b, sizeof(b), "F32 weight mirror allocation failed (%.1f MB)", M->bytes / 1048576.0);
        *err = b;
        return false;
    }
    ggml_backend_buffer_set_usage(M->buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    // TRANSIENT PEAK: the mirror is allocated while the original weight buffer is
    // still resident, so building it this way costs base + mirror at once. A
    // product trainer must stream the mirror straight from the GGUF instead.
    fprintf(stderr, "[D2] mirror transient peak (base + mirror both resident): %zu MB device-wide\n",
            lm_vram_used_mb(m->backend));

    std::vector<uint8_t> raw;
    std::vector<float>   f32;
    for (size_t i = 0; i < slots.size(); i++) {
        ggml_tensor * s  = *slots[i].field;
        ggml_tensor * d  = dsts[i];
        const size_t  ne = (size_t) ggml_nelements(s);
        raw.resize(ggml_nbytes(s));
        ggml_backend_tensor_get(s, raw.data(), 0, raw.size());

        if (d->type == s->type) {
            ggml_backend_tensor_set(d, raw.data(), 0, raw.size());
        } else if (s->type == GGML_TYPE_BF16) {
            f32.resize(ne);
            const uint16_t * u = (const uint16_t *) raw.data();
            for (size_t j = 0; j < ne; j++) {
                const uint32_t b = (uint32_t) u[j] << 16;
                float          v;
                memcpy(&v, &b, 4);
                f32[j] = v;
            }
            ggml_backend_tensor_set(d, f32.data(), 0, ne * sizeof(float));
        } else if (s->type == GGML_TYPE_F16) {
            f32.resize(ne);
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), f32.data(), (int64_t) ne);
            ggml_backend_tensor_set(d, f32.data(), 0, ne * sizeof(float));
        } else {
            char b[256];
            snprintf(b, sizeof(b),
                     "base weight '%s' is %s — DiT training needs a BF16/F16/F32 base "
                     "(quantized bases cannot be trained: ggml_out_prod is F32-only)",
                     s->name, ggml_type_name(s->type));
            *err = b;
            return false;
        }
        *slots[i].field = d;
    }

    // Release the original weight buffer — the mirror fully replaces it.
    if (m->wctx.buffer) {
        ggml_backend_buffer_free(m->wctx.buffer);
        m->wctx.buffer = nullptr;
    }
    fprintf(stderr, "[D2] weight mirror: %d promoted to F32 (%.1f MB) + %d kept native, %.1f MB total "
                    "(BF16 buffer released)\n",
            M->n_f32, M->bytes_f32 / 1048576.0, M->n_keep, M->bytes / 1048576.0);
    return true;
}

// ─── LoRA bank ──────────────────────────────────────────────────────────────
//
// Sites: self-attn q/k/v/o + cross-attn q/o, per layer, all SEPARATE (the
// unfused load makes every one of them individually addressable).  Shapes are
// PEFT-on-disk shapes already: A is [in, r] in ggml == [r, in] row-major,
// B is [r, out] in ggml == [out, r] row-major, which is exactly what
// adapter-runtime.h:371 expects (in_feat == base ne0, out_feat == base ne1).

enum {
    D2_SA_Q = 0,
    D2_SA_K,
    D2_SA_V,
    D2_SA_O,
    D2_CA_Q,
    D2_CA_O,
    D2_NSLOTS
};

static const char * d2_slot_peft(int s) {
    switch (s) {
        case D2_SA_Q: return "self_attn.q_proj";
        case D2_SA_K: return "self_attn.k_proj";
        case D2_SA_V: return "self_attn.v_proj";
        case D2_SA_O: return "self_attn.o_proj";
        case D2_CA_Q: return "cross_attn.q_proj";
        default:      return "cross_attn.o_proj";
    }
}

struct D2Pair {
    ggml_tensor * A     = nullptr;  // [in, r]
    ggml_tensor * B     = nullptr;  // [r, out]
    float         scale = 1.0f;
};

struct D2Lora {
    ggml_context *                        ctx = nullptr;
    ggml_backend_buffer_t                 buf = nullptr;
    std::vector<std::array<D2Pair, D2_NSLOTS>> layers;  // indexed by (layer - lo)
    std::vector<ggml_tensor *>            params;
    int                                   lo = 0, hi = 0, rank = 0;
    float                                 alpha = 0.0f, scale = 1.0f;
    size_t                                n_params = 0;
};

static void d2_lora_free(D2Lora * L) {
    if (L->buf) {
        ggml_backend_buffer_free(L->buf);
    }
    if (L->ctx) {
        ggml_free(L->ctx);
    }
    *L = D2Lora{};
}

static ggml_tensor * d2_site_weight(DiTGGMLLayer * ly, int slot) {
    switch (slot) {
        case D2_SA_Q: return ly->sa_q_proj;
        case D2_SA_K: return ly->sa_k_proj;
        case D2_SA_V: return ly->sa_v_proj;
        case D2_SA_O: return ly->sa_o_proj;
        case D2_CA_Q: return ly->ca_q_proj;
        default:      return ly->ca_o_proj;
    }
}

static bool d2_lora_init(D2Lora * L, DiTGGML * m, int lo, int hi, int rank, float alpha, uint64_t seed, float b_sigma,
                         std::string * err) {
    L->lo    = lo;
    L->hi    = hi;
    L->rank  = rank;
    L->alpha = alpha;
    L->scale = alpha / (float) rank;
    L->layers.assign((size_t) (hi - lo), std::array<D2Pair, D2_NSLOTS>{});

    const int n_ten = (hi - lo) * D2_NSLOTS * 2;
    {
        ggml_init_params p = { (size_t) (n_ten + 8) * ggml_tensor_overhead(), nullptr, true };
        L->ctx             = ggml_init(p);
    }
    if (!L->ctx) {
        *err = "cannot create the LoRA context";
        return false;
    }

    for (int l = lo; l < hi; l++) {
        for (int s = 0; s < D2_NSLOTS; s++) {
            ggml_tensor * w = d2_site_weight(&m->layers[l], s);
            if (!w) {
                char b[128];
                snprintf(b, sizeof(b), "layer %d slot %s has no weight", l, d2_slot_peft(s));
                *err = b;
                return false;
            }
            D2Pair & pr = L->layers[(size_t) (l - lo)][(size_t) s];
            pr.A        = ggml_new_tensor_2d(L->ctx, GGML_TYPE_F32, w->ne[0], rank);
            pr.B        = ggml_new_tensor_2d(L->ctx, GGML_TYPE_F32, rank, w->ne[1]);
            pr.scale    = L->scale;
            char nm[96];
            snprintf(nm, sizeof(nm), "L%d.%s.lora_A", l, d2_slot_peft(s));
            ggml_set_name(pr.A, nm);
            snprintf(nm, sizeof(nm), "L%d.%s.lora_B", l, d2_slot_peft(s));
            ggml_set_name(pr.B, nm);
            ggml_set_param(pr.A);
            ggml_set_param(pr.B);
            L->params.push_back(pr.A);
            L->params.push_back(pr.B);
        }
    }

    L->buf = ggml_backend_alloc_ctx_tensors(L->ctx, m->backend);
    if (!L->buf) {
        *err = "LoRA parameter buffer allocation failed";
        return false;
    }

    LmRng rng;
    lm_rng_seed(&rng, seed);
    std::vector<float> a, b;
    for (size_t i = 0; i < L->layers.size(); i++) {
        for (int s = 0; s < D2_NSLOTS; s++) {
            D2Pair & pr = L->layers[i][(size_t) s];
            a.assign((size_t) ggml_nelements(pr.A), 0.0f);
            b.assign((size_t) ggml_nelements(pr.B), 0.0f);
            lm_rng_fill_normal(&rng, a, 1.0f / sqrtf((float) pr.A->ne[0]));
            if (b_sigma > 0.0f) {
                lm_rng_fill_normal(&rng, b, b_sigma);
            }
            ggml_backend_tensor_set(pr.A, a.data(), 0, a.size() * sizeof(float));
            ggml_backend_tensor_set(pr.B, b.data(), 0, b.size() * sizeof(float));
            L->n_params += a.size() + b.size();
        }
    }
    fprintf(stderr, "[D2] LoRA layers %d..%d rank %d alpha %.0f, %d sites/layer -> %zu params (%.1f MB f32)\n", lo, hi,
            rank, (double) alpha, D2_NSLOTS, L->n_params, L->n_params * 4.0 / 1048576.0);
    return true;
}

// ─── trainable forward ──────────────────────────────────────────────────────

static ggml_tensor * d2_lin(ggml_context * ctx, ggml_tensor * w, const D2Pair * p, ggml_tensor * x) {
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    if (p && p->A && p->B) {
        ggml_tensor * t = ggml_scale(ctx, ggml_mul_mat(ctx, p->A, x), p->scale);
        y               = ggml_add(ctx, y, ggml_mul_mat(ctx, p->B, t));
    }
    return y;
}

struct D2Inputs {
    ggml_tensor * t_input = nullptr;  // [in_ch, T]      concat(context, xt)
    ggml_tensor * t_enc   = nullptr;  // [enc_H, enc_S]
    ggml_tensor * t_pos   = nullptr;  // [S] i32
    ggml_tensor * t_temb  = nullptr;  // [H]   precomputed
    ggml_tensor * t_tproj = nullptr;  // [6H]  precomputed
    ggml_tensor * t_sa    = nullptr;  // [S, S] f16 sliding-window mask
    ggml_tensor * t_ca    = nullptr;  // [enc_S, S] f16 encoder-padding mask
    ggml_tensor * t_vtgt  = nullptr;  // [Oc, T] velocity target
};

// One decoder layer, N = 1, unfused, manual attention. hidden: [H, S]
static ggml_tensor * d2_layer(ggml_context * ctx, DiTGGML * m, int li, ggml_tensor * hidden, ggml_tensor * enc,
                              const D2Inputs & in, const D2Pair * pairs, int S, int enc_S) {
    const DiTGGMLConfig & c  = m->cfg;
    DiTGGMLLayer *        ly = &m->layers[li];
    const int             H = c.hidden_size, D = c.head_dim, Nh = c.n_heads, Nkv = c.n_kv_heads;
    const float           scale = 1.0f / sqrtf((float) D);

    ggml_tensor * ss_flat = ggml_reshape_1d(ctx, ly->scale_shift_table, 6 * H);
    ggml_tensor * adaln   = ggml_add(ctx, ss_flat, in.t_tproj);  // [6H]
    const size_t  Hb      = (size_t) H * sizeof(float);
    ggml_tensor * shift_sa = ggml_view_1d(ctx, adaln, H, 0 * Hb);
    ggml_tensor * scale_sa = ggml_view_1d(ctx, adaln, H, 1 * Hb);
    ggml_tensor * gate_sa  = ggml_view_1d(ctx, adaln, H, 2 * Hb);
    ggml_tensor * shift_ff = ggml_view_1d(ctx, adaln, H, 3 * Hb);
    ggml_tensor * scale_ff = ggml_view_1d(ctx, adaln, H, 4 * Hb);
    ggml_tensor * gate_ff  = ggml_view_1d(ctx, adaln, H, 5 * Hb);

    // ── self-attention ──
    ggml_tensor * res = hidden;
    ggml_tensor * nsa = dit_ggml_rms_norm_weighted(ctx, hidden, ly->self_attn_norm, c.rms_norm_eps);
    nsa               = dit_ggml_adaln(ctx, nsa, scale_sa, shift_sa, m->scalar_one);

    ggml_tensor * q = d2_lin(ctx, ly->sa_q_proj, pairs ? &pairs[D2_SA_Q] : nullptr, nsa);
    ggml_tensor * k = d2_lin(ctx, ly->sa_k_proj, pairs ? &pairs[D2_SA_K] : nullptr, nsa);
    ggml_tensor * v = d2_lin(ctx, ly->sa_v_proj, pairs ? &pairs[D2_SA_V] : nullptr, nsa);

    q = ggml_reshape_3d(ctx, q, D, Nh, S);
    k = ggml_reshape_3d(ctx, k, D, Nkv, S);
    v = ggml_reshape_3d(ctx, v, D, Nkv, S);
    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, c.rms_norm_eps), dit_ggml_f32(ctx, ly->sa_q_norm));
    k = ggml_mul(ctx, ggml_rms_norm(ctx, k, c.rms_norm_eps), dit_ggml_f32(ctx, ly->sa_k_norm));
    q = ggml_rope_ext(ctx, q, in.t_pos, NULL, D, 2 /*NEOX*/, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, in.t_pos, NULL, D, 2, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    // layer_type 0 = sliding window, 1 = full attention (dit.h:656)
    ggml_tensor * sa_mask = (ly->layer_type == 0) ? in.t_sa : nullptr;
    ggml_tensor * attn    = dit_attn_f32(ctx, q, k, v, sa_mask, scale);
    attn                  = ggml_reshape_2d(ctx, attn, Nh * D, S);
    ggml_tensor * sa_out  = d2_lin(ctx, ly->sa_o_proj, pairs ? &pairs[D2_SA_O] : nullptr, attn);
    hidden                = dit_ggml_gated_add(ctx, res, sa_out, gate_sa);

    // ── cross-attention (no gate, plain residual) ──
    ggml_tensor * nca = dit_ggml_rms_norm_weighted(ctx, hidden, ly->cross_attn_norm, c.rms_norm_eps);
    ggml_tensor * cq  = d2_lin(ctx, ly->ca_q_proj, pairs ? &pairs[D2_CA_Q] : nullptr, nca);
    ggml_tensor * ck  = ggml_mul_mat(ctx, ly->ca_k_proj, enc);
    ggml_tensor * cv  = ggml_mul_mat(ctx, ly->ca_v_proj, enc);
    cq                = ggml_permute(ctx, ggml_reshape_3d(ctx, cq, D, Nh, S), 0, 2, 1, 3);
    ck                = ggml_permute(ctx, ggml_reshape_3d(ctx, ck, D, Nkv, enc_S), 0, 2, 1, 3);
    cv                = ggml_permute(ctx, ggml_reshape_3d(ctx, cv, D, Nkv, enc_S), 0, 2, 1, 3);
    // NB the inference graph normalises BEFORE the permute; rms_norm acts on ne0
    // (= D, the head dim) in both layouts, so this is the same operation.
    cq                = ggml_mul(ctx, ggml_rms_norm(ctx, cq, c.rms_norm_eps), dit_ggml_f32(ctx, ly->ca_q_norm));
    ck                = ggml_mul(ctx, ggml_rms_norm(ctx, ck, c.rms_norm_eps), dit_ggml_f32(ctx, ly->ca_k_norm));
    ggml_tensor * cattn = dit_attn_f32(ctx, cq, ck, cv, in.t_ca, scale);
    cattn               = ggml_reshape_2d(ctx, cattn, Nh * D, S);
    hidden = ggml_add(ctx, hidden, d2_lin(ctx, ly->ca_o_proj, pairs ? &pairs[D2_CA_O] : nullptr, cattn));

    // ── MLP ──
    res               = hidden;
    ggml_tensor * nff = dit_ggml_rms_norm_weighted(ctx, hidden, ly->mlp_norm, c.rms_norm_eps);
    nff               = dit_ggml_adaln(ctx, nff, scale_ff, shift_ff, m->scalar_one);
    ggml_tensor * g   = ggml_mul_mat(ctx, ly->gate_proj, nff);
    ggml_tensor * u   = ggml_mul_mat(ctx, ly->up_proj, nff);
    ggml_tensor * ff  = ggml_swiglu_split(ctx, g, u);  // fused ggml_swiglu has NO backward
    ggml_tensor * fo  = ggml_mul_mat(ctx, ly->down_proj, ff);
    return dit_ggml_gated_add(ctx, res, fo, gate_ff);
}

// Full trainable forward: proj_in -> cond_emb -> L layers -> AdaLN out -> proj_out.
// Returns the velocity prediction [Oc, T].
static ggml_tensor * d2_build_forward(ggml_context * ctx, DiTGGML * m, D2Lora * L, const D2Inputs & in, int T,
                                      int enc_S) {
    const DiTGGMLConfig & c = m->cfg;
    const int             S = T / c.patch_size, H = c.hidden_size;

    ggml_tensor * patched = ggml_reshape_2d(ctx, in.t_input, (int64_t) c.in_channels * c.patch_size, S);
    ggml_tensor * hidden  = ggml_add(ctx, ggml_mul_mat(ctx, m->proj_in_w, patched), m->proj_in_b);
    ggml_tensor * enc     = ggml_add(ctx, ggml_mul_mat(ctx, m->cond_emb_w, in.t_enc), m->cond_emb_b);

    for (int i = 0; i < c.n_layers; i++) {
        const D2Pair * pairs =
            (i >= L->lo && i < L->hi) ? L->layers[(size_t) (i - L->lo)].data() : nullptr;
        hidden = d2_layer(ctx, m, i, hidden, enc, in, pairs, S, enc_S);
    }

    ggml_tensor * oss_flat  = ggml_reshape_1d(ctx, m->out_scale_shift, 2 * H);
    ggml_tensor * out_shift = ggml_add(ctx, ggml_view_1d(ctx, oss_flat, H, 0), in.t_temb);
    ggml_tensor * out_scale = ggml_add(ctx, ggml_view_1d(ctx, oss_flat, H, (size_t) H * sizeof(float)), in.t_temb);
    ggml_tensor * nout      = dit_ggml_rms_norm_weighted(ctx, hidden, m->norm_out, c.rms_norm_eps);
    nout                    = dit_ggml_adaln(ctx, nout, out_scale, out_shift, m->scalar_one);
    ggml_tensor * out       = ggml_add(ctx, ggml_mul_mat(ctx, m->proj_out_w, nout), m->proj_out_b);
    return ggml_reshape_2d(ctx, out, c.out_channels, T);  // [Oc, T]
}

// ─── PEFT export (round-1 gap #12) ──────────────────────────────────────────
//
// Round 1 put its LoRA on the FUSED sa_qkv/ca_qkv, which is not expressible as a
// PEFT q/k/v adapter at all. With the unfused load the shapes are already right:
//   ggml A [in, r] == row-major [r, in]  == PEFT lora_A
//   ggml B [r, out] == row-major [out, r] == PEFT lora_B
// (adapter-runtime.h:371 checks in_feat == base.ne0 and out_feat == base.ne1.)
// Writing a PEFT dir here turns "the shape looks right" into a load-back test:
// re-running the spike with this directory as --stub must report
// "N deltas precomputed (0 skipped)".
static bool d2_export_peft(const D2Lora * L, const std::string & dir, const std::string & base_name) {
    if (!pm_mkdir_p(dir)) {
        fprintf(stderr, "[D2] export: cannot create %s\n", dir.c_str());
        return false;
    }
    std::vector<std::vector<float>> host;
    std::vector<STWTensor>          ts;
    host.reserve(L->params.size());
    ts.reserve(L->params.size());
    for (int l = L->lo; l < L->hi; l++) {
        for (int s = 0; s < D2_NSLOTS; s++) {
            const D2Pair & pr = L->layers[(size_t) (l - L->lo)][(size_t) s];
            for (int ab = 0; ab < 2; ab++) {
                ggml_tensor * t = ab ? pr.B : pr.A;
                host.emplace_back((size_t) ggml_nelements(t));
                ggml_backend_tensor_get(t, host.back().data(), 0, host.back().size() * sizeof(float));
                char nm[192];
                snprintf(nm, sizeof(nm), "base_model.model.layers.%d.%s.lora_%c.weight", l, d2_slot_peft(s),
                         ab ? 'B' : 'A');
                // row-major shape = (ne1, ne0)
                ts.push_back({ nm, { t->ne[1], t->ne[0] }, host.back().data() });
            }
        }
    }
    const std::string st_path = dir + "/adapter_model.safetensors";
    std::vector<std::pair<std::string, std::string>> md;
    md.push_back({ "producer", "ace-train spike dit2" });
    if (!st_write_file(st_path.c_str(), ts, md, STW_F32)) {
        return false;
    }
    char cfg[1024];
    snprintf(cfg, sizeof(cfg),
             "{\n  \"peft_type\": \"LORA\",\n  \"task_type\": null,\n"
             "  \"base_model_name_or_path\": \"%s\",\n  \"r\": %d,\n  \"lora_alpha\": %d,\n"
             "  \"lora_dropout\": 0.0,\n  \"bias\": \"none\",\n  \"fan_in_fan_out\": false,\n"
             "  \"target_modules\": [\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\"]\n}\n",
             base_name.c_str(), L->rank, (int) L->alpha);
    if (!pm_write_atomic(dir + "/adapter_config.json", cfg)) {
        return false;
    }
    fprintf(stderr, "[D2] exported PEFT adapter: %zu tensors -> %s (r=%d alpha=%.0f)\n", ts.size(), dir.c_str(),
            L->rank, (double) L->alpha);
    return true;
}
