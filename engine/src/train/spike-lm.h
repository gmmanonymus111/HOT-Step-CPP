#pragma once
// spike-lm.h — trainable Qwen3 LM forward graph for the Phase-0 spike.
//
// This is the "training-only forward function" the recon brief called for:
//   * NO KV cache        (GGML_OP_SET_ROWS has no backward)
//   * NO flash attention (GGML_OP_FLASH_ATTN_EXT has no backward)
//   * NO fused QKV / gate_up (fused ggml_swiglu has no backward; fused
//     projections are not individually LoRA-addressable)
//   * base weights optionally cast to F32 in-graph, because
//     ggml_out_prod (the backward of mul_mat w.r.t. its ACTIVATION input)
//     is F32-only on CUDA and GGML_ABORTs for BF16/F16 on CPU.

#include "qwen3-lm.h"
#include "spike-common.h"

// ─── trainable LoRA bank ────────────────────────────────────────────────────

struct SpikeLora {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    QwLoraLayer           layers[QW3LM_MAX_LAYERS];
    std::vector<ggml_tensor *> params;  // every tensor marked GGML_TENSOR_FLAG_PARAM
    int   rank  = 0;
    float scale = 1.0f;
};

static void spike_lora_free(SpikeLora * L) {
    if (L->buf) ggml_backend_buffer_free(L->buf);
    if (L->ctx) ggml_free(L->ctx);
    L->buf = nullptr;
    L->ctx = nullptr;
    L->params.clear();
}

// out_dim of each slot for a given config.
static void spike_slot_dims(const Qwen3LMConfig & c, int slot, int * in_dim, int * out_dim) {
    const int H = c.hidden_size, D = c.head_dim, Nh = c.n_heads, Nkv = c.n_kv_heads, F = c.intermediate_size;
    switch (slot) {
        case QW_LORA_Q:    *in_dim = H;      *out_dim = Nh * D;  break;
        case QW_LORA_K:    *in_dim = H;      *out_dim = Nkv * D; break;
        case QW_LORA_V:    *in_dim = H;      *out_dim = Nkv * D; break;
        case QW_LORA_O:    *in_dim = Nh * D; *out_dim = H;       break;
        case QW_LORA_GATE: *in_dim = H;      *out_dim = F;       break;
        case QW_LORA_UP:   *in_dim = H;      *out_dim = F;       break;
        default:           *in_dim = F;      *out_dim = H;       break;  // DOWN
    }
}

// layer_lo..layer_hi (exclusive) get LoRA slots. b_sigma>0 breaks the PEFT
// B=0 init (which makes dL/dA identically zero at step 0 by construction).
static bool spike_lora_init(SpikeLora * L, Qwen3LM * lm, int layer_lo, int layer_hi, int rank, float alpha,
                            uint64_t seed, float b_sigma) {
    const Qwen3LMConfig & c = lm->cfg;
    L->rank  = rank;
    L->scale = alpha / (float) rank;

    const int n_lay = layer_hi - layer_lo;
    const int n_ten = n_lay * QW_LORA_NSLOTS * 2;
    ggml_init_params p = { (size_t) (n_ten + 8) * ggml_tensor_overhead(), nullptr, true };
    L->ctx = ggml_init(p);

    for (int i = 0; i < QW3LM_MAX_LAYERS; i++) {
        L->layers[i] = QwLoraLayer{};
    }

    for (int l = layer_lo; l < layer_hi; l++) {
        for (int s = 0; s < QW_LORA_NSLOTS; s++) {
            int in_dim = 0, out_dim = 0;
            spike_slot_dims(c, s, &in_dim, &out_dim);
            QwLoraPair & pr = L->layers[l].p[s];
            pr.A = ggml_new_tensor_2d(L->ctx, GGML_TYPE_F32, in_dim, rank);
            pr.B = ggml_new_tensor_2d(L->ctx, GGML_TYPE_F32, rank, out_dim);
            pr.scale = L->scale;
            char nm[96];
            snprintf(nm, sizeof(nm), "L%d.s%d.lora_A", l, s); ggml_set_name(pr.A, nm);
            snprintf(nm, sizeof(nm), "L%d.s%d.lora_B", l, s); ggml_set_name(pr.B, nm);
            ggml_set_param(pr.A);
            ggml_set_param(pr.B);
            L->params.push_back(pr.A);
            L->params.push_back(pr.B);
        }
    }

    L->buf = ggml_backend_alloc_ctx_tensors(L->ctx, lm->backend);
    if (!L->buf) {
        fprintf(stderr, "[SpikeLoRA] FATAL: buffer alloc failed\n");
        return false;
    }

    SpikeRng rng = { seed };
    size_t n_par = 0;
    for (int l = layer_lo; l < layer_hi; l++) {
        for (int s = 0; s < QW_LORA_NSLOTS; s++) {
            QwLoraPair & pr = L->layers[l].p[s];
            std::vector<float> a((size_t) ggml_nelements(pr.A));
            std::vector<float> b((size_t) ggml_nelements(pr.B));
            srng_fill_normal(&rng, a, 1.0f / sqrtf((float) pr.A->ne[0]));
            if (b_sigma > 0.0f) { srng_fill_normal(&rng, b, b_sigma); }
            else                { std::fill(b.begin(), b.end(), 0.0f); }
            spike_set(pr.A, a);
            spike_set(pr.B, b);
            n_par += a.size() + b.size();
        }
        lm->layers[l].lora = &L->layers[l];
    }
    fprintf(stderr, "[SpikeLoRA] layers %d..%d, rank %d, alpha %.0f -> %zu trainable params (%.1f MB f32)\n",
            layer_lo, layer_hi, rank, alpha, n_par, (double) n_par * 4.0 / (1024 * 1024));
    return true;
}

// ─── training forward graph ─────────────────────────────────────────────────

struct SpikeFwdOpts {
    bool cast_base_f32 = true;  // workaround for out_prod F32-only
    int  n_casts       = 0;     // out: how many ggml_cast nodes were emitted
};

static ggml_tensor * spike_linear(ggml_context * ctx, ggml_tensor * w, const QwLoraPair * pr, ggml_tensor * x,
                                  SpikeFwdOpts * o) {
    ggml_tensor * ww = w;
    if (o->cast_base_f32 && w->type != GGML_TYPE_F32) {
        ww = ggml_cast(ctx, w, GGML_TYPE_F32);
        o->n_casts++;
    }
    ggml_tensor * y = ggml_mul_mat(ctx, ww, x);
    if (pr && pr->A && pr->B) {
        ggml_tensor * t = ggml_mul_mat(ctx, pr->A, x);
        t               = ggml_scale(ctx, t, pr->scale);
        y               = ggml_add(ctx, y, ggml_mul_mat(ctx, pr->B, t));
    }
    return y;
}

static ggml_tensor * spike_rms(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), qwen3_f32(ctx, w));
}

// One transformer layer, cache-free, manual attention. hidden: [H, S]
static ggml_tensor * spike_lm_layer(ggml_context * ctx, const Qwen3LMConfig & c, Qwen3Layer * ly, ggml_tensor * hidden,
                                    ggml_tensor * positions, ggml_tensor * mask, int S, SpikeFwdOpts * o) {
    const int D = c.head_dim, Nh = c.n_heads, Nkv = c.n_kv_heads;
    const QwLoraLayer * ll = ly->lora;

    ggml_tensor * x = spike_rms(ctx, hidden, ly->input_layernorm, c.rms_norm_eps);

    ggml_tensor * q = spike_linear(ctx, ly->q_proj, qwen3_lora_slot(ll, QW_LORA_Q), x, o);
    ggml_tensor * k = spike_linear(ctx, ly->k_proj, qwen3_lora_slot(ll, QW_LORA_K), x, o);
    ggml_tensor * v = spike_linear(ctx, ly->v_proj, qwen3_lora_slot(ll, QW_LORA_V), x, o);

    q = ggml_reshape_3d(ctx, q, D, Nh, S);
    k = ggml_reshape_3d(ctx, k, D, Nkv, S);
    v = ggml_reshape_3d(ctx, v, D, Nkv, S);

    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, c.rms_norm_eps), qwen3_f32(ctx, ly->q_norm));
    k = ggml_mul(ctx, ggml_rms_norm(ctx, k, c.rms_norm_eps), qwen3_f32(ctx, ly->k_norm));

    q = ggml_rope_ext(ctx, q, positions, NULL, D, 2, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, positions, NULL, D, 2, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    q = ggml_permute(ctx, q, 0, 2, 1, 3);  // [D, S, Nh]
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    // manual F32 attention (flash_attn_ext has no backward)
    ggml_tensor * attn = qwen3_attn_f32(ctx, q, k, v, mask, 1.0f / sqrtf((float) D));
    attn               = ggml_reshape_2d(ctx, attn, Nh * D, S);

    ggml_tensor * ao = spike_linear(ctx, ly->o_proj, qwen3_lora_slot(ll, QW_LORA_O), attn, o);
    hidden           = ggml_add(ctx, hidden, ao);

    ggml_tensor * xm  = spike_rms(ctx, hidden, ly->post_attn_layernorm, c.rms_norm_eps);
    ggml_tensor * g   = spike_linear(ctx, ly->gate_proj, qwen3_lora_slot(ll, QW_LORA_GATE), xm, o);
    ggml_tensor * u   = spike_linear(ctx, ly->up_proj, qwen3_lora_slot(ll, QW_LORA_UP), xm, o);
    ggml_tensor * ff  = ggml_swiglu_split(ctx, g, u);  // split form: fused ggml_swiglu has NO backward
    ggml_tensor * dn  = spike_linear(ctx, ly->down_proj, qwen3_lora_slot(ll, QW_LORA_DOWN), ff, o);
    return ggml_add(ctx, hidden, dn);
}

// ─── mask helper: causal [S, S] f32, 0 / -inf ───────────────────────────────

static std::vector<float> spike_causal_mask(int S) {
    std::vector<float> m((size_t) S * S, 0.0f);
    for (int i = 0; i < S; i++) {      // query row i (ne1)
        for (int j = 0; j < S; j++) {  // key col j (ne0)
            m[(size_t) i * S + j] = (j <= i) ? 0.0f : -INFINITY;
        }
    }
    return m;
}
