#pragma once
// spike-s4.h — Rung 4: ONE real DiT decoder block, real preprocessed sample,
// manual attention, trainable LoRA on the attention projections.
//
// The op sequence mirrors dit_ggml_build_layer exactly (same helpers:
// dit_ggml_rms_norm_weighted / dit_ggml_adaln / dit_ggml_gated_add /
// dit_attn_f32), with two deliberate substitutions, both recorded:
//   * flash-attn forced OFF  (GGML_OP_FLASH_ATTN_EXT has no backward)
//   * base weights cast to F32 in-graph (ggml_out_prod is F32-only)
// and one omission: the timestep-embedding front end (ggml_timestep_embedding
// has no backward) is replaced by a constant tproj upload — legitimate because
// tproj is frozen input-side data with nothing trainable upstream of it.

#include "dit-graph.h"
#include "dit.h"
#include "safetensors.h"
#include "spike-common.h"

struct S4Lora {
    ggml_context *             ctx = nullptr;
    ggml_backend_buffer_t      buf = nullptr;
    std::vector<ggml_tensor *> params;
    std::vector<QwLoraPair>    pairs;  // one per site
    std::vector<const char *>  names;
};

static ggml_tensor * s4_linear(ggml_context * ctx, ggml_tensor * w, const QwLoraPair * p, ggml_tensor * x,
                               int * n_casts) {
    ggml_tensor * ww = w;
    if (w->type != GGML_TYPE_F32) { ww = ggml_cast(ctx, w, GGML_TYPE_F32); (*n_casts)++; }
    ggml_tensor * y = ggml_mul_mat(ctx, ww, x);
    if (p && p->A && p->B) {
        ggml_tensor * t = ggml_scale(ctx, ggml_mul_mat(ctx, p->A, x), p->scale);
        y = ggml_add(ctx, y, ggml_mul_mat(ctx, p->B, t));
    }
    return y;
}

static ggml_opt_optimizer_params s4_opt_pars(void * ud) { return *(ggml_opt_optimizer_params *) ud; }

static int spike_s4(const char * dit_path, const char * sample_path, int S, int enc_S, int rank) {
    fprintf(stderr, "\n=== S4: one REAL DiT decoder block (%s) S=%d enc_S=%d rank=%d ===\n", dit_path, S, enc_S, rank);

    DiTGGML m;
    if (!dit_ggml_load(&m, dit_path)) { fprintf(stderr, "[S4] FATAL: DiT load failed\n"); return 1; }
    m.use_flash_attn = false;  // SUBSTITUTION 1: flash-attn has no backward
    DiTGGMLConfig & c = m.cfg;
    const int H = c.hidden_size, D = c.head_dim, Nh = c.n_heads, Nkv = c.n_kv_heads;
    fprintf(stderr, "[S4] cfg: H=%d FFN=%d Nh=%d Nkv=%d D=%d L=%d in_ch=%d patch=%d\n", H, c.intermediate_size, Nh,
            Nkv, D, c.n_layers, c.in_channels, c.patch_size);
    fprintf(stderr, "[S4] VRAM after DiT load: %zu MB\n", spike_vram_used_mb(m.backend));

    DiTGGMLLayer * ly = &m.layers[0];
    const bool fused_sa = ly->sa_qkv || ly->sa_qk;
    const bool fused_ca = ly->ca_qkv || ly->ca_kv;
    const bool fused_mlp = ly->gate_up != nullptr;
    fprintf(stderr, "[S4] layer0 fusion: sa_qkv=%d sa_qk=%d ca_qkv=%d ca_kv=%d gate_up=%d  convrot=%d\n",
            ly->sa_qkv != nullptr, ly->sa_qk != nullptr, ly->ca_qkv != nullptr, ly->ca_kv != nullptr,
            ly->gate_up != nullptr, (int) m.convrot.active);

    // ── real preprocessed sample ────────────────────────────────────────
    STFile st;
    if (!st_open(&st, sample_path)) { fprintf(stderr, "[S4] FATAL: cannot open %s\n", sample_path); return 1; }
    ggml_type ty_lat = GGML_TYPE_F32, ty_enc = GGML_TYPE_F32;
    const float * lat = (const float *) st_find_data(st, "target_latents", &ty_lat);
    const float * enc = (const float *) st_find_data(st, "encoder_hidden_states", &ty_enc);
    const STEntry * e_lat = st_find(st, "target_latents");
    const STEntry * e_enc = st_find(st, "encoder_hidden_states");
    if (!lat || !enc || ty_lat != GGML_TYPE_F32 || ty_enc != GGML_TYPE_F32) {
        fprintf(stderr, "[S4] FATAL: sample missing F32 target_latents / encoder_hidden_states\n"); return 1;
    }
    const int n_frames = (int) e_lat->shape[0], n_ch = (int) e_lat->shape[1];
    const int n_enc    = (int) e_enc->shape[0], enc_H = (int) e_enc->shape[1];
    fprintf(stderr, "[S4] sample: target_latents [%d,%d]  encoder_hidden_states [%d,%d]\n", n_frames, n_ch, n_enc, enc_H);
    if (enc_S > n_enc) enc_S = n_enc;
    if (S * c.patch_size > n_frames) S = n_frames / c.patch_size;

    // ── static tensors ──────────────────────────────────────────────────
    ggml_context * ctxs;
    { ggml_init_params p = { 16 * ggml_tensor_overhead(), nullptr, true }; ctxs = ggml_init(p); }
    const int64_t proj_in_dim = m.proj_in_w->ne[0];
    ggml_tensor * t_x    = ggml_new_tensor_2d(ctxs, GGML_TYPE_F32, proj_in_dim, S);   // patchified latent
    ggml_tensor * t_enc  = ggml_new_tensor_2d(ctxs, GGML_TYPE_F32, enc_H, enc_S);
    ggml_tensor * t_pos  = ggml_new_tensor_1d(ctxs, GGML_TYPE_I32, S);
    ggml_tensor * t_tpj  = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 6 * H);            // constant (see header note)
    ggml_tensor * t_tgt  = ggml_new_tensor_2d(ctxs, GGML_TYPE_F32, H, S);
    ggml_set_input(t_x); ggml_set_input(t_enc); ggml_set_input(t_pos); ggml_set_input(t_tpj); ggml_set_input(t_tgt);
    ggml_backend_buffer_t bufs = ggml_backend_alloc_ctx_tensors(ctxs, m.backend);
    if (!bufs) { fprintf(stderr, "[S4] FATAL: static alloc failed\n"); return 1; }

    {   // patchify: token t = frames [t*P .. t*P+P), channel-major; zero-pad the rest
        std::vector<float> x((size_t) proj_in_dim * S, 0.0f);
        const int P = c.patch_size;
        for (int t = 0; t < S; t++) {
            for (int q = 0; q < P; q++) {
                for (int ch = 0; ch < n_ch; ch++) {
                    size_t di = (size_t) t * proj_in_dim + (size_t) q * n_ch + ch;
                    if ((size_t) (q * n_ch + ch) >= (size_t) proj_in_dim) continue;
                    x[di] = lat[(size_t) (t * P + q) * n_ch + ch];
                }
            }
        }
        spike_set(t_x, x);
        std::vector<float> ev((size_t) enc_H * enc_S);
        memcpy(ev.data(), enc, ev.size() * sizeof(float));
        spike_set(t_enc, ev);
        std::vector<int32_t> ip(S); for (int i = 0; i < S; i++) ip[i] = i;
        ggml_backend_tensor_set(t_pos, ip.data(), 0, ip.size() * 4);
        std::vector<float> z((size_t) 6 * H, 0.0f);
        spike_set(t_tpj, z);
        SpikeRng rng = { 0xd17ULL };
        std::vector<float> tg((size_t) H * S);
        srng_fill_normal(&rng, tg, 1.0f);
        spike_set(t_tgt, tg);
    }
    st_close(&st);

    // ── trainable LoRA on the attention projections ─────────────────────
    struct Site { ggml_tensor * w; const char * name; int64_t in_dim; int64_t out_dim; };
    std::vector<Site> sites;
    auto S_full = [&](ggml_tensor * w, const char * n) { sites.push_back({ w, n, w->ne[0], w->ne[1] }); };
    if (ly->sa_qkv)      S_full(ly->sa_qkv, "sa_qkv(fused)");
    else if (ly->sa_qk) { S_full(ly->sa_qk, "sa_qk(fused)"); S_full(ly->sa_v_proj, "sa_v"); }
    else { S_full(ly->sa_q_proj, "sa_q"); S_full(ly->sa_k_proj, "sa_k"); S_full(ly->sa_v_proj, "sa_v"); }
    S_full(ly->sa_o_proj, "sa_o");
    if (ly->ca_qkv) {
        // Q comes from the hidden state, K/V from the encoder — a single fused
        // LoRA is impossible, so the adapter covers the Q slice only.
        sites.push_back({ ly->ca_qkv, "ca_q(slice of fused qkv)", ly->ca_qkv->ne[0], (int64_t) Nh * D });
    } else if (ly->ca_kv) { S_full(ly->ca_q_proj, "ca_q"); S_full(ly->ca_kv, "ca_kv(fused)"); }
    else { S_full(ly->ca_q_proj, "ca_q"); S_full(ly->ca_k_proj, "ca_k"); S_full(ly->ca_v_proj, "ca_v"); }
    S_full(ly->ca_o_proj, "ca_o");

    S4Lora L;
    { ggml_init_params p = { (sites.size() * 2 + 8) * ggml_tensor_overhead(), nullptr, true }; L.ctx = ggml_init(p); }
    L.pairs.resize(sites.size());
    for (size_t i = 0; i < sites.size(); i++) {
        const int64_t in_dim = sites[i].in_dim, out_dim = sites[i].out_dim;
        L.pairs[i].A = ggml_new_tensor_2d(L.ctx, GGML_TYPE_F32, in_dim, rank);
        L.pairs[i].B = ggml_new_tensor_2d(L.ctx, GGML_TYPE_F32, rank, out_dim);
        L.pairs[i].scale = 2.0f;
        char nm[96];
        snprintf(nm, sizeof(nm), "%s.lora_A", sites[i].name); ggml_set_name(L.pairs[i].A, nm);
        snprintf(nm, sizeof(nm), "%s.lora_B", sites[i].name); ggml_set_name(L.pairs[i].B, nm);
        ggml_set_param(L.pairs[i].A);
        ggml_set_param(L.pairs[i].B);
        L.params.push_back(L.pairs[i].A);
        L.params.push_back(L.pairs[i].B);
    }
    L.buf = ggml_backend_alloc_ctx_tensors(L.ctx, m.backend);
    if (!L.buf) { fprintf(stderr, "[S4] FATAL: LoRA alloc failed\n"); return 1; }
    {
        SpikeRng rng = { 0x5a4eULL };
        size_t np = 0;
        for (auto & pr : L.pairs) {
            std::vector<float> a((size_t) ggml_nelements(pr.A)), b((size_t) ggml_nelements(pr.B));
            srng_fill_normal(&rng, a, 1.0f / sqrtf((float) pr.A->ne[0]));
            srng_fill_normal(&rng, b, 1e-2f);  // nonzero so dL/dA is measurable at step 0
            spike_set(pr.A, a); spike_set(pr.B, b);
            np += a.size() + b.size();
        }
        fprintf(stderr, "[S4] %zu LoRA sites, %zu trainable params (%.1f MB f32)\n", sites.size(), np,
                np * 4.0 / 1048576.0);
    }

    // ── graph: proj_in -> cond_emb -> ONE decoder block ──────────────────
    ggml_context * ctxc;
    { ggml_init_params p = { (size_t) 8192 * ggml_tensor_overhead() +
                                 8 * ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, true) + (4u << 20),
                             nullptr, true };
      ctxc = ggml_init(p); }

    int n_casts = 0;
    ggml_tensor * hidden = ggml_add(ctxc, ggml_mul_mat(ctxc, m.proj_in_w, t_x), m.proj_in_b);   // [H, S]
    ggml_tensor * encp   = ggml_add(ctxc, ggml_mul_mat(ctxc, m.cond_emb_w, t_enc), m.cond_emb_b);  // [H, enc_S]

    // AdaLN modulation vectors (real scale_shift_table + constant tproj)
    ggml_tensor * ss = ly->scale_shift_table;
    if (ss->type != GGML_TYPE_F32) ss = ggml_cast(ctxc, ss, GGML_TYPE_F32);
    ggml_tensor * adaln = ggml_add(ctxc, ggml_reshape_1d(ctxc, ss, 6 * H), t_tpj);
    const size_t Hb = (size_t) H * sizeof(float);
    ggml_tensor * shift_sa = ggml_view_1d(ctxc, adaln, H, 0 * Hb);
    ggml_tensor * scale_sa = ggml_view_1d(ctxc, adaln, H, 1 * Hb);
    ggml_tensor * gate_sa  = ggml_view_1d(ctxc, adaln, H, 2 * Hb);
    ggml_tensor * shift_ff = ggml_view_1d(ctxc, adaln, H, 3 * Hb);
    ggml_tensor * scale_ff = ggml_view_1d(ctxc, adaln, H, 4 * Hb);
    ggml_tensor * gate_ff  = ggml_view_1d(ctxc, adaln, H, 5 * Hb);

    size_t si = 0;
    // ---- self-attention ----
    ggml_tensor * res = hidden;
    ggml_tensor * nsa = dit_ggml_rms_norm_weighted(ctxc, hidden, ly->self_attn_norm, c.rms_norm_eps);
    nsa = dit_ggml_adaln(ctxc, nsa, scale_sa, shift_sa, m.scalar_one);
    ggml_tensor *q, *k, *v;
    const int q_dim = Nh * D, kv_dim = Nkv * D;
    if (ly->sa_qkv) {
        ggml_tensor * qkv = s4_linear(ctxc, ly->sa_qkv, &L.pairs[si++], nsa, &n_casts);
        q = ggml_cont(ctxc, ggml_view_2d(ctxc, qkv, q_dim, S, qkv->nb[1], 0));
        k = ggml_cont(ctxc, ggml_view_2d(ctxc, qkv, kv_dim, S, qkv->nb[1], (size_t) q_dim * qkv->nb[0]));
        v = ggml_cont(ctxc, ggml_view_2d(ctxc, qkv, kv_dim, S, qkv->nb[1], (size_t) (q_dim + kv_dim) * qkv->nb[0]));
    } else if (ly->sa_qk) {
        ggml_tensor * qk = s4_linear(ctxc, ly->sa_qk, &L.pairs[si++], nsa, &n_casts);
        q = ggml_cont(ctxc, ggml_view_2d(ctxc, qk, q_dim, S, qk->nb[1], 0));
        k = ggml_cont(ctxc, ggml_view_2d(ctxc, qk, kv_dim, S, qk->nb[1], (size_t) q_dim * qk->nb[0]));
        v = s4_linear(ctxc, ly->sa_v_proj, &L.pairs[si++], nsa, &n_casts);
    } else {
        q = s4_linear(ctxc, ly->sa_q_proj, &L.pairs[si++], nsa, &n_casts);
        k = s4_linear(ctxc, ly->sa_k_proj, &L.pairs[si++], nsa, &n_casts);
        v = s4_linear(ctxc, ly->sa_v_proj, &L.pairs[si++], nsa, &n_casts);
    }
    q = ggml_reshape_3d(ctxc, q, D, Nh, S);
    k = ggml_reshape_3d(ctxc, k, D, Nkv, S);
    v = ggml_reshape_3d(ctxc, v, D, Nkv, S);
    q = ggml_mul(ctxc, ggml_rms_norm(ctxc, q, c.rms_norm_eps), dit_ggml_f32(ctxc, ly->sa_q_norm));
    k = ggml_mul(ctxc, ggml_rms_norm(ctxc, k, c.rms_norm_eps), dit_ggml_f32(ctxc, ly->sa_k_norm));
    q = ggml_rope_ext(ctxc, q, t_pos, NULL, D, 2, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctxc, k, t_pos, NULL, D, 2, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    q = ggml_permute(ctxc, q, 0, 2, 1, 3);
    k = ggml_permute(ctxc, k, 0, 2, 1, 3);
    v = ggml_permute(ctxc, v, 0, 2, 1, 3);
    ggml_tensor * attn = dit_attn_f32(ctxc, q, k, v, nullptr, 1.0f / sqrtf((float) D));
    attn = ggml_reshape_2d(ctxc, attn, Nh * D, S);
    ggml_tensor * sa_out = s4_linear(ctxc, ly->sa_o_proj, &L.pairs[si++], attn, &n_casts);
    hidden = dit_ggml_gated_add(ctxc, res, sa_out, gate_sa);

    // ---- cross-attention ----
    ggml_tensor * nca = dit_ggml_rms_norm_weighted(ctxc, hidden, ly->cross_attn_norm, c.rms_norm_eps);
    ggml_tensor *cq, *ck, *cv;
    if (ly->ca_qkv) {
        ggml_tensor * cw = ly->ca_qkv->type == GGML_TYPE_F32 ? ly->ca_qkv
                                                             : ggml_cast(ctxc, ly->ca_qkv, GGML_TYPE_F32);
        n_casts++;
        ggml_tensor * w_q  = ggml_cont(ctxc, ggml_view_2d(ctxc, cw, cw->ne[0], q_dim, cw->nb[1], 0));
        ggml_tensor * w_kv = ggml_cont(ctxc, ggml_view_2d(ctxc, cw, cw->ne[0], 2 * kv_dim, cw->nb[1],
                                                          (size_t) q_dim * cw->nb[1]));
        cq = s4_linear(ctxc, w_q, &L.pairs[si], nca, &n_casts);
        ggml_tensor * kv = ggml_mul_mat(ctxc, w_kv, encp);
        si++;
        ck = ggml_cont(ctxc, ggml_view_2d(ctxc, kv, kv_dim, enc_S, kv->nb[1], 0));
        cv = ggml_cont(ctxc, ggml_view_2d(ctxc, kv, kv_dim, enc_S, kv->nb[1], (size_t) kv_dim * kv->nb[0]));
    } else if (ly->ca_kv) {
        cq = s4_linear(ctxc, ly->ca_q_proj, &L.pairs[si++], nca, &n_casts);
        ggml_tensor * kv = s4_linear(ctxc, ly->ca_kv, &L.pairs[si++], encp, &n_casts);
        ck = ggml_cont(ctxc, ggml_view_2d(ctxc, kv, kv_dim, enc_S, kv->nb[1], 0));
        cv = ggml_cont(ctxc, ggml_view_2d(ctxc, kv, kv_dim, enc_S, kv->nb[1], (size_t) kv_dim * kv->nb[0]));
    } else {
        cq = s4_linear(ctxc, ly->ca_q_proj, &L.pairs[si++], nca, &n_casts);
        ck = s4_linear(ctxc, ly->ca_k_proj, &L.pairs[si++], encp, &n_casts);
        cv = s4_linear(ctxc, ly->ca_v_proj, &L.pairs[si++], encp, &n_casts);
    }
    cq = ggml_permute(ctxc, ggml_reshape_3d(ctxc, cq, D, Nh, S), 0, 2, 1, 3);
    ck = ggml_permute(ctxc, ggml_reshape_3d(ctxc, ck, D, Nkv, enc_S), 0, 2, 1, 3);
    cv = ggml_permute(ctxc, ggml_reshape_3d(ctxc, cv, D, Nkv, enc_S), 0, 2, 1, 3);
    if (ly->ca_q_norm) cq = ggml_mul(ctxc, ggml_rms_norm(ctxc, cq, c.rms_norm_eps), dit_ggml_f32(ctxc, ly->ca_q_norm));
    if (ly->ca_k_norm) ck = ggml_mul(ctxc, ggml_rms_norm(ctxc, ck, c.rms_norm_eps), dit_ggml_f32(ctxc, ly->ca_k_norm));
    ggml_tensor * cattn = dit_attn_f32(ctxc, cq, ck, cv, nullptr, 1.0f / sqrtf((float) D));
    cattn = ggml_reshape_2d(ctxc, cattn, Nh * D, S);
    hidden = ggml_add(ctxc, hidden, s4_linear(ctxc, ly->ca_o_proj, &L.pairs[si++], cattn, &n_casts));

    // ---- MLP ----
    res = hidden;
    ggml_tensor * nff = dit_ggml_rms_norm_weighted(ctxc, hidden, ly->mlp_norm, c.rms_norm_eps);
    nff = dit_ggml_adaln(ctxc, nff, scale_ff, shift_ff, m.scalar_one);
    ggml_tensor * ff;
    if (ly->gate_up) {
        ggml_tensor * gu = s4_linear(ctxc, ly->gate_up, nullptr, nff, &n_casts);
        // SUBSTITUTION 3: fused ggml_swiglu has NO backward — split the fused
        // projection output and use ggml_swiglu_split instead.
        const int64_t F = gu->ne[0] / 2;
        ggml_tensor * g = ggml_cont(ctxc, ggml_view_2d(ctxc, gu, F, S, gu->nb[1], 0));
        ggml_tensor * u = ggml_cont(ctxc, ggml_view_2d(ctxc, gu, F, S, gu->nb[1], (size_t) F * gu->nb[0]));
        ff = ggml_swiglu_split(ctxc, g, u);
    } else {
        ggml_tensor * g = s4_linear(ctxc, ly->gate_proj, nullptr, nff, &n_casts);
        ggml_tensor * u = s4_linear(ctxc, ly->up_proj, nullptr, nff, &n_casts);
        ff = ggml_swiglu_split(ctxc, g, u);
    }
    ggml_tensor * ffo = s4_linear(ctxc, ly->down_proj, nullptr, ff, &n_casts);
    hidden = dit_ggml_gated_add(ctxc, res, ffo, gate_ff);

    ggml_tensor * outputs = ggml_sqr(ctxc, ggml_sub(ctxc, hidden, t_tgt));
    ggml_set_name(outputs, "sq_err");
    fprintf(stderr, "[S4] graph built (%d base-weight casts, fused sa=%d ca=%d mlp=%d)\n", n_casts, (int) fused_sa,
            (int) fused_ca, (int) fused_mlp);

    // ── gradients ───────────────────────────────────────────────────────
    bool ok = true;
    {
        ggml_opt_params op = ggml_opt_default_params(m.sched, GGML_OPT_LOSS_TYPE_SUM);
        op.ctx_compute = ctxc; op.inputs = t_x; op.outputs = outputs;
        op.build_type = GGML_OPT_BUILD_TYPE_OPT; op.opt_period = 1 << 20;
        ggml_opt_optimizer_params pars = ggml_opt_get_default_optimizer_params(nullptr);
        op.get_opt_pars = s4_opt_pars; op.get_opt_pars_ud = &pars;
        ggml_opt_context_t oc = ggml_opt_init(op);
        fprintf(stderr, "[S4] backward graph built OK (no missing-backward abort)\n");

        const int64_t t0 = ggml_time_ms();
        ggml_opt_alloc(oc, true);
        ggml_opt_eval(oc, nullptr);
        ggml_backend_sched_synchronize(m.sched);
        const long long ms = (long long) (ggml_time_ms() - t0);
        float loss = 0;
        ggml_backend_tensor_get(ggml_opt_loss(oc), &loss, 0, sizeof(float));
        fprintf(stderr, "[S4] loss(SSE)=%.6e  fwd+bwd=%lld ms  VRAM=%zu MB\n", (double) loss, ms,
                spike_vram_used_mb(m.backend));

        int nz = 0;
        for (auto * p : L.params) {
            ggml_tensor * g = ggml_opt_grad_acc(oc, p);
            if (!g) { fprintf(stderr, "[S4] FATAL: no grad for %s\n", p->name); return 1; }
            std::vector<float> gv = spike_get(g);
            double n = spike_l2(gv);
            if (!spike_all_finite(gv)) { ok = false; fprintf(stderr, "[S4] NON-FINITE grad %s\n", p->name); }
            if (n == 0.0) { nz++; fprintf(stderr, "[S4] ZERO grad %s\n", p->name); }
            fprintf(stderr, "[S4]   |g| %-24s = %.6e\n", p->name, n);
        }
        ok = ok && nz == 0;
        ggml_opt_free(oc);
    }

    // ── one AdamW step ──────────────────────────────────────────────────
    {
        std::vector<std::vector<float>> before;
        for (auto * p : L.params) before.push_back(spike_get(p));
        ggml_opt_params op = ggml_opt_default_params(m.sched, GGML_OPT_LOSS_TYPE_SUM);
        op.ctx_compute = ctxc; op.inputs = t_x; op.outputs = outputs;
        op.build_type = GGML_OPT_BUILD_TYPE_OPT; op.opt_period = 1;
        ggml_opt_optimizer_params pars = ggml_opt_get_default_optimizer_params(nullptr);
        pars.adamw.alpha = 1e-3f;
        op.get_opt_pars = s4_opt_pars; op.get_opt_pars_ud = &pars;
        ggml_opt_context_t oc = ggml_opt_init(op);
        float l0 = 0, l1 = 0;
        ggml_opt_alloc(oc, true); ggml_opt_eval(oc, nullptr);
        ggml_backend_tensor_get(ggml_opt_loss(oc), &l0, 0, sizeof(float));
        ggml_opt_alloc(oc, true); ggml_opt_eval(oc, nullptr);
        ggml_backend_tensor_get(ggml_opt_loss(oc), &l1, 0, sizeof(float));
        double d = 0; bool fin = true;
        for (size_t i = 0; i < L.params.size(); i++) {
            std::vector<float> a = spike_get(L.params[i]);
            if (!spike_all_finite(a)) fin = false;
            for (size_t j = 0; j < a.size(); j++) d += fabs((double) a[j] - before[i][j]);
        }
        fprintf(stderr, "[S4] AdamW: loss %.6e -> %.6e, sum|dparam|=%.6e, finite=%d\n", (double) l0, (double) l1, d,
                (int) fin);
        ok = ok && fin && d > 0 && l1 < l0;
        ggml_opt_free(oc);
    }

    fprintf(stderr, "\n[S4] SUMMARY S=%d enc_S=%d sites=%zu params=%zu -> %s\n", S, enc_S, sites.size(),
            L.params.size(), ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
