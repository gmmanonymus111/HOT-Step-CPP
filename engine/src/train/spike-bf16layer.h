#pragma once
// spike-bf16layer.h — Rung `bf16layer`: P2 prototype on ONE REAL 4B layer.
//
// EVIDENCE CODE, NOT PRODUCT CODE.
//
// Proves (or disproves) the engine-side backward-surgery path with NO ggml
// changes:
//
//   REF  arm (what ships today, low-vram mode):
//        forward  mul_mat(cast(W_bf16 -> F32), x)      <- per-segment F32 window
//        backward out_prod(W_f32, transpose(grad))     <- F32-only, cublasSgemm/TF32
//
//   BF16 arm (proposed):
//        forward  mul_mat(W_bf16, x)                   <- cublasGemmEx CUDA_R_16BF
//        backward mul_mat(Wt_bf16, grad)               <- same kernel, BF16 too
//
// The BF16 arm is produced by running ggml_build_backward_expand() UNCHANGED
// and then rewriting, in place, every GGML_OP_OUT_PROD node whose src[0] is one
// of the layer's BF16 base weights:
//
//        node->op     = GGML_OP_MUL_MAT
//        node->src[0] = Wt_bf16   (physical transpose, built ONCE per weight)
//        node->src[1] = grad      (recovered from the TRANSPOSE view's src[0])
//
// Shapes line up exactly, which is why the rewrite is legal:
//        out_prod(W[K,N], gT[S,N]) -> [K,S]
//        mul_mat (Wt[N,K], g[N,S]) -> [K,S]
//
// This is the same trick lm-ckpt.h already ships for the output head
// (`t_embT`, kept in the base dtype, `ggml_mul_mat(t_embT, dl)` instead of
// out_prod) — here generalised to the 7 per-layer projections.

#include "spike-lm.h"

#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// ─── physical BF16 transpose, built once per weight ─────────────────────────
//
// Byte-wise and dtype-agnostic, exactly like lm_ckpt_build_embed_t()
// (lm-ckpt.h:272). ggml's mul_mat reduces over ne0 only, so a physical
// transpose is unavoidable — cuBLAS's transa flag is not reachable from here.
struct GbWtBank {
    ggml_context *                          ctx = nullptr;
    ggml_backend_buffer_t                   buf = nullptr;
    std::map<ggml_tensor *, ggml_tensor *>  map;  // W -> Wt
    size_t                                  bytes = 0;
};

static void gb_wt_free(GbWtBank * b) {
    if (b->buf) ggml_backend_buffer_free(b->buf);
    if (b->ctx) ggml_free(b->ctx);
    b->map.clear();
    *b = GbWtBank{};
}

static bool gb_wt_build(GbWtBank * bank, ggml_backend_t backend, const std::vector<ggml_tensor *> & ws) {
    ggml_init_params p = { (ws.size() + 8) * ggml_tensor_overhead(), nullptr, true };
    bank->ctx          = ggml_init(p);
    if (!bank->ctx) return false;

    std::vector<ggml_tensor *> wts;
    for (ggml_tensor * w : ws) {
        if (ggml_blck_size(w->type) != 1) {
            fprintf(stderr, "[bf16layer] FATAL: %s is block-quantized, cannot transpose\n", w->name);
            return false;
        }
        ggml_tensor * wt = ggml_new_tensor_2d(bank->ctx, w->type, w->ne[1], w->ne[0]);
        ggml_format_name(wt, "%s.T", w->name);
        ggml_set_input(wt);
        wts.push_back(wt);
        bank->bytes += ggml_nbytes(wt);
    }
    bank->buf = ggml_backend_alloc_ctx_tensors(bank->ctx, backend);
    if (!bank->buf) {
        fprintf(stderr, "[bf16layer] FATAL: Wt bank alloc failed\n");
        return false;
    }

    for (size_t i = 0; i < ws.size(); i++) {
        ggml_tensor * w  = ws[i];
        ggml_tensor * wt = wts[i];
        const size_t  ts = ggml_type_size(w->type);
        const int64_t K = w->ne[0], N = w->ne[1];
        std::vector<uint8_t> src(ggml_nbytes(w)), dst(ggml_nbytes(wt));
        ggml_backend_tensor_get(w, src.data(), 0, src.size());
        for (int64_t n = 0; n < N; n++) {
            for (int64_t k = 0; k < K; k++) {
                memcpy(&dst[(size_t) (k * N + n) * ts], &src[(size_t) (n * K + k) * ts], ts);
            }
        }
        ggml_backend_tensor_set(wt, dst.data(), 0, dst.size());
        bank->map[w] = wt;
    }
    return true;
}

// ─── the surgery ────────────────────────────────────────────────────────────

static int gb_rewrite_outprod(ggml_cgraph * gf, const std::map<ggml_tensor *, ggml_tensor *> & wt, int * n_skipped) {
    int            n     = 0;
    ggml_tensor ** nodes = ggml_graph_nodes(gf);
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        ggml_tensor * nd = nodes[i];
        if (nd->op != GGML_OP_OUT_PROD) continue;

        auto it = wt.find(nd->src[0]);
        if (it == wt.end()) continue;  // LoRA A/B out_prods: leave alone (F32, tiny)

        ggml_tensor * Wt = it->second;
        ggml_tensor * tT = nd->src[1];
        if (tT->op != GGML_OP_TRANSPOSE || !tT->src[0]) { (*n_skipped)++; continue; }
        ggml_tensor * g = tT->src[0];

        // out_prod(W[K,N], gT[S,N]) -> [K,S]   ==   mul_mat(Wt[N,K], g[N,S]) -> [K,S]
        if (Wt->ne[0] != g->ne[0] || nd->ne[0] != Wt->ne[1] || nd->ne[1] != g->ne[1]) { (*n_skipped)++; continue; }
        if (!ggml_is_contiguous(g)) { (*n_skipped)++; continue; }

        nd->op     = GGML_OP_MUL_MAT;
        nd->src[0] = Wt;
        nd->src[1] = g;
        memset(nd->op_params, 0, sizeof(nd->op_params));
        n++;
    }
    return n;
}

// ─── one arm ────────────────────────────────────────────────────────────────

struct GbArm {
    bool                            ok = false;
    double                          ms = 0;
    double                          loss = 0;
    size_t                          vram_mb = 0;
    int                             n_casts = 0, n_rewritten = 0, n_outprod_left = 0;
    std::vector<std::vector<float>> grads;  // per LoRA param
};

static bool gb_run_arm(Qwen3LM * lm, SpikeLora * L, int layer, int S, bool bf16_mode, const GbWtBank * bank,
                       ggml_tensor * hin, ggml_tensor * pos, ggml_tensor * msk, ggml_tensor * tgt, int reps,
                       GbArm * out) {
    ggml_context * ctx = nullptr;
    {
        ggml_init_params p = { (size_t) 8192 * ggml_tensor_overhead() +
                                   4 * ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE * 4, true) + (4u << 20),
                               nullptr, true };
        ctx = ggml_init(p);
    }
    if (!ctx) return false;

    SpikeFwdOpts fo;
    fo.cast_base_f32 = !bf16_mode;

    ggml_tensor * y    = spike_lm_layer(ctx, lm->cfg, &lm->layers[layer], hin, pos, msk, S, &fo);
    ggml_tensor * diff = ggml_sub(ctx, y, tgt);
    ggml_tensor * loss = ggml_sum(ctx, ggml_sqr(ctx, diff));
    ggml_set_name(loss, "sse");
    ggml_set_loss(loss);
    ggml_set_output(loss);
    out->n_casts = fo.n_casts;

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE * 4, /*grads=*/true);
    ggml_build_forward_expand(gf, loss);
    ggml_build_backward_expand(ctx, gf, /*grad_accs=*/nullptr);

    if (bf16_mode) {
        int skipped     = 0;
        out->n_rewritten = gb_rewrite_outprod(gf, bank->map, &skipped);
        if (skipped) fprintf(stderr, "[bf16layer]   WARNING: %d out_prod nodes matched a weight but were SKIPPED\n", skipped);
    }
    // Count any base-weight out_prod still present (must be 0 in the BF16 arm).
    {
        ggml_tensor ** nodes = ggml_graph_nodes(gf);
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            if (nodes[i]->op == GGML_OP_OUT_PROD && bank->map.count(nodes[i]->src[0])) out->n_outprod_left++;
        }
    }

    // The LoRA grads are ordinary graph nodes (grad_accs was nullptr, so only
    // the loss gets an accumulator — ggml.c:7083). gallocr WILL recycle their
    // buffers into later nodes unless they are flagged OUTPUT
    // (ggml-alloc.c:645,692), which would silently corrupt the parity read-back.
    for (ggml_tensor * p : L->params) {
        ggml_tensor * g = ggml_graph_get_grad_acc(gf, p);
        if (!g) g = ggml_graph_get_grad(gf, p);
        if (g) ggml_set_output(g);
    }

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(lm->backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "[bf16layer] FATAL: graph alloc failed (%s arm)\n", bf16_mode ? "BF16" : "REF");
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return false;
    }

    // correctness pass
    ggml_graph_reset(gf);
    if (ggml_backend_graph_compute(lm->backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[bf16layer] FATAL: compute failed (%s arm)\n", bf16_mode ? "BF16" : "REF");
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_synchronize(lm->backend);

    float lv = 0;
    ggml_backend_tensor_get(loss, &lv, 0, sizeof(float));
    out->loss = (double) lv;

    out->grads.clear();
    for (ggml_tensor * p : L->params) {
        ggml_tensor * g = ggml_graph_get_grad_acc(gf, p);
        if (!g) g = ggml_graph_get_grad(gf, p);
        if (!g) {
            fprintf(stderr, "[bf16layer] FATAL: no grad for %s\n", p->name);
            ggml_gallocr_free(galloc);
            ggml_free(ctx);
            return false;
        }
        std::vector<float> v((size_t) ggml_nelements(g));
        ggml_backend_tensor_get(g, v.data(), 0, v.size() * sizeof(float));
        out->grads.push_back(std::move(v));
    }
    out->vram_mb = spike_vram_used_mb(lm->backend);

    // timing pass
    for (int i = 0; i < 3; i++) ggml_backend_graph_compute(lm->backend, gf);
    ggml_backend_synchronize(lm->backend);
    const int64_t t0 = ggml_time_us();
    for (int i = 0; i < reps; i++) ggml_backend_graph_compute(lm->backend, gf);
    ggml_backend_synchronize(lm->backend);
    out->ms = (double) (ggml_time_us() - t0) / 1000.0 / (double) reps;

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    out->ok = true;
    return true;
}

// ─── the rung ───────────────────────────────────────────────────────────────

static int spike_bf16layer(const char * lm_path, int S, int rank, int layer, int reps) {
    fprintf(stderr, "\n=== bf16layer: ONE REAL layer %d, S=%d rank=%d ===\n", layer, S, rank);
    g_qwen3_load_no_fuse = true;

    Qwen3LM lm;
    if (!qw3lm_load(&lm, lm_path, /*max_seq_len=*/S + 8, /*n_kv_sets=*/1)) {
        fprintf(stderr, "[bf16layer] FATAL: LM load failed\n");
        return 1;
    }
    const Qwen3LMConfig & c = lm.cfg;
    fprintf(stderr, "[bf16layer] H=%d F=%d Nh=%d Nkv=%d D=%d L=%d  q_proj dtype=%s\n", c.hidden_size,
            c.intermediate_size, c.n_heads, c.n_kv_heads, c.head_dim, c.n_layers,
            ggml_type_name(lm.layers[layer].q_proj->type));
    const size_t vram_load = spike_vram_used_mb(lm.backend);
    fprintf(stderr, "[bf16layer] VRAM after load: %zu MB\n", vram_load);

    if (lm.layers[layer].q_proj->type != GGML_TYPE_BF16) {
        fprintf(stderr, "[bf16layer] FATAL: base is %s, this rung needs a BF16 base\n",
                ggml_type_name(lm.layers[layer].q_proj->type));
        return 1;
    }

    SpikeLora L;
    if (!spike_lora_init(&L, &lm, layer, layer + 1, rank, (float) (2 * rank), 0xbeefULL, 1e-2f)) return 1;

    // ── static inputs ────────────────────────────────────────────────────
    const int      H = c.hidden_size;
    ggml_context * ctx_static = nullptr;
    {
        ggml_init_params p = { 8 * ggml_tensor_overhead(), nullptr, true };
        ctx_static = ggml_init(p);
    }
    ggml_tensor * hin = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, H, S);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx_static, GGML_TYPE_I32, S);
    ggml_tensor * msk = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, S, S);
    ggml_tensor * tgt = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, H, S);
    ggml_set_input(hin); ggml_set_input(pos); ggml_set_input(msk); ggml_set_input(tgt);
    ggml_backend_buffer_t buf_static = ggml_backend_alloc_ctx_tensors(ctx_static, lm.backend);
    if (!buf_static) { fprintf(stderr, "[bf16layer] FATAL: static alloc failed\n"); return 1; }
    {
        SpikeRng           rng = { 0xc0ffeeULL };
        std::vector<float> h((size_t) H * S), t((size_t) H * S);
        srng_fill_normal(&rng, h, 1.0f);
        srng_fill_normal(&rng, t, 1.0f);
        spike_set(hin, h);
        spike_set(tgt, t);
        std::vector<int32_t> ip(S);
        for (int i = 0; i < S; i++) ip[i] = i;
        ggml_backend_tensor_set(pos, ip.data(), 0, ip.size() * sizeof(int32_t));
        std::vector<float> m = spike_causal_mask(S);
        spike_set(msk, m);
    }

    // ── Wt bank for the 7 projections ────────────────────────────────────
    Qwen3Layer *               ly = &lm.layers[layer];
    std::vector<ggml_tensor *> ws = { ly->q_proj, ly->k_proj, ly->v_proj, ly->o_proj,
                                      ly->gate_proj, ly->up_proj, ly->down_proj };
    GbWtBank bank;
    const size_t vram_before_bank = spike_vram_used_mb(lm.backend);
    if (!gb_wt_build(&bank, lm.backend, ws)) return 1;
    fprintf(stderr, "[bf16layer] Wt bank: %d tensors, %.1f MB (BF16 transposed window)\n", (int) bank.map.size(),
            (double) bank.bytes / 1048576.0);
    (void) vram_before_bank;

    // ── run both arms ────────────────────────────────────────────────────
    GbArm ref, bf;
    fprintf(stderr, "\n[bf16layer] --- REF arm (F32 window: cast + out_prod) ---\n");
    if (!gb_run_arm(&lm, &L, layer, S, /*bf16_mode=*/false, &bank, hin, pos, msk, tgt, reps, &ref)) return 1;
    fprintf(stderr, "[bf16layer]   casts=%d  loss=%.6e  %.3f ms/step  vram=%zu MB\n", ref.n_casts, ref.loss, ref.ms,
            ref.vram_mb);

    fprintf(stderr, "\n[bf16layer] --- BF16 arm (no cast; out_prod -> mul_mat(Wt_bf16)) ---\n");
    if (!gb_run_arm(&lm, &L, layer, S, /*bf16_mode=*/true, &bank, hin, pos, msk, tgt, reps, &bf)) return 1;
    fprintf(stderr, "[bf16layer]   casts=%d  rewritten=%d  base-out_prod-left=%d  loss=%.6e  %.3f ms/step  vram=%zu MB\n",
            bf.n_casts, bf.n_rewritten, bf.n_outprod_left, bf.loss, bf.ms, bf.vram_mb);

    // ── gradient parity over ALL LoRA params ─────────────────────────────
    double gmax_rel = 0, gse = 0, gsa = 0, dot = 0, na = 0, nb = 0;
    bool   finite = true;
    size_t nel = 0;
    for (size_t i = 0; i < ref.grads.size(); i++) {
        const std::vector<float> & a = ref.grads[i];
        const std::vector<float> & b = bf.grads[i];
        if (a.size() != b.size()) { fprintf(stderr, "[bf16layer] FATAL: grad size mismatch\n"); return 1; }
        double amax = 0, dmax = 0;
        for (size_t j = 0; j < a.size(); j++) {
            if (!std::isfinite(a[j]) || !std::isfinite(b[j])) finite = false;
            const double x = a[j], y = b[j], d = x - y;
            amax = std::max(amax, fabs(x));
            dmax = std::max(dmax, fabs(d));
            gse += d * d; gsa += x * x;
            dot += x * y; na += x * x; nb += y * y;
        }
        nel += a.size();
        const double rel = dmax / std::max(amax, 1e-30);
        gmax_rel         = std::max(gmax_rel, rel);
        if (i < 14) {
            fprintf(stderr, "[bf16layer]   %-22s max_rel=%.4e  |g|max=%.4e\n", L.params[i]->name, rel, amax);
        }
    }
    const double rms_rel = sqrt(gse) / std::max(sqrt(gsa), 1e-30);
    const double cosine  = dot / std::max(sqrt(na) * sqrt(nb), 1e-30);

    printf("\n=== bf16layer RESULT  (layer %d, S=%d, rank=%d, %zu LoRA tensors, %zu grad elems) ===\n", layer, S, rank,
           L.params.size(), nel);
    printf("  REF  (F32 window) : %8.3f ms/step   casts=%2d   loss=%.6e   vram=%zu MB\n", ref.ms, ref.n_casts, ref.loss,
           ref.vram_mb);
    printf("  BF16 (surgery)    : %8.3f ms/step   rewritten=%2d  loss=%.6e   vram=%zu MB\n", bf.ms, bf.n_rewritten,
           bf.loss, bf.vram_mb);
    printf("  SPEEDUP           : %8.2fx        VRAM delta = %+d MB (Wt bank = %.1f MB)\n",
           bf.ms > 0 ? ref.ms / bf.ms : 0.0, (int) bf.vram_mb - (int) ref.vram_mb, (double) bank.bytes / 1048576.0);
    printf("  GRAD PARITY       : max_rel=%.4e  rms_rel=%.4e  cosine=%.9f  finite=%s\n", gmax_rel, rms_rel, cosine,
           finite ? "yes" : "NO");
    printf("  loss rel diff     : %.4e\n", fabs(ref.loss - bf.loss) / std::max(fabs(ref.loss), 1e-30));

    const bool pass = finite && bf.n_outprod_left == 0 && bf.n_rewritten == 7 && cosine > 1.0 - 1e-3;
    printf("  -> %s\n", pass ? "PASS" : "FAIL");

    gb_wt_free(&bank);
    ggml_backend_buffer_free(buf_static);
    ggml_free(ctx_static);
    spike_lora_free(&L);
    return pass ? 0 : 1;
}
