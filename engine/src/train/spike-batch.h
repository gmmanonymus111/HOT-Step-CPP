#pragma once
// spike-batch.h — S-B1 / S-B2 / S-C1: the §4 spike gates for the DiT
// micro-batching + gradient-checkpointing design.
//
//   docs/plans/2026-07-29-dit-batching-checkpointing.md §4
//
// THROWAWAY EVIDENCE CODE. Nothing here is on the training path — the production
// batch axis lands in dit-train-graph.h (ENGINE-A) and the segment driver in
// dit-train-run.h (ENGINE-B). This header exists so the mechanisms are PROVEN on
// a real DiT layer before either is written.
//
//   S-B1  batched forward parity   B=3 batched forward (per-element t / enc /
//                                  ca-mask / CFG state, design B1+B3) vs three
//                                  single-sample dit_train_forward() calls.
//   S-B2  batched backward parity  adapter grads from ONE batched backward vs
//                                  the sum of three single-sample backwards,
//                                  accumulated into the SAME persistent
//                                  LmOptim::acc[] (design B4 weights, C1
//                                  accumulation rule).
//   S-C1  segmented recompute      2 layers split into 2 segments, driven by the
//                                  spike-s3 surrogate loss through
//                                  lm_optim_fill_gacc + the persistent accs, vs
//                                  the monolithic 2-layer backward.
//
// FINDING (S-B2, load-bearing for ENGINE-A — see sb_expand_heads): design B1's
// 4-D attention layout puts the batch on ne3, which the GQA head broadcast in
// ggml_compute_backward cannot survive: for a mul_mat whose src0 is broadcast
// over ne2 (K/V with n_kv_heads < n_heads), the src0 gradient path builds
// `tmp = out_prod(src1, grad)` and then asserts `tmp->ne[3] == 1`
// (ggml.c:6585-6588) before folding the head repetition with ggml_repeat_back.
// With a batch axis on ne3 that assert is `B == 1` and it aborts. The forward is
// unaffected. The fix used here (and the one ENGINE-A must adopt) is to expand
// K/V to n_heads BEFORE the attention mul_mats so no broadcast remains; run the
// spike with --naive-bwd to see the abort verbatim.
//
// MEASURED 2026-07-29, acestep-v15-base-BF16 (24 layers, H=2048, Nh=16/8),
// B=3 over three different songs, T=64, enc_S=48, rank 8, layers 22-23:
//
//   --cpu  (one ggml kernel for both graph shapes — the authoritative run)
//     S-B1 0.000e+00   S-B2 0.000e+00   S-C1 0.000e+00      3/3 PASS, BIT-EXACT
//   CUDA, TF32 off
//     S-B1 9.4e-5 max-abs on |v|max 3.71   S-B2 2.7e-5 rel   S-C1 0.000e+00
//   CUDA, TF32 on (shipping numerics)
//     S-B1 3.2e-3 max-abs                  S-B2 1.6e-3 rel   S-C1 0.000e+00
//   CUDA, batch order reversed (route-identical control): 0.000e+00 on every run
//
// So the graph is exact and the CUDA residual is entirely GEMM ROUTE, not the
// batch axis: adding ne2 > 1 to a mul_mat's src1 makes ggml pick a different
// cuBLAS routine than the single-sample path uses (see sb_promote_f32), and the
// two disagree at F32-accumulation level (TF32 off) or TF32 level (TF32 on).
// The design's §2.3.2/§2.3.3 bars (1e-6 max-abs / 1e-5 rel) are therefore NOT
// reachable on CUDA against a single-sample reference, by construction. What IS
// reachable, and what ENGINE-A's rungs should assert, is (a) the CPU-backend
// identity and (b) the batch-permutation control, which is bit-exact on CUDA and
// is the measurement that actually catches cross-element contamination.
//
// S-C1 is bit-exact on BOTH backends: segmented recompute uses the same graph
// shapes as the monolithic build, so no route change is involved.

#include "backend.h"
#include "train/dit-adapter.h"
#include "train/dit-data.h"
#include "train/dit-train-graph.h"
#include "train/dit-vram.h"
#include "train/lm-optim.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// ─── reporting ──────────────────────────────────────────────────────────────

struct SbResult {
    std::string name;
    bool        pass = false;
    std::string detail;
};

static void sb_report(std::vector<SbResult> & rs, const char * name, bool pass, const std::string & detail) {
    SbResult r;
    r.name   = name;
    r.pass   = pass;
    r.detail = detail;
    rs.push_back(r);
    fprintf(stderr, "[spike-batch] %-5s %-4s %s\n", name, pass ? "PASS" : "FAIL", detail.c_str());
}

// ─── batched graph inputs (batch is ALWAYS the last axis) ───────────────────

struct SbInputs {
    ggml_tensor * t_input = nullptr;  // [in_ch, T,     B]
    ggml_tensor * t_enc   = nullptr;  // [enc_H, enc_S, B]
    ggml_tensor * t_pos   = nullptr;  // [S*B] i32, 0..S-1 repeated per element
    ggml_tensor * t_temb  = nullptr;  // [H,  1, B]
    ggml_tensor * t_tproj = nullptr;  // [6H, 1, B]
    ggml_tensor * t_sa    = nullptr;  // [S, S] f16          — shared (ne2=ne3=1)
    ggml_tensor * t_ca    = nullptr;  // [enc_S, S, 1, B] f16 — per element
    ggml_tensor * t_vtgt  = nullptr;  // [Oc, T, B]
    ggml_tensor * t_lw    = nullptr;  // [1,  T, B]  design B4 loss weights
};

// One [H,1,B] adaLN slot out of the [6H,1,B] adaln tensor. ggml_cont because the
// stride-6H view is not contiguous and every consumer broadcasts it against
// [H,S,B]; the copy is H*B floats and carries no gradient (neither
// scale_shift_table nor t_tproj is a parameter).
static ggml_tensor * sb_adaln_slot(ggml_context * ctx, ggml_tensor * adaln, int k, int H, int B) {
    return ggml_cont(ctx, ggml_view_3d(ctx, adaln, H, 1, B, adaln->nb[1], adaln->nb[2],
                                       (size_t) k * (size_t) H * sizeof(float)));
}

// Grouped head expansion Nkv -> Nh, byte-identical to the mapping ggml's mul_mat
// broadcast performs (query head h reads kv head h / (Nh/Nkv)): a [D,Nkv,S,B]
// tensor is viewed as [D,1,Nkv,S*B], tiled to [D,G,Nkv,S*B] and re-viewed as
// [D,Nh,S,B], so the head index is g + G*kv and h/G == kv exactly.
//
// Cost: K and V activations grow x(Nh/Nkv). The backward is ggml_repeat_back,
// which the CUDA backend supports for F32 while grad->ne[2]*grad->ne[3] (here
// Nkv * S * B) stays <= 32768 (ggml-cuda.cu:5305) — at Nkv=8 that caps S*B at
// 4096, i.e. exactly the design's B5 x S750 point. ENGINE-A must check it.
static ggml_tensor * sb_expand_heads(ggml_context * ctx, ggml_tensor * x, int D, int Nkv, int Nh, int S, int B) {
    if (Nh == Nkv) {
        return x;
    }
    const int64_t G  = (int64_t) Nh / (int64_t) Nkv;
    const int64_t SB = (int64_t) S * (int64_t) B;
    ggml_tensor * a  = ggml_reshape_4d(ctx, x, D, 1, Nkv, SB);
    ggml_tensor * r  = ggml_repeat_4d(ctx, a, D, G, Nkv, SB);
    return ggml_reshape_4d(ctx, r, D, Nh, S, B);
}

// ─── batched decoder layer (design B1) ──────────────────────────────────────
//
// Op-for-op dit_train_layer(), with:
//   hidden  [H,S]        -> [H,S,B]
//   adaLN   [H]          -> [H,1,B]  (broadcasts over S)
//   q/k/v   [D,Nh,S]     -> [D,Nh,S,B], merged to [D,Nh,S*B] around ggml_rope_ext
//   ca_mask [enc_S,S]    -> [enc_S,S,1,B]
//   sa_mask [S,S]        unchanged — ne2 == ne3 == 1 broadcasts over heads AND
//                        batch under the soft_max_ext divisibility rule.
static ggml_tensor * sb_layer(ggml_context * ctx, DitTrainModel * M, int li, ggml_tensor * hidden, ggml_tensor * enc,
                              const SbInputs & in, const DitAdapter * ad, int S, int enc_S, int B, bool expand_kv) {
    const DiTGGMLConfig & c  = M->m.cfg;
    DiTGGMLLayer *        ly = &M->m.layers[li];
    const int             H = c.hidden_size, D = c.head_dim, Nh = c.n_heads, Nkv = c.n_kv_heads;
    const float           scale = 1.0f / sqrtf((float) D);

    ggml_tensor * ss_flat = ggml_reshape_1d(ctx, ly->scale_shift_table, 6 * H);
    ggml_tensor * adaln   = ggml_add(ctx, in.t_tproj, ss_flat);  // [6H,1,B] + [6H]
    ggml_tensor * shift_sa = sb_adaln_slot(ctx, adaln, 0, H, B);
    ggml_tensor * scale_sa = sb_adaln_slot(ctx, adaln, 1, H, B);
    ggml_tensor * gate_sa  = sb_adaln_slot(ctx, adaln, 2, H, B);
    ggml_tensor * shift_ff = sb_adaln_slot(ctx, adaln, 3, H, B);
    ggml_tensor * scale_ff = sb_adaln_slot(ctx, adaln, 4, H, B);
    ggml_tensor * gate_ff  = sb_adaln_slot(ctx, adaln, 5, H, B);

    // ── self-attention ──
    ggml_tensor * res = hidden;
    ggml_tensor * nsa = dit_ggml_rms_norm_weighted(ctx, hidden, ly->self_attn_norm, c.rms_norm_eps);
    nsa               = dit_ggml_adaln(ctx, nsa, scale_sa, shift_sa, M->m.scalar_one);

    ggml_tensor * q = ad->apply(ctx, ly->sa_q_proj, li, DIT_SA_Q, nsa);
    ggml_tensor * k = ad->apply(ctx, ly->sa_k_proj, li, DIT_SA_K, nsa);
    ggml_tensor * v = ad->apply(ctx, ly->sa_v_proj, li, DIT_SA_V, nsa);

    q = ggml_reshape_4d(ctx, q, D, Nh, S, B);
    k = ggml_reshape_4d(ctx, k, D, Nkv, S, B);
    v = ggml_reshape_4d(ctx, v, D, Nkv, S, B);
    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, c.rms_norm_eps), dit_ggml_f32(ctx, ly->sa_q_norm));
    k = ggml_mul(ctx, ggml_rms_norm(ctx, k, c.rms_norm_eps), dit_ggml_f32(ctx, ly->sa_k_norm));
    // positions are [S*B]; merge S and B for rope, restore 4-D after
    q = ggml_reshape_3d(ctx, q, D, Nh, (int64_t) S * B);
    k = ggml_reshape_3d(ctx, k, D, Nkv, (int64_t) S * B);
    q = ggml_rope_ext(ctx, q, in.t_pos, NULL, D, 2 /*NEOX*/, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, in.t_pos, NULL, D, 2, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    q = ggml_reshape_4d(ctx, q, D, Nh, S, B);
    k = ggml_reshape_4d(ctx, k, D, Nkv, S, B);
    if (expand_kv) {
        k = sb_expand_heads(ctx, k, D, Nkv, Nh, S, B);
        v = sb_expand_heads(ctx, v, D, Nkv, Nh, S, B);
    }
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    ggml_tensor * sa_mask = (ly->layer_type == 0) ? in.t_sa : nullptr;
    ggml_tensor * attn    = dit_attn_f32(ctx, q, k, v, sa_mask, scale);  // [D,Nh,S,B]
    attn                  = ggml_reshape_3d(ctx, attn, (int64_t) Nh * D, S, B);
    ggml_tensor * sa_out  = ad->apply(ctx, ly->sa_o_proj, li, DIT_SA_O, attn);
    hidden                = dit_ggml_gated_add(ctx, res, sa_out, gate_sa);

    // ── cross-attention (no gate, plain residual) ──
    ggml_tensor * nca = dit_ggml_rms_norm_weighted(ctx, hidden, ly->cross_attn_norm, c.rms_norm_eps);
    ggml_tensor * cq  = ad->apply(ctx, ly->ca_q_proj, li, DIT_CA_Q, nca);
    ggml_tensor * ck  = ad->apply(ctx, ly->ca_k_proj, li, DIT_CA_K, enc);
    ggml_tensor * cv  = ad->apply(ctx, ly->ca_v_proj, li, DIT_CA_V, enc);
    cq = ggml_reshape_4d(ctx, cq, D, Nh, S, B);
    ck = ggml_reshape_4d(ctx, ck, D, Nkv, enc_S, B);
    cv = ggml_reshape_4d(ctx, cv, D, Nkv, enc_S, B);
    // QK-norm BEFORE the permute (dit-train-graph.h:156-164 — rms_norm_back on a
    // non-contiguous src is wrong; the batched layout does not change that).
    cq = ggml_mul(ctx, ggml_rms_norm(ctx, cq, c.rms_norm_eps), dit_ggml_f32(ctx, ly->ca_q_norm));
    ck = ggml_mul(ctx, ggml_rms_norm(ctx, ck, c.rms_norm_eps), dit_ggml_f32(ctx, ly->ca_k_norm));
    if (expand_kv) {
        ck = sb_expand_heads(ctx, ck, D, Nkv, Nh, enc_S, B);
        cv = sb_expand_heads(ctx, cv, D, Nkv, Nh, enc_S, B);
    }
    cq = ggml_permute(ctx, cq, 0, 2, 1, 3);
    ck = ggml_permute(ctx, ck, 0, 2, 1, 3);
    cv = ggml_permute(ctx, cv, 0, 2, 1, 3);
    ggml_tensor * cattn = dit_attn_f32(ctx, cq, ck, cv, in.t_ca, scale);
    cattn               = ggml_reshape_3d(ctx, cattn, (int64_t) Nh * D, S, B);
    hidden              = ggml_add(ctx, hidden, ad->apply(ctx, ly->ca_o_proj, li, DIT_CA_O, cattn));

    // ── MLP ──
    res               = hidden;
    ggml_tensor * nff = dit_ggml_rms_norm_weighted(ctx, hidden, ly->mlp_norm, c.rms_norm_eps);
    nff               = dit_ggml_adaln(ctx, nff, scale_ff, shift_ff, M->m.scalar_one);
    ggml_tensor * g   = ad->apply(ctx, ly->gate_proj, li, DIT_MLP_GATE, nff);
    ggml_tensor * u   = ad->apply(ctx, ly->up_proj, li, DIT_MLP_UP, nff);
    ggml_tensor * ff  = ggml_swiglu_split(ctx, g, u);
    ggml_tensor * fo  = ad->apply(ctx, ly->down_proj, li, DIT_MLP_DOWN, ff);
    return dit_ggml_gated_add(ctx, res, fo, gate_ff);
}

// ─── pre / post stacks (design C2: they belong to the first / last segment) ──

static ggml_tensor * sb_cond(ggml_context * ctx, DitTrainModel * M, const SbInputs & in) {
    return ggml_add(ctx, ggml_mul_mat(ctx, M->m.cond_emb_w, in.t_enc), M->m.cond_emb_b);  // [H,enc_S,B]
}

static ggml_tensor * sb_proj_in(ggml_context * ctx, DitTrainModel * M, const SbInputs & in, int T, int B) {
    const DiTGGMLConfig & c = M->m.cfg;
    const int             S = T / c.patch_size;
    ggml_tensor * patched   = ggml_reshape_3d(ctx, in.t_input, (int64_t) c.in_channels * c.patch_size, S, B);
    return ggml_add(ctx, ggml_mul_mat(ctx, M->m.proj_in_w, patched), M->m.proj_in_b);  // [H,S,B]
}

static ggml_tensor * sb_head(ggml_context * ctx, DitTrainModel * M, ggml_tensor * hidden, const SbInputs & in, int T,
                             int B) {
    const DiTGGMLConfig & c = M->m.cfg;
    const int             H = c.hidden_size;
    ggml_tensor * oss       = ggml_reshape_1d(ctx, M->m.out_scale_shift, 2 * H);
    ggml_tensor * out_shift = ggml_add(ctx, in.t_temb, ggml_view_1d(ctx, oss, H, 0));
    ggml_tensor * out_scale = ggml_add(ctx, in.t_temb, ggml_view_1d(ctx, oss, H, (size_t) H * sizeof(float)));
    ggml_tensor * nout      = dit_ggml_rms_norm_weighted(ctx, hidden, M->m.norm_out, c.rms_norm_eps);
    nout                    = dit_ggml_adaln(ctx, nout, out_scale, out_shift, M->m.scalar_one);
    ggml_tensor * out       = ggml_add(ctx, ggml_mul_mat(ctx, M->m.proj_out_w, nout), M->m.proj_out_b);
    return ggml_reshape_3d(ctx, out, c.out_channels, T, B);  // [Oc,T,B]
}

// Design B4 reduction: sum(per_elem (*) lw). `lw` [1,T,B] (or [1,T] for a single
// sample) carries the per-element weight AND the valid-frame mask, so padded
// frames contribute exactly zero to both the loss and the gradients, and the
// scalar t_lossgrad is left as a pure grad-accum scale.
static ggml_tensor * sb_loss(ggml_context * ctx, ggml_tensor * vpred, ggml_tensor * vtgt, ggml_tensor * lw) {
    ggml_tensor * pe   = ggml_sqr(ctx, ggml_sub(ctx, vpred, vtgt));
    ggml_tensor * loss = ggml_sum(ctx, ggml_mul(ctx, pe, lw));
    ggml_set_name(loss, "sb_loss");
    ggml_set_output(loss);
    return loss;
}

// lm_optim_fill_gacc() asserts that every PARAM-flagged node is a known
// optimizer slot, and a checkpoint boundary deliberately is not one. The flag is
// hidden for the duration of that call — the node list is already fixed by then,
// so nothing else changes — and the boundary is then handed its own persistent
// accumulator. lm-optim.h is untouched; in ENGINE-B this wrapper belongs in
// dit-train-run.h next to the segment driver.
static void sb_fill_gacc(const LmOptim * o, ggml_cgraph * gf, ggml_tensor * bnd, ggml_tensor * bnd_grad,
                         std::vector<ggml_tensor *> * gacc) {
    const bool was_param = bnd && (bnd->flags & GGML_TENSOR_FLAG_PARAM) != 0;
    if (was_param) {
        bnd->flags &= ~(int32_t) GGML_TENSOR_FLAG_PARAM;
    }
    lm_optim_fill_gacc(o, gf, gacc);
    if (was_param) {
        bnd->flags |= GGML_TENSOR_FLAG_PARAM;
        const int n = ggml_graph_n_nodes(gf);
        for (int i = 0; i < n; i++) {
            if (ggml_graph_node(gf, i) == bnd) {
                (*gacc)[(size_t) i] = bnd_grad;
            }
        }
    }
}

// FINDING (S-B1, load-bearing for ENGINE-A). dit_mirror_slots() deliberately
// leaves proj_in / cond_emb at their native dtype (dit-mirror.h:94-99: they
// consume graph INPUTS, never an activation gradient), i.e. BF16 on a BF16 base.
// Adding a batch axis gives their mul_mat a src1 with ne2 > 1, and
// ggml_cuda_mul_mat() dispatches on exactly that: `src1->ne[2]*src1->ne[3] > 1`
// routes to ggml_cuda_mul_mat_batched_cublas (ggml-cuda.cu:2618-2621), a BF16
// GEMM, instead of the F32-accurate ggml_cuda_mul_mat_f the single-sample path
// gets. Measured on cond_emb (enc_S=48 columns, past the mmf column limit): the
// batched result moves by 1.25e-1 on a 36.1 tensor == 3.5e-3 relative == BF16
// eps, which swamps the batch-axis identity these gates measure. proj_in
// (32 columns) stays inside mmf and is bit-identical.
//
// Consequences for the real work: (a) invariant §2.3.1 (`--batch 1 --ckpt 0`
// bit-identical to today) still holds, because B=1 keeps ne2==1 and the same
// dispatch; (b) DIT_MIRROR_BF16, which keeps every trainable matmul weight in
// BF16, will change numerics the moment batching is switched on — it needs
// re-validating, not assuming; (c) the §2.3.2 "<=1e-6 max-abs" bar is only
// meaningful against an all-F32 weight set.
//
// The spike therefore promotes proj_in / cond_emb to F32 into its own buffer for
// the duration of the run. Nothing in dit-mirror.h changes.
static bool sb_promote_f32(DitTrainModel * M, ggml_context ** ctx_out, ggml_backend_buffer_t * buf_out, int * n_done) {
    ggml_tensor ** slots[4] = { &M->m.proj_in_w, &M->m.proj_in_b, &M->m.cond_emb_w, &M->m.cond_emb_b };
    ggml_init_params p      = { 16 * ggml_tensor_overhead(), nullptr, true };
    ggml_context *   ctx    = ggml_init(p);
    if (!ctx) {
        return false;
    }
    ggml_tensor * dst[4] = { nullptr, nullptr, nullptr, nullptr };
    for (int i = 0; i < 4; i++) {
        ggml_tensor * s = *slots[i];
        if (!s || s->type == GGML_TYPE_F32) {
            continue;
        }
        dst[i] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s->ne[0], s->ne[1], s->ne[2], s->ne[3]);
        ggml_set_name(dst[i], s->name);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, M->backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<uint8_t> raw;
    std::vector<float>   f32;
    for (int i = 0; i < 4; i++) {
        if (!dst[i]) {
            continue;
        }
        ggml_tensor * s  = *slots[i];
        const size_t  ne = (size_t) ggml_nelements(s);
        raw.resize(ggml_nbytes(s));
        ggml_backend_tensor_get(s, raw.data(), 0, raw.size());
        f32.resize(ne);
        if (s->type == GGML_TYPE_BF16) {
            const uint16_t * u = (const uint16_t *) raw.data();
            for (size_t j = 0; j < ne; j++) {
                const uint32_t bits = (uint32_t) u[j] << 16;
                memcpy(&f32[j], &bits, 4);
            }
        } else {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), f32.data(), (int64_t) ne);
        }
        ggml_backend_tensor_set(dst[i], f32.data(), 0, ne * sizeof(float));
        *slots[i] = dst[i];
        (*n_done)++;
    }
    *ctx_out = ctx;
    *buf_out = buf;
    return true;
}

// ─── the driver ─────────────────────────────────────────────────────────────

struct SbArgs {
    std::string dit_path, tensors_dir;
    uint64_t    seed      = 42;
    int         T         = 64;
    int         enc_S     = 48;
    int         B         = 3;
    int         rank      = 8;
    bool        naive_bwd = false;  // opt-in: prove the GQA/ne3 backward abort
    // Run on the CPU backend. ggml's CPU mul_mat is one kernel that loops over
    // ne2/ne3, so the batched and single-sample graphs execute the SAME code with
    // the SAME accumulation order — which is what makes an exact (bar 1e-6)
    // batch-axis identity measurable at all. On CUDA the batch axis itself picks
    // a different cuBLAS routine (see sb_promote_f32), putting a ~1e-5 relative
    // floor under any batched-vs-single comparison.
    bool        cpu       = false;
};

static int spike_batch(const SbArgs & a) {
    std::vector<SbResult> rs;
    fprintf(stderr, "[spike-batch] gates S-B1 / S-B2 / S-C1 (docs/plans/2026-07-29-dit-batching-checkpointing.md §4)\n");

    // ── data ─────────────────────────────────────────────────────────────
    std::vector<DitSample> samples;
    {
        std::string err;
        if (!dit_scan_samples(a.tensors_dir.c_str(), 0, &samples, &err)) {
            fprintf(stderr, "[spike-batch] DATA FAIL %s\n", err.c_str());
            return 1;
        }
    }
    if ((int) samples.size() < a.B) {
        fprintf(stderr, "[spike-batch] DATA FAIL need >= %d songs for a B=%d batch (design B2), found %d\n", a.B, a.B,
                (int) samples.size());
        return 1;
    }
    const int enc_H = samples[0].enc_H;
    for (int b = 0; b < a.B; b++) {
        if (samples[(size_t) b].enc_H != enc_H) {
            fprintf(stderr, "[spike-batch] DATA FAIL enc_H differs across songs (%d vs %d)\n", enc_H,
                    samples[(size_t) b].enc_H);
            return 1;
        }
    }

    // ── model: mirror only the two layers this spike trains ──────────────
    DitTrainModel M;
    int           L = 0;
    {
        std::string err;
        if (!dit_train_load(&M, a.dit_path.c_str(), 0, &err)) {
            fprintf(stderr, "[spike-batch] LOAD FAIL %s\n", err.c_str());
            dit_train_free(&M);
            return 1;
        }
        L = M.m.cfg.n_layers;
        if (a.cpu) {
            M.backend = cpu_backend_new(16);
        } else if (!dit_train_backend_init(&M, &err)) {
            fprintf(stderr, "[spike-batch] BACKEND FAIL %s\n", err.c_str());
            dit_train_free(&M);
            return 1;
        }
        if (!M.backend) {
            fprintf(stderr, "[spike-batch] BACKEND FAIL no compute backend\n");
            dit_train_free(&M);
            return 1;
        }
        fprintf(stderr, "[spike-batch] backend: %s\n", ggml_backend_name(M.backend));
        // lora_lo = L-2: only the two layers under test are promoted to F32, so
        // the mirror stays near the base size and the spike fits any GPU that can
        // already run the trainer.
        if (!dit_build_mirror(&M, L - 2, DIT_MIRROR_F32, &err)) {
            fprintf(stderr, "[spike-batch] MIRROR FAIL %s\n", err.c_str());
            dit_train_free(&M);
            return 1;
        }
    }
    ggml_context *        ctx_pf32 = nullptr;
    ggml_backend_buffer_t buf_pf32 = nullptr;
    int                   n_pf32   = 0;
    if (!sb_promote_f32(&M, &ctx_pf32, &buf_pf32, &n_pf32)) {
        fprintf(stderr, "[spike-batch] PROMOTE FAIL cannot promote proj_in/cond_emb to F32\n");
        dit_train_free(&M);
        return 1;
    }
    fprintf(stderr, "[spike-batch] promoted %d proj_in/cond_emb tensors to F32 (see sb_promote_f32 — the batched "
                    "CUDA mul_mat dispatch drops BF16 weights to a BF16 GEMM)\n",
            n_pf32);

    const DiTGGMLConfig & c = M.m.cfg;
    const int H = c.hidden_size, Oc = c.out_channels, P = c.patch_size, Ic = c.in_channels;
    {
        BackendPair bp;
        bp.backend = M.backend;
        // A single-backend sched in --cpu mode (backend_sched_new collapses to
        // n == 1 when the two match), so nothing can silently land elsewhere.
        bp.cpu_backend = a.cpu ? M.backend : M.cpu;
        bp.has_gpu     = !a.cpu;
        M.sched        = backend_sched_new(bp, 65536);
    }

    const int B  = a.B;
    int       T  = a.T - (a.T % P);
    if (T < P) {
        T = P;
    }
    const int S     = T / P;
    int       enc_S = std::min(a.enc_S, samples[0].enc_S);
    for (int b = 0; b < B; b++) {
        enc_S = std::min(enc_S, samples[(size_t) b].enc_S);
    }
    if (enc_S < 8) {
        fprintf(stderr, "[spike-batch] DATA FAIL enc_S too small (%d)\n", enc_S);
        ggml_backend_buffer_free(buf_pf32);
        ggml_free(ctx_pf32);
        dit_train_free(&M);
        return 1;
    }
    fprintf(stderr, "[spike-batch] %d layers H=%d Nh=%d/%d D=%d P=%d | B=%d T=%d S=%d enc_S=%d enc_H=%d rank=%d\n", L,
            H, c.n_heads, c.n_kv_heads, c.head_dim, P, B, T, S, enc_S, enc_H, a.rank);

    // ── static buffers (1-D bases; every graph tensor is a contiguous view) ─
    ggml_context * ctxs;
    {
        ggml_init_params p = { 32 * ggml_tensor_overhead(), nullptr, true };
        ctxs               = ggml_init(p);
    }
    ggml_tensor * b_input = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, (int64_t) Ic * T * B);
    ggml_tensor * b_enc   = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, (int64_t) enc_H * enc_S * B);
    ggml_tensor * b_pos   = ggml_new_tensor_1d(ctxs, GGML_TYPE_I32, (int64_t) S * B);
    ggml_tensor * b_temb  = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, (int64_t) H * B);
    ggml_tensor * b_tproj = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, (int64_t) 6 * H * B);
    ggml_tensor * b_sa    = ggml_new_tensor_1d(ctxs, GGML_TYPE_F16, (int64_t) S * S);
    ggml_tensor * b_ca    = ggml_new_tensor_1d(ctxs, GGML_TYPE_F16, (int64_t) enc_S * S * B);
    ggml_tensor * b_vtgt  = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, (int64_t) Oc * T * B);
    ggml_tensor * b_lw    = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, (int64_t) T * B);
    ggml_tensor * t_bnd   = ggml_new_tensor_3d(ctxs, GGML_TYPE_F32, H, S, B);  // S-C1 boundary activation
    ggml_tensor * t_bndg  = ggml_new_tensor_3d(ctxs, GGML_TYPE_F32, H, S, B);  // S-C1 boundary gradient
    ggml_tensor * t_adamw    = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 7);
    ggml_tensor * t_lossgrad = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 1);
    ggml_tensor * t_clip     = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 1);
    ggml_tensor * t_eps      = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 1);
    ggml_tensor * t_gnorm2   = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 1);
    for (ggml_tensor * t : { b_input, b_enc, b_pos, b_temb, b_tproj, b_sa, b_ca, b_vtgt, b_lw }) {
        ggml_set_input(t);
    }
    ggml_backend_buffer_t buf_static = ggml_backend_alloc_ctx_tensors(ctxs, M.backend);
    if (!buf_static) {
        fprintf(stderr, "[spike-batch] VRAM FAIL static input allocation failed\n");
        ggml_free(ctxs);
        ggml_backend_buffer_free(buf_pf32);
        ggml_free(ctx_pf32);
        dit_train_free(&M);
        return 1;
    }
    ggml_set_name(t_bnd, "seg_boundary");
    ggml_set_name(t_bndg, "seg_boundary_grad");
    {
        const float epsv = 1e-6f, clipv = 1.0f, lg = 1.0f;
        ggml_backend_tensor_set(t_eps, &epsv, 0, 4);
        ggml_backend_tensor_set(t_clip, &clipv, 0, 4);
        ggml_backend_tensor_set(t_lossgrad, &lg, 0, 4);
    }

    // ── host-side batch assembly (design B2/B3/B4) ───────────────────────
    //
    // B DIFFERENT songs; per-ELEMENT t, CFG state, encoder-padding mask and
    // valid-frame count. Element 1 is the CFG-dropped one (its enc slice is
    // null_condition_emb); element 2 is short, so its tail frames carry weight 0.
    LmRng rng;
    lm_rng_seed(&rng, a.seed);
    std::vector<float>    input_buf((size_t) Ic * T * B, 0.0f);
    std::vector<float>    vtgt_buf((size_t) Oc * T * B, 0.0f);
    std::vector<float>    enc_buf((size_t) enc_H * enc_S * B, 0.0f);
    std::vector<float>    lw_buf((size_t) T * B, 0.0f);
    std::vector<int32_t>  pos_buf((size_t) S * B, 0);
    std::vector<uint16_t> sa_buf, ca_buf, ca_all((size_t) enc_S * S * B, 0);
    std::vector<float>    ts((size_t) B, 0.5f), ws((size_t) B, 1.0f);
    std::vector<int>      nvalid((size_t) B, T);
    const bool            cfg_ok = ((int) M.null_cond.size() == enc_H);

    for (int b = 0; b < B; b++) {
        ts[(size_t) b] = dit_sample_t(&rng, -0.4f, 1.0f, 0.0f, 1.0f);
        ws[(size_t) b] = dit_flow_snr_w(ts[(size_t) b], 0.0f, 5.0f);
    }
    {
        double wbar = 0.0;
        for (int b = 0; b < B; b++) {
            wbar += (double) ws[(size_t) b];
        }
        wbar /= (double) B;
        for (int b = 0; b < B; b++) {
            ws[(size_t) b] = (float) ((double) ws[(size_t) b] / std::max(1e-12, wbar));
        }
    }
    nvalid[(size_t) (B - 1)] = std::max(P, T - 8 * P);  // one short element

    std::vector<float> flow_xt, flow_v;
    for (int b = 0; b < B; b++) {
        const DitSample & s = samples[(size_t) b];
        LmRng             noise;
        lm_rng_seed(&noise, a.seed ^ (0x9e3779b97f4a7c15ull * (uint64_t) (b + 1)));
        const int len = std::min(T, s.T - (s.T % P));
        dit_flow_target(s.lat.data(), (size_t) len * (size_t) s.Oc, ts[(size_t) b], &noise, &flow_xt, &flow_v);
        for (int f = 0; f < T; f++) {
            const int   sf  = f % len;
            float *     dst = &input_buf[((size_t) b * T + (size_t) f) * (size_t) Ic];
            memcpy(dst, &s.ctxl[(size_t) sf * (size_t) s.Cc], (size_t) s.Cc * sizeof(float));
            memcpy(dst + s.Cc, &flow_xt[(size_t) sf * (size_t) s.Oc], (size_t) s.Oc * sizeof(float));
            memcpy(&vtgt_buf[((size_t) b * T + (size_t) f) * (size_t) Oc], &flow_v[(size_t) sf * (size_t) s.Oc],
                   (size_t) Oc * sizeof(float));
        }
        // encoder conditioning: element 1 is CFG-dropped
        const bool drop = (b == 1) && cfg_ok;
        for (int i = 0; i < enc_S; i++) {
            float * dst = &enc_buf[((size_t) b * enc_S + (size_t) i) * (size_t) enc_H];
            if (drop) {
                memcpy(dst, M.null_cond.data(), (size_t) enc_H * sizeof(float));
            } else {
                memcpy(dst, &s.enc[(size_t) i * (size_t) s.enc_H], (size_t) enc_H * sizeof(float));
            }
        }
        // per-element encoder-padding mask: distinct valid lengths
        std::vector<float> emask((size_t) enc_S, 0.0f);
        const int          evalid = drop ? enc_S : std::max(4, enc_S - 5 * b);
        for (int i = 0; i < evalid; i++) {
            emask[(size_t) i] = 1.0f;
        }
        dit_ca_mask(enc_S, S, emask, &ca_buf);
        memcpy(&ca_all[(size_t) b * (size_t) enc_S * (size_t) S], ca_buf.data(), ca_buf.size() * sizeof(uint16_t));
        // design B4 flow_snr reduction: (w_i/wbar) / (n_valid_element * B)
        const float wl = ws[(size_t) b] / (float) ((int64_t) nvalid[(size_t) b] * (int64_t) B);
        for (int f = 0; f < T; f++) {
            lw_buf[(size_t) b * T + (size_t) f] = (f < nvalid[(size_t) b]) ? wl : 0.0f;
        }
        for (int i = 0; i < S; i++) {
            pos_buf[(size_t) b * S + (size_t) i] = i;
        }
        fprintf(stderr, "[spike-batch]   element %d: song=%s t=%.4f w/wbar=%.4f cfg_drop=%s enc_valid=%d n_valid=%d\n",
                b, s.id.c_str(), (double) ts[(size_t) b], (double) ws[(size_t) b], drop ? "yes" : "no", evalid,
                nvalid[(size_t) b]);
    }
    dit_sa_mask(S, c.sliding_window, &sa_buf);

    // temb / tproj per element (design B3: [H,B] / [6H,B])
    std::vector<std::vector<float>> tbv, tpv;
    if (!dit_train_temb(&M, ts, &tbv, &tpv)) {
        fprintf(stderr, "[spike-batch] TEMB FAIL temb precompute failed\n");
        ggml_backend_buffer_free(buf_static);
        ggml_free(ctxs);
        ggml_backend_buffer_free(buf_pf32);
        ggml_free(ctx_pf32);
        dit_train_free(&M);
        return 1;
    }
    std::vector<float> temb_buf((size_t) H * B, 0.0f), tproj_buf((size_t) 6 * H * B, 0.0f);
    for (int b = 0; b < B; b++) {
        memcpy(&temb_buf[(size_t) b * H], tbv[(size_t) b].data(), (size_t) H * sizeof(float));
        memcpy(&tproj_buf[(size_t) b * 6 * H], tpv[(size_t) b].data(), (size_t) 6 * H * sizeof(float));
    }

    // Every input is re-uploaded before every compute (§3.0 clobber rule).
    //
    // `rev` uploads the SAME B elements in reversed order. Shapes, kernels and
    // cuBLAS routine are then bit-identical to the normal upload, so comparing
    // element b of a reversed run against element B-1-b of a normal one is a
    // route-identical test for cross-element contamination — the thing the batch
    // axis can actually get wrong. It is the measurement that still has teeth on
    // CUDA, where the batched-vs-single comparison sits on a GEMM-route floor.
    std::vector<float>    rv_f;
    std::vector<uint16_t> rv_u;
    auto upload_all = [&](bool rev) {
        auto put_f = [&](ggml_tensor * dst, const std::vector<float> & src, size_t per) {
            if (!rev) {
                ggml_backend_tensor_set(dst, src.data(), 0, src.size() * sizeof(float));
                return;
            }
            rv_f.resize(src.size());
            for (int b = 0; b < B; b++) {
                memcpy(&rv_f[(size_t) b * per], &src[(size_t) (B - 1 - b) * per], per * sizeof(float));
            }
            ggml_backend_tensor_set(dst, rv_f.data(), 0, rv_f.size() * sizeof(float));
        };
        put_f(b_input, input_buf, (size_t) Ic * T);
        put_f(b_vtgt, vtgt_buf, (size_t) Oc * T);
        put_f(b_enc, enc_buf, (size_t) enc_H * enc_S);
        put_f(b_lw, lw_buf, (size_t) T);
        put_f(b_temb, temb_buf, (size_t) H);
        put_f(b_tproj, tproj_buf, (size_t) 6 * H);
        ggml_backend_tensor_set(b_pos, pos_buf.data(), 0, pos_buf.size() * sizeof(int32_t));
        ggml_backend_tensor_set(b_sa, sa_buf.data(), 0, sa_buf.size() * sizeof(uint16_t));
        if (rev) {
            rv_u.resize(ca_all.size());
            const size_t per = (size_t) enc_S * (size_t) S;
            for (int b = 0; b < B; b++) {
                memcpy(&rv_u[(size_t) b * per], &ca_all[(size_t) (B - 1 - b) * per], per * sizeof(uint16_t));
            }
            ggml_backend_tensor_set(b_ca, rv_u.data(), 0, rv_u.size() * sizeof(uint16_t));
        } else {
            ggml_backend_tensor_set(b_ca, ca_all.data(), 0, ca_all.size() * sizeof(uint16_t));
        }
    };

    // ── views ────────────────────────────────────────────────────────────
    auto mk_batched = [&](ggml_context * ctx) {
        SbInputs in;
        in.t_input = ggml_view_3d(ctx, b_input, Ic, T, B, (size_t) Ic * 4, (size_t) Ic * T * 4, 0);
        in.t_enc   = ggml_view_3d(ctx, b_enc, enc_H, enc_S, B, (size_t) enc_H * 4, (size_t) enc_H * enc_S * 4, 0);
        in.t_pos   = ggml_view_1d(ctx, b_pos, (int64_t) S * B, 0);
        in.t_temb  = ggml_view_3d(ctx, b_temb, H, 1, B, (size_t) H * 4, (size_t) H * 4, 0);
        in.t_tproj = ggml_view_3d(ctx, b_tproj, 6 * H, 1, B, (size_t) 6 * H * 4, (size_t) 6 * H * 4, 0);
        in.t_sa    = ggml_view_2d(ctx, b_sa, S, S, (size_t) S * sizeof(ggml_fp16_t), 0);
        in.t_ca    = ggml_view_4d(ctx, b_ca, enc_S, S, 1, B, (size_t) enc_S * sizeof(ggml_fp16_t),
                                  (size_t) enc_S * S * sizeof(ggml_fp16_t), (size_t) enc_S * S * sizeof(ggml_fp16_t), 0);
        in.t_vtgt  = ggml_view_3d(ctx, b_vtgt, Oc, T, B, (size_t) Oc * 4, (size_t) Oc * T * 4, 0);
        in.t_lw    = ggml_view_3d(ctx, b_lw, 1, T, B, 4, (size_t) T * 4, 0);
        return in;
    };
    // element b of the SAME buffers, in the shapes the production single-sample
    // graph (dit-train-graph.h) expects.
    auto mk_single = [&](ggml_context * ctx, int b) {
        DitInputs in;
        in.t_input = ggml_view_2d(ctx, b_input, Ic, T, (size_t) Ic * 4, (size_t) b * Ic * T * 4);
        in.t_enc   = ggml_view_2d(ctx, b_enc, enc_H, enc_S, (size_t) enc_H * 4, (size_t) b * enc_H * enc_S * 4);
        in.t_pos   = ggml_view_1d(ctx, b_pos, S, (size_t) b * S * 4);
        in.t_temb  = ggml_view_1d(ctx, b_temb, H, (size_t) b * H * 4);
        in.t_tproj = ggml_view_1d(ctx, b_tproj, 6 * H, (size_t) b * 6 * H * 4);
        in.t_sa    = ggml_view_2d(ctx, b_sa, S, S, (size_t) S * sizeof(ggml_fp16_t), 0);
        in.t_ca    = ggml_view_2d(ctx, b_ca, enc_S, S, (size_t) enc_S * sizeof(ggml_fp16_t),
                                  (size_t) b * enc_S * S * sizeof(ggml_fp16_t));
        in.t_vtgt  = ggml_view_2d(ctx, b_vtgt, Oc, T, (size_t) Oc * 4, (size_t) b * Oc * T * 4);
        in.t_cw    = nullptr;
        return in;
    };
    auto mk_single_lw = [&](ggml_context * ctx, int b) {
        return ggml_view_2d(ctx, b_lw, 1, T, 4, (size_t) b * T * 4);
    };

    // ── adapter over the two layers under test ───────────────────────────
    DitAdapterLora lora;
    {
        DitAdapterCfg cfg;
        cfg.rank    = a.rank;
        cfg.alpha   = (float) (2 * a.rank);
        cfg.seed    = a.seed ^ 0x1234ull;
        cfg.b_sigma = 1e-2f;  // B == 0 would make dL/dA identically zero
        std::string err;
        if (!lora.init(&M.m, M.backend, L - 2, L, cfg, &err)) {
            fprintf(stderr, "[spike-batch] ADAPTER FAIL %s\n", err.c_str());
            ggml_backend_buffer_free(buf_static);
            ggml_free(ctxs);
            ggml_backend_buffer_free(buf_pf32);
            ggml_free(ctx_pf32);
            dit_train_free(&M);
            return 1;
        }
    }
    LmOptim opt;
    {
        std::string err;
        if (!lm_optim_init(&opt, lora.params(), M.backend, &err)) {
            fprintf(stderr, "[spike-batch] OPT FAIL %s\n", err.c_str());
            lora.free();
            ggml_backend_buffer_free(buf_static);
            ggml_free(ctxs);
            ggml_backend_buffer_free(buf_pf32);
            ggml_free(ctx_pf32);
            dit_train_free(&M);
            return 1;
        }
    }
    opt.t_adamw    = t_adamw;
    opt.t_lossgrad = t_lossgrad;
    opt.t_clip     = t_clip;
    opt.t_eps      = t_eps;
    opt.t_gnorm2   = t_gnorm2;

    std::vector<uint8_t> arena((size_t) 512 << 20);

    auto read_accs = [&](std::vector<std::vector<float>> * out) {
        out->assign(opt.acc.size(), std::vector<float>());
        for (size_t j = 0; j < opt.acc.size(); j++) {
            (*out)[j].resize((size_t) ggml_nelements(opt.acc[j]));
            ggml_backend_tensor_get(opt.acc[j], (*out)[j].data(), 0, (*out)[j].size() * sizeof(float));
        }
    };
    // max_j ( max_k |x-y| / max_k |y| ) over the parameter tensors.
    auto cmp_accs = [&](const std::vector<std::vector<float>> & x, const std::vector<std::vector<float>> & y,
                        std::string * worst) {
        double best = 0.0;
        *worst      = "-";
        for (size_t j = 0; j < y.size() && j < x.size(); j++) {
            double num = 0.0, den = 0.0;
            for (size_t k = 0; k < y[j].size() && k < x[j].size(); k++) {
                num = std::max(num, fabs((double) x[j][k] - (double) y[j][k]));
                den = std::max(den, fabs((double) y[j][k]));
            }
            const double rel = num / std::max(den, 1e-30);
            if (rel > best) {
                best   = rel;
                *worst = opt.params[j]->name;
            }
        }
        return best;
    };

    const int lo = L - 2, hi = L;

    // ── S-B1: batched forward parity ─────────────────────────────────────
    {
        // Stage taps, so a divergence can be localised instead of guessed at.
        // Names match dit-train-graph.h's own probes on the single-sample side.
        const char * stage_names[3] = { "hidden_after_proj_in", "enc_after_cond_emb", "velocity" };
        std::vector<float> vsin((size_t) Oc * T);
        std::vector<std::vector<float>> stage_bat(3), stage_sin(3);
        auto run_batched = [&](bool expand_kv, std::vector<float> * out, int * nodes,
                               std::vector<std::vector<float>> * stages, bool rev = false) -> bool {
            upload_all(rev);
            ggml_init_params ip  = { arena.size(), arena.data(), true };
            ggml_context *   ctx = ggml_init(ip);
            ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 65536, false);
            SbInputs         in  = mk_batched(ctx);
            ggml_tensor *    h   = sb_proj_in(ctx, &M, in, T, B);
            ggml_tensor *    e   = sb_cond(ctx, &M, in);
            ggml_tensor *    h0  = h;
            for (int li = lo; li < hi; li++) {
                h = sb_layer(ctx, &M, li, h, e, in, &lora, S, enc_S, B, expand_kv);
            }
            ggml_tensor * vel = sb_head(ctx, &M, h, in, T, B);
            ggml_tensor * taps3[3] = { h0, e, vel };
            for (int i = 0; i < 3; i++) {
                ggml_set_output(taps3[i]);
                for (ggml_tensor * p = taps3[i]->view_src; p; p = p->view_src) {
                    ggml_set_output(p);
                }
                ggml_build_forward_expand(gf, taps3[i]);
            }
            *nodes = ggml_graph_n_nodes(gf);
            ggml_backend_sched_reset(M.sched);
            const bool ok = ggml_backend_sched_graph_compute(M.sched, gf) == GGML_STATUS_SUCCESS;
            if (ok) {
                out->resize((size_t) Oc * T * B);
                ggml_backend_tensor_get(vel, out->data(), 0, out->size() * sizeof(float));
                if (stages) {
                    for (int i = 0; i < 3; i++) {
                        (*stages)[(size_t) i].resize((size_t) ggml_nelements(taps3[i]));
                        ggml_backend_tensor_get(taps3[i], (*stages)[(size_t) i].data(), 0,
                                                (*stages)[(size_t) i].size() * sizeof(float));
                    }
                }
            }
            ggml_free(ctx);
            return ok;
        };

        int  nodes_bcast = 0, nodes_exp = 0;
        std::vector<float> v_bcast, v_exp;
        const bool ok_bcast = run_batched(false, &v_bcast, &nodes_bcast, &stage_bat);
        const bool ok_exp   = run_batched(true, &v_exp, &nodes_exp, nullptr);
        // route-identical control: same shapes, elements in reversed order
        int                nodes_rev = 0;
        std::vector<float> v_rev;
        const bool         ok_rev = run_batched(false, &v_rev, &nodes_rev, nullptr, true);
        double             perm_max = 0.0;
        if (ok_bcast && ok_rev) {
            const size_t per = (size_t) Oc * T;
            for (int b = 0; b < B; b++) {
                for (size_t k = 0; k < per; k++) {
                    perm_max = std::max(perm_max, fabs((double) v_rev[(size_t) b * per + k] -
                                                       (double) v_bcast[(size_t) (B - 1 - b) * per + k]));
                }
            }
        }

        double worst_bcast = 0.0, worst_exp = 0.0, ref_mag = 0.0;
        bool   ok_single   = ok_bcast && ok_exp && ok_rev;
        std::string per_el;
        for (int b = 0; b < B && ok_single; b++) {
            upload_all(false);
            ggml_init_params ip  = { arena.size(), arena.data(), true };
            ggml_context *   ctx = ggml_init(ip);
            ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 65536, false);
            DitInputs        in  = mk_single(ctx, b);
            DitTaps          taps;
            ggml_tensor *    vel = dit_train_forward(ctx, &M, &lora, in, T, enc_S, &taps, lo, hi);
            ggml_set_output(vel);
            ggml_build_forward_expand(gf, vel);
            ggml_backend_sched_reset(M.sched);
            ok_single = ggml_backend_sched_graph_compute(M.sched, gf) == GGML_STATUS_SUCCESS;
            if (ok_single) {
                ggml_backend_tensor_get(vel, vsin.data(), 0, vsin.size() * sizeof(float));
                double mb = 0.0, me = 0.0;
                for (size_t k = 0; k < vsin.size(); k++) {
                    const double r = (double) vsin[k];
                    ref_mag        = std::max(ref_mag, fabs(r));
                    mb = std::max(mb, fabs((double) v_bcast[(size_t) b * vsin.size() + k] - r));
                    me = std::max(me, fabs((double) v_exp[(size_t) b * vsin.size() + k] - r));
                }
                worst_bcast = std::max(worst_bcast, mb);
                worst_exp   = std::max(worst_exp, me);
                char eb[96];
                snprintf(eb, sizeof(eb), "e%d=%.3e/%.3e ", b, mb, me);
                per_el += eb;

                // stage localisation
                std::string line;
                for (int st = 0; st < 3; st++) {
                    ggml_tensor * rt = nullptr;
                    for (size_t i = 0; i < taps.v.size(); i++) {
                        if (taps.v[i].first == stage_names[st]) {
                            rt = taps.v[i].second;
                        }
                    }
                    if (!rt || stage_bat[(size_t) st].empty()) {
                        line += std::string(stage_names[st]) + "=MISSING ";
                        continue;
                    }
                    const size_t n = (size_t) ggml_nelements(rt);
                    stage_sin[(size_t) st].resize(n);
                    ggml_backend_tensor_get(rt, stage_sin[(size_t) st].data(), 0, n * sizeof(float));
                    double mx = 0.0, mg = 0.0;
                    for (size_t k = 0; k < n && (size_t) b * n + k < stage_bat[(size_t) st].size(); k++) {
                        const double r = (double) stage_sin[(size_t) st][k];
                        mg = std::max(mg, fabs(r));
                        mx = std::max(mx, fabs((double) stage_bat[(size_t) st][(size_t) b * n + k] - r));
                    }
                    char sb2[128];
                    snprintf(sb2, sizeof(sb2), "%s=%.3e(|%.3g|) ", stage_names[st], mx, mg);
                    line += sb2;
                }
                fprintf(stderr, "[spike-batch]   stage diff element %d: %s\n", b, line.c_str());
            }
            ggml_free(ctx);
        }

        char d[704];
        snprintf(d, sizeof(d),
                 "[%s] B=%d batched forward (%d nodes broadcast-KV / %d nodes expanded-KV) vs %d single-sample "
                 "dit_train_forward on layers %d-%d, per-element max|delta| (bcast/exp): %s| worst bcast=%.3e "
                 "exp=%.3e (bar 1e-6), |velocity|max=%.4f; route-identical control (batch order reversed) "
                 "max|delta|=%.3e",
                 ggml_backend_name(M.backend), B, nodes_bcast, nodes_exp, B, lo, hi - 1, per_el.c_str(), worst_bcast,
                 worst_exp, ref_mag, perm_max);
        sb_report(rs, "S-B1", ok_single && worst_bcast <= 1e-6 && worst_exp <= 1e-6 && perm_max == 0.0, d);
    }

    // ── S-B2: batched backward parity ────────────────────────────────────
    //
    // Reference: three single-sample fwd+bwd graphs, built and computed one after
    // the other, accumulating into the SAME LmOptim::acc[] with no zeroing in
    // between — the design-C1 accumulation rule, and by construction the B4
    // weighted sum (each element's graph carries its own [1,T] weight slice).
    {
        std::vector<std::vector<float>> gref, gbat;
        bool                            ok = true;
        int                             nodes_single = 0, nodes_batched = 0;

        lm_optim_zero_grad(&opt);
        for (int b = 0; b < B && ok; b++) {
            upload_all(false);
            ggml_init_params ip  = { arena.size(), arena.data(), true };
            ggml_context *   ctx = ggml_init(ip);
            ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 65536, true);
            DitInputs        in  = mk_single(ctx, b);
            ggml_tensor *    vel = dit_train_forward(ctx, &M, &lora, in, T, enc_S, nullptr, lo, hi);
            ggml_tensor *    ls  = sb_loss(ctx, vel, in.t_vtgt, mk_single_lw(ctx, b));
            ggml_set_loss(ls);
            ggml_build_forward_expand(gf, ls);
            nodes_single = ggml_graph_n_nodes(gf);
            std::vector<ggml_tensor *> ga;
            lm_optim_fill_gacc(&opt, gf, &ga);
            ggml_build_backward_expand(ctx, gf, ga.data());
            ggml_backend_sched_reset(M.sched);
            ok = ggml_backend_sched_graph_compute(M.sched, gf) == GGML_STATUS_SUCCESS;
            ggml_free(ctx);
        }
        if (ok) {
            read_accs(&gref);
        }

        if (ok) {
            lm_optim_zero_grad(&opt);
            upload_all(false);
            ggml_init_params ip  = { arena.size(), arena.data(), true };
            ggml_context *   ctx = ggml_init(ip);
            ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 65536, true);
            SbInputs         in  = mk_batched(ctx);
            ggml_tensor *    h   = sb_proj_in(ctx, &M, in, T, B);
            ggml_tensor *    e   = sb_cond(ctx, &M, in);
            for (int li = lo; li < hi; li++) {
                h = sb_layer(ctx, &M, li, h, e, in, &lora, S, enc_S, B, !a.naive_bwd);
            }
            ggml_tensor * vel = sb_head(ctx, &M, h, in, T, B);
            ggml_tensor * ls  = sb_loss(ctx, vel, in.t_vtgt, in.t_lw);
            ggml_set_loss(ls);
            ggml_build_forward_expand(gf, ls);
            nodes_batched = ggml_graph_n_nodes(gf);
            std::vector<ggml_tensor *> ga;
            lm_optim_fill_gacc(&opt, gf, &ga);
            if (a.naive_bwd) {
                fprintf(stderr, "[spike-batch] --naive-bwd: building the backward on the raw ne3-batched GQA graph; "
                                "ggml.c:6588 GGML_ASSERT(tmp->ne[3] == 1) is expected to abort here\n");
            }
            ggml_build_backward_expand(ctx, gf, ga.data());
            ggml_backend_sched_reset(M.sched);
            ok = ggml_backend_sched_graph_compute(M.sched, gf) == GGML_STATUS_SUCCESS;
            ggml_free(ctx);
            if (ok) {
                read_accs(&gbat);
            }
        }

        std::string  worst = "-";
        const double rel   = ok ? cmp_accs(gbat, gref, &worst) : 1e30;
        char         d[560];
        snprintf(d, sizeof(d),
                 "[%s] %zu adapter tensors: ONE B=%d batched backward (%d nodes) vs the sum of %d single-sample "
                 "backwards (%d nodes each) accumulated into the same persistent LmOptim::acc[] with no zeroing "
                 "between them; max relative delta %.3e on %s (bar 1e-5)",
                 ggml_backend_name(M.backend), opt.acc.size(), B, nodes_batched, B, nodes_single, rel, worst.c_str());
        sb_report(rs, "S-B2", ok && rel <= 1e-5, d);
    }

    // ── S-C1: segmented recompute vs monolithic backward ─────────────────
    //
    // Segment 1 = proj_in/cond_emb + layer lo.  Segment 2 = layer lo+1 + head +
    // the real loss.  Mechanism: (1) one no-grad forward that ggml_cpy's the
    // boundary hidden into a persistent buffer; (2) segment 2 rebuilt WITH grads,
    // its input boundary ggml_set_param'd so build_backward_expand gives it a
    // grad slot, which we point at a persistent buffer via the grad_accs array;
    // (3) segment 1 rebuilt with the spike-s3 surrogate loss sum(seg_out (*) G),
    // whose gradient w.r.t. seg_out is exactly that boundary grad.
    {
        std::vector<std::vector<float>> gmono, gseg;
        bool                            ok = true;
        int                             n_mono = 0, n_p1 = 0, n_p2 = 0, n_p3 = 0;
        double                          loss_mono = 0.0, loss_seg = 0.0, bndg_mag = 0.0;

        // monolithic
        lm_optim_zero_grad(&opt);
        {
            upload_all(false);
            ggml_init_params ip  = { arena.size(), arena.data(), true };
            ggml_context *   ctx = ggml_init(ip);
            ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 65536, true);
            SbInputs         in  = mk_batched(ctx);
            ggml_tensor *    h   = sb_proj_in(ctx, &M, in, T, B);
            ggml_tensor *    e   = sb_cond(ctx, &M, in);
            for (int li = lo; li < hi; li++) {
                h = sb_layer(ctx, &M, li, h, e, in, &lora, S, enc_S, B, true);
            }
            ggml_tensor * ls = sb_loss(ctx, sb_head(ctx, &M, h, in, T, B), in.t_vtgt, in.t_lw);
            ggml_set_loss(ls);
            ggml_build_forward_expand(gf, ls);
            n_mono = ggml_graph_n_nodes(gf);
            std::vector<ggml_tensor *> ga;
            lm_optim_fill_gacc(&opt, gf, &ga);
            ggml_build_backward_expand(ctx, gf, ga.data());
            ggml_backend_sched_reset(M.sched);
            ok = ggml_backend_sched_graph_compute(M.sched, gf) == GGML_STATUS_SUCCESS;
            if (ok) {
                float lv = 0.0f;
                ggml_backend_tensor_get(ls, &lv, 0, sizeof(float));
                loss_mono = (double) lv;
            }
            ggml_free(ctx);
        }
        if (ok) {
            read_accs(&gmono);
        }

        // segmented
        lm_optim_zero_grad(&opt);
        ggml_backend_tensor_memset(t_bndg, 0, 0, ggml_nbytes(t_bndg));
        if (ok) {  // phase 1: no-grad forward, boundary copied out
            upload_all(false);
            ggml_init_params ip  = { arena.size(), arena.data(), true };
            ggml_context *   ctx = ggml_init(ip);
            ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 65536, false);
            SbInputs         in  = mk_batched(ctx);
            ggml_tensor *    h   = sb_proj_in(ctx, &M, in, T, B);
            ggml_tensor *    e   = sb_cond(ctx, &M, in);
            h                    = sb_layer(ctx, &M, lo, h, e, in, &lora, S, enc_S, B, true);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, h, t_bnd));
            n_p1 = ggml_graph_n_nodes(gf);
            ggml_backend_sched_reset(M.sched);
            ok = ggml_backend_sched_graph_compute(M.sched, gf) == GGML_STATUS_SUCCESS;
            ggml_free(ctx);
        }
        if (ok) {  // phase 2: LAST segment, real head + loss, boundary grad out
            upload_all(false);
            ggml_set_param(t_bnd);
            ggml_init_params ip  = { arena.size(), arena.data(), true };
            ggml_context *   ctx = ggml_init(ip);
            ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 65536, true);
            SbInputs         in  = mk_batched(ctx);
            ggml_tensor *    e   = sb_cond(ctx, &M, in);
            ggml_tensor *    h   = sb_layer(ctx, &M, hi - 1, t_bnd, e, in, &lora, S, enc_S, B, true);
            ggml_tensor *    ls  = sb_loss(ctx, sb_head(ctx, &M, h, in, T, B), in.t_vtgt, in.t_lw);
            ggml_set_loss(ls);
            ggml_build_forward_expand(gf, ls);
            n_p2 = ggml_graph_n_nodes(gf);
            std::vector<ggml_tensor *> ga;
            sb_fill_gacc(&opt, gf, t_bnd, t_bndg, &ga);
            ggml_build_backward_expand(ctx, gf, ga.data());
            ggml_backend_sched_reset(M.sched);
            ok = ggml_backend_sched_graph_compute(M.sched, gf) == GGML_STATUS_SUCCESS;
            if (ok) {
                float lv = 0.0f;
                ggml_backend_tensor_get(ls, &lv, 0, sizeof(float));
                loss_seg = (double) lv;
                std::vector<float> g((size_t) ggml_nelements(t_bndg));
                ggml_backend_tensor_get(t_bndg, g.data(), 0, g.size() * sizeof(float));
                for (size_t k = 0; k < g.size(); k++) {
                    bndg_mag = std::max(bndg_mag, fabs((double) g[k]));
                }
            }
            ggml_free(ctx);
            t_bnd->flags &= ~(int32_t) GGML_TENSOR_FLAG_PARAM;
        }
        if (ok) {  // phase 3: FIRST segment, spike-s3 surrogate loss
            upload_all(false);
            ggml_init_params ip  = { arena.size(), arena.data(), true };
            ggml_context *   ctx = ggml_init(ip);
            ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 65536, true);
            SbInputs         in  = mk_batched(ctx);
            ggml_tensor *    h   = sb_proj_in(ctx, &M, in, T, B);
            ggml_tensor *    e   = sb_cond(ctx, &M, in);
            h                    = sb_layer(ctx, &M, lo, h, e, in, &lora, S, enc_S, B, true);
            ggml_tensor * surr   = ggml_sum(ctx, ggml_mul(ctx, h, t_bndg));  // dL/dh == t_bndg exactly
            ggml_set_name(surr, "sb_surrogate");
            ggml_set_output(surr);
            ggml_set_loss(surr);
            ggml_build_forward_expand(gf, surr);
            n_p3 = ggml_graph_n_nodes(gf);
            std::vector<ggml_tensor *> ga;
            lm_optim_fill_gacc(&opt, gf, &ga);
            ggml_build_backward_expand(ctx, gf, ga.data());
            ggml_backend_sched_reset(M.sched);
            ok = ggml_backend_sched_graph_compute(M.sched, gf) == GGML_STATUS_SUCCESS;
            ggml_free(ctx);
        }
        if (ok) {
            read_accs(&gseg);
        }

        std::string  worst = "-";
        const double rel   = ok ? cmp_accs(gseg, gmono, &worst) : 1e30;
        char         d[640];
        snprintf(d, sizeof(d),
                 "[%s] 2 layers (%d-%d) / 2 segments at B=%d: monolithic %d-node fwd+bwd (loss %.9f) vs segmented "
                 "%d(no-grad fwd)+%d(seg2 real loss, loss %.9f)+%d(seg1 surrogate) nodes; boundary grad extracted via "
                 "ggml_set_param, max|dL/dboundary|=%.4e; max relative grad delta %.3e on %s (bar 1e-6)",
                 ggml_backend_name(M.backend), lo, hi - 1, B, n_mono, loss_mono, n_p1, n_p2, loss_seg, n_p3,
                 bndg_mag, rel, worst.c_str());
        sb_report(rs, "S-C1", ok && bndg_mag > 0.0 && rel <= 1e-6, d);
    }

    // ── verdict ──────────────────────────────────────────────────────────
    lm_optim_free(&opt);
    lora.free();
    ggml_backend_buffer_free(buf_static);
    ggml_free(ctxs);
    ggml_backend_buffer_free(buf_pf32);
    ggml_free(ctx_pf32);
    dit_train_free(&M);

    int failed = 0;
    fprintf(stderr, "\n[spike-batch] ─── summary ───\n");
    for (size_t i = 0; i < rs.size(); i++) {
        fprintf(stderr, "[spike-batch] %-5s %s\n", rs[i].name.c_str(), rs[i].pass ? "PASS" : "FAIL");
        failed += rs[i].pass ? 0 : 1;
    }
    fprintf(stderr, "[spike-batch] %d/%d gates passed\n", (int) rs.size() - failed, (int) rs.size());
    return failed == 0 ? 0 : 1;
}
