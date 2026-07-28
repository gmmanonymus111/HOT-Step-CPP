#pragma once
// dit-train-graph.h — the trainable DiT forward and the §4.5 product loss.
//
// Productionised from spike-dit2-graph.h (d2_layer / d2_build_forward), which
// the adversarial verifier audited op-for-op against dit-graph.h. The spike
// headers are reference material only and are NOT included here.
//
// Properties reproduced exactly, each load-bearing (plan §3.4):
//  1. NO flash attention — GGML_OP_FLASH_ATTN_EXT has no backward. dit_attn_f32
//     (soft_max_ext) is exact: its backward uses the softmax OUTPUT, so mask and
//     scale are baked in and masked positions get exactly zero gradient. The
//     retained [S,S,Nh] softmax output is the S^2 term in the VRAM model.
//  2. REAL masks, both. sa_mask only when layer_type == 0 (sliding window);
//     layer_type == 1 is full attention (dit.h:656). ca_mask always. Spike round 1
//     passed nullptr and half the layers silently ran full attention.
//  3. ggml_swiglu_split — the fused ggml_swiglu has no backward.
//  4. Cross-attn Q/K norm BEFORE the permute is the same op as the inference
//     graph's norm-after-permute: rms_norm acts on ne0 == D in both layouts.
//  5. AdaLN: adaln = reshape_1d(scale_shift_table, 6H) + tproj, six [H] views in
//     order shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff. Output head
//     adds temb to BOTH out_shift and out_scale (dit-graph.h:883-884).
//  6. Cross-attention has NO gate — plain residual add.
//
// docs/plans/2026-07-28-dit-trainer-implementation.md §3.4 / §3.5

#include "dit-graph.h"
#include "dit.h"
#include "train/dit-adapter.h"
#include "train/dit-mirror.h"

#include <string>
#include <utility>
#include <vector>

// ─── graph inputs (every one re-uploaded EVERY micro-step, §3.0) ────────────

struct DitInputs {
    ggml_tensor * t_input = nullptr;  // [in_ch, T]     concat(context, xt) per frame
    ggml_tensor * t_enc   = nullptr;  // [enc_H, enc_S]
    ggml_tensor * t_pos   = nullptr;  // [S] i32, 0..S-1 FOR THE CROP
    ggml_tensor * t_temb  = nullptr;  // [H]   precomputed
    ggml_tensor * t_tproj = nullptr;  // [6H]  precomputed
    ggml_tensor * t_sa    = nullptr;  // [S, S] f16 sliding-window mask
    ggml_tensor * t_ca    = nullptr;  // [enc_S, S] f16 encoder-padding mask
    ggml_tensor * t_vtgt  = nullptr;  // [Oc, T] velocity target
    ggml_tensor * t_cw    = nullptr;  // [Oc] channel weights (mean 1)
};

// Named probes for the §6.2 V1 forward diff. Populated only when non-null, so
// training pays nothing (ggml_set_output pins a tensor for the whole graph).
struct DitTaps {
    std::vector<std::pair<std::string, ggml_tensor *>> v;
};

static void dit_tap(DitTaps * taps, const char * name, ggml_tensor * t) {
    if (!taps) {
        return;
    }
    ggml_set_name(t, name);
    ggml_set_output(t);
    // A reshape/permute is a VIEW. ggml_gallocr only refuses to free a node that
    // carries GGML_TENSOR_FLAG_OUTPUT itself — marking the view does NOT protect
    // its parent's storage, so the parent gets recycled and the probe reads back
    // whatever landed there next. Mark the whole view chain.
    for (ggml_tensor * p = t->view_src; p; p = p->view_src) {
        ggml_set_output(p);
    }
    taps->v.push_back({ std::string(name), t });
}

// Same protection for a graph we did not build (the inference graph's own debug
// outputs are views for exactly the same reason).
static void dit_protect_output_views(ggml_cgraph * gf) {
    const int n = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n; i++) {
        ggml_tensor * nd = ggml_graph_node(gf, i);
        if (!(nd->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            continue;
        }
        for (ggml_tensor * p = nd->view_src; p; p = p->view_src) {
            ggml_set_output(p);
        }
    }
}

// ─── one decoder layer (N == 1, unfused, manual attention) ──────────────────

static ggml_tensor * dit_train_layer(ggml_context * ctx, DitTrainModel * M, int li, ggml_tensor * hidden,
                                     ggml_tensor * enc, const DitInputs & in, const DitAdapter * ad, int S, int enc_S,
                                     DitTaps * taps) {
    const DiTGGMLConfig & c  = M->m.cfg;
    DiTGGMLLayer *        ly = &M->m.layers[li];
    const int             H = c.hidden_size, D = c.head_dim, Nh = c.n_heads, Nkv = c.n_kv_heads;
    const float           scale = 1.0f / sqrtf((float) D);
    const bool            tap0  = (taps != nullptr && li == 0);

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
    nsa               = dit_ggml_adaln(ctx, nsa, scale_sa, shift_sa, M->m.scalar_one);
    if (tap0) {
        dit_tap(taps, "layer0_sa_input", nsa);
    }

    ggml_tensor * q = ad->apply(ctx, ly->sa_q_proj, li, DIT_SA_Q, nsa);
    ggml_tensor * k = ad->apply(ctx, ly->sa_k_proj, li, DIT_SA_K, nsa);
    ggml_tensor * v = ad->apply(ctx, ly->sa_v_proj, li, DIT_SA_V, nsa);

    q = ggml_reshape_3d(ctx, q, D, Nh, S);
    k = ggml_reshape_3d(ctx, k, D, Nkv, S);
    v = ggml_reshape_3d(ctx, v, D, Nkv, S);
    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, c.rms_norm_eps), dit_ggml_f32(ctx, ly->sa_q_norm));
    k = ggml_mul(ctx, ggml_rms_norm(ctx, k, c.rms_norm_eps), dit_ggml_f32(ctx, ly->sa_k_norm));
    q = ggml_rope_ext(ctx, q, in.t_pos, NULL, D, 2 /*NEOX*/, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, in.t_pos, NULL, D, 2, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    if (tap0) {
        dit_tap(taps, "layer0_q_after_rope", q);
        dit_tap(taps, "layer0_k_after_rope", k);
    }
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    // layer_type 0 = sliding window, 1 = full attention (dit.h:656)
    ggml_tensor * sa_mask = (ly->layer_type == 0) ? in.t_sa : nullptr;
    ggml_tensor * attn    = dit_attn_f32(ctx, q, k, v, sa_mask, scale);
    attn                  = ggml_reshape_2d(ctx, attn, Nh * D, S);
    if (tap0) {
        dit_tap(taps, "layer0_attn_out", attn);
    }
    ggml_tensor * sa_out = ad->apply(ctx, ly->sa_o_proj, li, DIT_SA_O, attn);
    if (tap0) {
        dit_tap(taps, "layer0_sa_output", sa_out);
    }
    hidden = dit_ggml_gated_add(ctx, res, sa_out, gate_sa);
    if (tap0) {
        dit_tap(taps, "layer0_after_self_attn", hidden);
    }

    // ── cross-attention (no gate, plain residual) ──
    ggml_tensor * nca = dit_ggml_rms_norm_weighted(ctx, hidden, ly->cross_attn_norm, c.rms_norm_eps);
    ggml_tensor * cq  = ad->apply(ctx, ly->ca_q_proj, li, DIT_CA_Q, nca);
    ggml_tensor * ck  = ad->apply(ctx, ly->ca_k_proj, li, DIT_CA_K, enc);
    ggml_tensor * cv  = ad->apply(ctx, ly->ca_v_proj, li, DIT_CA_V, enc);
    // QK-norm BEFORE the permute. The inference graph (dit-graph.h:534-547) and
    // the spike both normalise AFTER it; rms_norm acts on ne0 == D, which stays
    // contiguous under permute(0,2,1,3), so the FORWARD is bit-identical either
    // way (gate T3 confirms it). The BACKWARD is not: ggml's rms_norm_back on a
    // non-contiguous src produced a systematically wrong gradient — measured by
    // gate T4 as fd/analytic ~= 0.16-0.24 on cross_attn.k_proj and ~0.9 on
    // cross_attn.q_proj, while every site without a QK-norm (v/o) was clean to
    // 1e-4 and self-attention (which normalises pre-permute already) was clean.
    // Normalising pre-permute hands rms_norm a contiguous tensor and the
    // discrepancy disappears. Self-attention above uses the same ordering.
    cq                = ggml_reshape_3d(ctx, cq, D, Nh, S);
    ck                = ggml_reshape_3d(ctx, ck, D, Nkv, enc_S);
    cq                = ggml_mul(ctx, ggml_rms_norm(ctx, cq, c.rms_norm_eps), dit_ggml_f32(ctx, ly->ca_q_norm));
    ck                = ggml_mul(ctx, ggml_rms_norm(ctx, ck, c.rms_norm_eps), dit_ggml_f32(ctx, ly->ca_k_norm));
    cq                = ggml_permute(ctx, cq, 0, 2, 1, 3);
    ck                = ggml_permute(ctx, ck, 0, 2, 1, 3);
    cv                = ggml_permute(ctx, ggml_reshape_3d(ctx, cv, D, Nkv, enc_S), 0, 2, 1, 3);
    ggml_tensor * cattn = dit_attn_f32(ctx, cq, ck, cv, in.t_ca, scale);
    cattn               = ggml_reshape_2d(ctx, cattn, Nh * D, S);
    hidden              = ggml_add(ctx, hidden, ad->apply(ctx, ly->ca_o_proj, li, DIT_CA_O, cattn));
    if (tap0) {
        dit_tap(taps, "layer0_after_cross_attn", hidden);
    }

    // ── MLP ──
    res               = hidden;
    ggml_tensor * nff = dit_ggml_rms_norm_weighted(ctx, hidden, ly->mlp_norm, c.rms_norm_eps);
    nff               = dit_ggml_adaln(ctx, nff, scale_ff, shift_ff, M->m.scalar_one);
    ggml_tensor * g   = ad->apply(ctx, ly->gate_proj, li, DIT_MLP_GATE, nff);
    ggml_tensor * u   = ad->apply(ctx, ly->up_proj, li, DIT_MLP_UP, nff);
    ggml_tensor * ff  = ggml_swiglu_split(ctx, g, u);  // fused ggml_swiglu has NO backward
    ggml_tensor * fo  = ad->apply(ctx, ly->down_proj, li, DIT_MLP_DOWN, ff);
    return dit_ggml_gated_add(ctx, res, fo, gate_ff);
}

// ─── full trainable forward -> velocity [Oc, T] ─────────────────────────────
//
// `layer_first`/`layer_last` run a SUB-STACK; the finite-difference gate (T4)
// uses [n_layers-1, n_layers) so the graph stays small. -1/-1 = the whole stack.
static ggml_tensor * dit_train_forward(ggml_context * ctx, DitTrainModel * M, const DitAdapter * ad,
                                       const DitInputs & in, int T, int enc_S, DitTaps * taps = nullptr,
                                       int layer_first = -1, int layer_last = -1) {
    const DiTGGMLConfig & c = M->m.cfg;
    const int             S = T / c.patch_size, H = c.hidden_size;
    const int             lo = (layer_first < 0) ? 0 : layer_first;
    const int             hi = (layer_last < 0) ? c.n_layers : layer_last;

    ggml_tensor * patched = ggml_reshape_2d(ctx, in.t_input, (int64_t) c.in_channels * c.patch_size, S);
    ggml_tensor * hidden  = ggml_add(ctx, ggml_mul_mat(ctx, M->m.proj_in_w, patched), M->m.proj_in_b);
    dit_tap(taps, "hidden_after_proj_in", hidden);
    ggml_tensor * enc = ggml_add(ctx, ggml_mul_mat(ctx, M->m.cond_emb_w, in.t_enc), M->m.cond_emb_b);
    dit_tap(taps, "enc_after_cond_emb", enc);

    for (int i = lo; i < hi; i++) {
        hidden = dit_train_layer(ctx, M, i, hidden, enc, in, ad, S, enc_S, taps);
        if (taps && (i == 0 || i == 6 || i == 12 || i == 18 || i == c.n_layers - 1)) {
            char nm[64];
            snprintf(nm, sizeof(nm), "hidden_after_layer%d", i);
            dit_tap(taps, nm, hidden);
        }
    }

    ggml_tensor * oss_flat  = ggml_reshape_1d(ctx, M->m.out_scale_shift, 2 * H);
    ggml_tensor * out_shift = ggml_add(ctx, ggml_view_1d(ctx, oss_flat, H, 0), in.t_temb);
    ggml_tensor * out_scale = ggml_add(ctx, ggml_view_1d(ctx, oss_flat, H, (size_t) H * sizeof(float)), in.t_temb);
    ggml_tensor * nout      = dit_ggml_rms_norm_weighted(ctx, hidden, M->m.norm_out, c.rms_norm_eps);
    nout                    = dit_ggml_adaln(ctx, nout, out_scale, out_shift, M->m.scalar_one);
    ggml_tensor * out       = ggml_add(ctx, ggml_mul_mat(ctx, M->m.proj_out_w, nout), M->m.proj_out_b);
    ggml_tensor * vel       = ggml_reshape_2d(ctx, out, c.out_channels, T);  // [Oc, T]
    dit_tap(taps, "velocity", vel);
    return vel;
}

// ─── the §4.5 product loss (D8) ─────────────────────────────────────────────
//
//   err[c,i] = v_pred - v_tgt
//   per_elem = err^2
//   if channel_balance: per_elem[c,i] *= cw[c]     (cw = (1/std)/mean(1/std))
//   raw_loss = sum(per_elem) / (Oc * len)
//
// The flow_snr timestep weighting is NOT in this graph: with batch 1 it would be
// w/w == 1. It arrives per micro-step as t_lossgrad = (w_i/wbar)/window (§3.5),
// which is exact and needs zero change to lm-optim.h.
static ggml_tensor * dit_train_loss(ggml_context * ctx, ggml_tensor * vpred, const DitInputs & in, int Oc, int len,
                                    bool channel_balance) {
    ggml_tensor * diff = ggml_sub(ctx, vpred, in.t_vtgt);  // [Oc, T]
    ggml_tensor * pe   = ggml_sqr(ctx, diff);
    if (channel_balance && in.t_cw) {
        pe = ggml_mul(ctx, pe, in.t_cw);  // [Oc,T] * [Oc] row-broadcast
    }
    ggml_tensor * loss = ggml_scale(ctx, ggml_sum(ctx, pe), 1.0f / (float) ((int64_t) Oc * (int64_t) len));
    ggml_set_name(loss, "flow_loss");
    ggml_set_output(loss);
    return loss;
}

// ─── temb / tproj precompute (substitution S4, equivalence class EXACT) ─────
//
// ggml_timestep_embedding has no backward and nothing upstream of temb is
// trainable, so temb [H] / tproj [6H] are computed in a separate tiny graph on
// the TRAINING backend from the MIRRORED time_embed weights and uploaded as
// constants. r == t (D12), so time_embed_r sees t - t_r == 0.
//
// Called once per optimizer window (grad_accum timesteps at a time) because
// §3.5's loss weighting already needs the whole window's t values up front.
static bool dit_train_temb(DitTrainModel * M, const std::vector<float> & ts, std::vector<std::vector<float>> * temb_out,
                           std::vector<std::vector<float>> * tproj_out) {
    const int H = M->m.cfg.hidden_size;

    ggml_context * ctxs;
    {
        ggml_init_params p = { 8 * ggml_tensor_overhead(), nullptr, true };
        ctxs               = ggml_init(p);
    }
    if (!ctxs) {
        return false;
    }
    ggml_tensor * t_t = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 1);
    ggml_tensor * t_r = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 1);
    ggml_set_input(t_t);
    ggml_set_input(t_r);
    ggml_backend_buffer_t bufs = ggml_backend_alloc_ctx_tensors(ctxs, M->backend);
    if (!bufs) {
        ggml_free(ctxs);
        return false;
    }

    std::vector<uint8_t> arena((size_t) 16 << 20);
    bool                 ok = true;
    temb_out->assign(ts.size(), {});
    tproj_out->assign(ts.size(), {});

    for (size_t i = 0; i < ts.size() && ok; i++) {
        ggml_backend_tensor_set(t_t, &ts[i], 0, sizeof(float));
        ggml_backend_tensor_set(t_r, &ts[i], 0, sizeof(float));  // r == t, so t - t_r == 0

        ggml_init_params ip  = { arena.size(), arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 2048, false);

        ggml_tensor *tproj_t = nullptr, *tproj_r = nullptr;
        ggml_tensor * temb_t =
            dit_ggml_build_temb(ctx, &M->m, &M->m.time_embed, t_t, &tproj_t, nullptr, nullptr, nullptr, "_t");
        ggml_tensor * tdiff = ggml_sub(ctx, t_t, t_r);
        ggml_tensor * temb_r =
            dit_ggml_build_temb(ctx, &M->m, &M->m.time_embed_r, tdiff, &tproj_r, nullptr, nullptr, nullptr, "_r");
        ggml_tensor * temb  = ggml_add(ctx, temb_t, temb_r);
        ggml_tensor * tproj = ggml_add(ctx, tproj_t, tproj_r);
        ggml_set_output(temb);
        ggml_set_output(tproj);
        ggml_build_forward_expand(gf, temb);
        ggml_build_forward_expand(gf, tproj);

        ggml_backend_sched_reset(M->sched);
        ok = ggml_backend_sched_graph_compute(M->sched, gf) == GGML_STATUS_SUCCESS;
        if (ok) {
            (*temb_out)[i].resize((size_t) H);
            (*tproj_out)[i].resize((size_t) 6 * H);
            ggml_backend_tensor_get(temb, (*temb_out)[i].data(), 0, (size_t) H * sizeof(float));
            ggml_backend_tensor_get(tproj, (*tproj_out)[i].data(), 0, (size_t) 6 * H * sizeof(float));
        }
        ggml_free(ctx);
    }

    ggml_backend_buffer_free(bufs);
    ggml_free(ctxs);
    return ok;
}
