#pragma once
// lm-ckpt.h — per-layer gradient checkpointing + chunked cross-entropy: the
// low-VRAM LM training path that makes 4B fit in ~12 GB.
//
// docs/plans/2026-07-28-lm-4b-training.md §3.3 - §3.7
//
// Why this exists (D1/D2). ggml_build_backward_expand computes the ACTIVATION
// gradient of mul_mat(W, x) as ggml_out_prod(W, transpose(grad)); out_prod is
// F32-only on CUDA and GGML_ABORTs for BF16 on CPU. So every frozen projection
// on the gradient path must be F32 *while its graph runs*. In a whole-model
// graph all L layers' F32 copies are live at once (16 GB at 4B); with one
// checkpoint segment per layer exactly ONE is (385 MiB). That single fact is
// what unlocks 4B.
//
// Shape of a micro-step (§3.5):
//
//   P1  embedding      -> C[lo]                              grads = false
//   P2  collect        C[l] -> C[l+1], l = lo .. hi-2        grads = false
//   P3  tail forward   C[hi-1] -> t_H (layer hi-1 + final norm)
//   P4  clear t_G
//   P5  chunked CE head: per-chunk logits/loss/dL-dh into t_G, no autodiff
//   P7  backward segments l = hi-1 .. lo: recompute the layer from C[l], build
//       the surrogate loss SUM(Y (.) dY), backward it, accumulate LoRA grads
//       into opt.acc[] and dL/dC[l] into the other Gh ping-pong buffer.
//
// Why the surrogate is exact: ggml_sum's backward is repeat(grad); with
// grad = t_one = 1.0 that is an all-ones [H,S], and ggml_mul's backward w.r.t.
// Y is mul(ones, dY) = dY — exact in fp32. So the segment sees precisely the
// upstream gradient and the parameter gradients equal a whole-graph backward's
// by the chain rule.
//
// D13 — the recomputed forward MUST use the same LmLayerOpts as the collect
// pass. Skipping the F32 cast in the collect pass is faster and silently turns
// checkpointing from an identity into a ~1e-3 approximation.
//
// lm-optim.h is NOT modified (D8): lm_ckpt_fill_gacc() below reads only the
// public LmOptim::param_slot / LmOptim::acc members.

#include "train/lm-common.h"
#include "train/lm-data.h"
#include "train/lm-graph.h"
#include "train/lm-optim.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

// ─── configuration + persistent state ───────────────────────────────────────

struct LmCkptCfg {
    int chunk           = 128;  // trained positions per CE chunk
    int attn_head_block = 0;    // q heads per attention block (0 = off)
    int s_max           = 0;    // longest accepted sequence
    int layer_lo        = 0;
    int layer_hi        = 0;    // exclusive; <= 0 means cfg.n_layers
};

// Each group gets its OWN backend buffer because ggml_backend_buffer_clear()
// is whole-buffer.
struct LmCkptState {
    Qwen3LM * lm = nullptr;
    LmCkptCfg cfg;

    ggml_context *        ctx_ckpt = nullptr;  // C[0 .. L-1]                     never cleared
    ggml_backend_buffer_t buf_ckpt = nullptr;
    ggml_context *        ctx_gh0  = nullptr;  // Gh[0] == t_G                    cleared per micro-step
    ggml_backend_buffer_t buf_gh0  = nullptr;
    ggml_context *        ctx_gh1  = nullptr;  // Gh[1]                           cleared per segment
    ggml_backend_buffer_t buf_gh1  = nullptr;
    ggml_context *        ctx_misc = nullptr;  // t_H + t_zero_attn               cleared once
    ggml_backend_buffer_t buf_misc = nullptr;
    ggml_context *        ctx_embt = nullptr;  // t_embT                          written once
    ggml_backend_buffer_t buf_embt = nullptr;
    ggml_context *        ctx_labc = nullptr;  // t_labc                          sparse set/clear
    ggml_backend_buffer_t buf_labc = nullptr;

    std::vector<ggml_tensor *> C;
    ggml_tensor *              Gh[2]       = { nullptr, nullptr };
    ggml_tensor *              t_H         = nullptr;
    ggml_tensor *              t_zero_attn = nullptr;
    ggml_tensor *              t_embT      = nullptr;
    ggml_tensor *              t_labc      = nullptr;

    std::vector<uint8_t>       arena;  // one reused 32 MiB graph arena
    std::vector<ggml_tensor *> gacc;
    std::vector<int32_t>       pos_scratch;
    int                        last_mask_S = 0;

    size_t fixed_bytes() const {
        size_t b = 0;
        const ggml_backend_buffer_t bufs[6] = { buf_ckpt, buf_gh0, buf_gh1, buf_misc, buf_embt, buf_labc };
        for (int i = 0; i < 6; i++) {
            if (bufs[i]) {
                b += ggml_backend_buffer_get_size(bufs[i]);
            }
        }
        return b;
    }
};

static void lm_ckpt_free(LmCkptState * st) {
    ggml_backend_buffer_t bufs[6] = { st->buf_ckpt, st->buf_gh0, st->buf_gh1, st->buf_misc, st->buf_embt, st->buf_labc };
    for (int i = 0; i < 6; i++) {
        if (bufs[i]) {
            ggml_backend_buffer_free(bufs[i]);
        }
    }
    ggml_context * ctxs[6] = { st->ctx_ckpt, st->ctx_gh0, st->ctx_gh1, st->ctx_misc, st->ctx_embt, st->ctx_labc };
    for (int i = 0; i < 6; i++) {
        if (ctxs[i]) {
            ggml_free(ctxs[i]);
        }
    }
    *st = LmCkptState{};
}

// The low-VRAM path never mirrors, so the "is this base trainable?" check that
// lm_build_f32_mirror() performs on the way past has to be done explicitly.
// Same wording as the mirror's, so the server sees the same fatal message.
static bool lm_ckpt_check_base(Qwen3LM * lm, std::string * err) {
    const int L = lm->cfg.n_layers;
    for (int i = 0; i < L; i++) {
        const Qwen3Layer & ly = lm->layers[i];
        if (!ly.q_proj || !ly.k_proj || !ly.v_proj || !ly.o_proj || !ly.gate_proj || !ly.up_proj || !ly.down_proj) {
            char b[160];
            snprintf(b, sizeof(b), "layer %d has fused projections — the trainer requires an unfused load", i);
            *err = b;
            return false;
        }
    }
    std::vector<ggml_tensor *> all;
    all.push_back(lm->embed_tokens);
    all.push_back(lm->final_norm);
    for (int i = 0; i < L; i++) {
        Qwen3Layer & ly = lm->layers[i];
        ggml_tensor * t[11] = { ly.input_layernorm, ly.post_attn_layernorm, ly.q_proj,  ly.k_proj,
                                ly.v_proj,          ly.o_proj,              ly.q_norm,  ly.k_norm,
                                ly.gate_proj,       ly.up_proj,             ly.down_proj };
        for (int j = 0; j < 11; j++) {
            all.push_back(t[j]);
        }
    }
    for (size_t i = 0; i < all.size(); i++) {
        ggml_tensor * s = all[i];
        if (!s) {
            continue;
        }
        if (s->type != GGML_TYPE_F32 && s->type != GGML_TYPE_BF16 && s->type != GGML_TYPE_F16) {
            char b[256];
            snprintf(b, sizeof(b),
                     "base weight '%s' is %s — LM training needs a BF16/F16/F32 base "
                     "(quantized bases cannot be trained: ggml_out_prod is F32-only)",
                     s->name, ggml_type_name(s->type));
            *err = b;
            return false;
        }
    }
    return true;
}

// Allocate after sample building, when S_max / V are known.
static bool lm_ckpt_alloc(LmCkptState * st, Qwen3LM * lm, const LmCkptCfg & cfg, std::string * err) {
    const Qwen3LMConfig & c = lm->cfg;
    const int             L = c.n_layers, H = c.hidden_size, V = c.vocab_size;
    const int             D = c.head_dim, Nh = c.n_heads;

    st->lm  = lm;
    st->cfg = cfg;
    if (st->cfg.layer_hi <= 0) {
        st->cfg.layer_hi = L;
    }
    const int S = st->cfg.s_max;
    if (S <= 0 || st->cfg.chunk <= 0) {
        *err = "invalid checkpoint configuration (s_max / chunk)";
        return false;
    }
    if (!lm_ckpt_head_block_ok(c, st->cfg.attn_head_block)) {
        char b[160];
        snprintf(b, sizeof(b), "--attn-head-block %d does not divide n_heads %d with n_kv_heads %d",
                 st->cfg.attn_head_block, c.n_heads, c.n_kv_heads);
        *err = b;
        return false;
    }

    auto mkctx = [](int n) {
        ggml_init_params p = { (size_t) (n + 8) * ggml_tensor_overhead(), nullptr, true };
        return ggml_init(p);
    };

    st->ctx_ckpt = mkctx(L);
    st->ctx_gh0  = mkctx(1);
    st->ctx_gh1  = mkctx(1);
    st->ctx_misc = mkctx(2);
    st->ctx_embt = mkctx(1);
    st->ctx_labc = mkctx(1);
    if (!st->ctx_ckpt || !st->ctx_gh0 || !st->ctx_gh1 || !st->ctx_misc || !st->ctx_embt || !st->ctx_labc) {
        *err = "cannot create the checkpoint contexts";
        return false;
    }

    st->C.assign((size_t) L, nullptr);
    for (int l = 0; l < L; l++) {
        st->C[(size_t) l] = ggml_new_tensor_2d(st->ctx_ckpt, GGML_TYPE_F32, H, S);
        char nm[32];
        snprintf(nm, sizeof(nm), "ckpt.%d", l);
        ggml_set_name(st->C[(size_t) l], nm);
        // Legal on a pre-allocated leaf: ggml_set_param only asserts op == NONE.
        // Being a PARAM makes it a NODE (not a leaf) of the segment graph, which
        // is what lets lm_ckpt_fill_gacc() hand it a persistent dX buffer.
        ggml_set_param(st->C[(size_t) l]);
        ggml_set_input(st->C[(size_t) l]);
    }

    st->Gh[0] = ggml_new_tensor_2d(st->ctx_gh0, GGML_TYPE_F32, H, S);
    st->Gh[1] = ggml_new_tensor_2d(st->ctx_gh1, GGML_TYPE_F32, H, S);
    ggml_set_name(st->Gh[0], "ckpt.G0");
    ggml_set_name(st->Gh[1], "ckpt.G1");
    ggml_set_input(st->Gh[0]);
    ggml_set_input(st->Gh[1]);

    st->t_H = ggml_new_tensor_2d(st->ctx_misc, GGML_TYPE_F32, H, S);
    ggml_set_name(st->t_H, "ckpt.H");
    ggml_set_input(st->t_H);
    if (st->cfg.attn_head_block > 0) {
        st->t_zero_attn = ggml_new_tensor_2d(st->ctx_misc, GGML_TYPE_F32, (int64_t) Nh * D, S);
        ggml_set_name(st->t_zero_attn, "ckpt.attn_zero");
        ggml_set_input(st->t_zero_attn);
    }

    // Dtype stays the base's own (D4): the chunked head never runs out_prod on
    // it, so no F32 copy is needed — 1,060 MiB instead of 2,120 MiB at 4B.
    st->t_embT = ggml_new_tensor_2d(st->ctx_embt, lm->embed_tokens->type, V, H);
    ggml_set_name(st->t_embT, "ckpt.embT");
    ggml_set_input(st->t_embT);

    st->t_labc = ggml_new_tensor_2d(st->ctx_labc, GGML_TYPE_F32, V, st->cfg.chunk);
    ggml_set_name(st->t_labc, "ckpt.labels");
    ggml_set_input(st->t_labc);

    st->buf_ckpt = ggml_backend_alloc_ctx_tensors(st->ctx_ckpt, lm->backend);
    st->buf_gh0  = ggml_backend_alloc_ctx_tensors(st->ctx_gh0, lm->backend);
    st->buf_gh1  = ggml_backend_alloc_ctx_tensors(st->ctx_gh1, lm->backend);
    st->buf_misc = ggml_backend_alloc_ctx_tensors(st->ctx_misc, lm->backend);
    st->buf_embt = ggml_backend_alloc_ctx_tensors(st->ctx_embt, lm->backend);
    st->buf_labc = ggml_backend_alloc_ctx_tensors(st->ctx_labc, lm->backend);
    if (!st->buf_ckpt || !st->buf_gh0 || !st->buf_gh1 || !st->buf_misc || !st->buf_embt || !st->buf_labc) {
        *err = "checkpoint buffer allocation failed";
        return false;
    }
    ggml_backend_buffer_clear(st->buf_ckpt, 0);
    ggml_backend_buffer_clear(st->buf_gh0, 0);
    ggml_backend_buffer_clear(st->buf_gh1, 0);
    ggml_backend_buffer_clear(st->buf_misc, 0);  // t_zero_attn: once, then NEVER again
    ggml_backend_buffer_clear(st->buf_labc, 0);

    st->arena.resize((size_t) 32 << 20);
    st->pos_scratch.resize((size_t) S);

    fprintf(stderr,
            "[train-lm] low-vram state: %d checkpoints [%d,%d] f32 (%.1f MB) + embT %s (%.1f MB) + labels [%d,%d] "
            "(%.1f MB) = %.1f MB\n",
            L, H, S, (double) L * H * S * 4.0 / 1048576.0, ggml_type_name(st->t_embT->type),
            ggml_nbytes(st->t_embT) / 1048576.0, V, st->cfg.chunk, ggml_nbytes(st->t_labc) / 1048576.0,
            st->fixed_bytes() / 1048576.0);
    return true;
}

// Host-side [H,V] -> [V,H] scatter at ggml_type_size() granularity, dtype
// preserved. PyTorch gets lm_head.weight.T free as a cuBLAS transa flag; ggml's
// mul_mat reduces over ne0 only, so a physical transpose is unavoidable (§6.2).
static bool lm_ckpt_build_embed_t(LmCkptState * st, std::string * err) {
    Qwen3LM * lm = st->lm;
    const int H = lm->cfg.hidden_size, V = lm->cfg.vocab_size;
    if (st->t_embT->type != lm->embed_tokens->type) {
        *err = "embedT dtype mismatch";
        return false;
    }
    const size_t ts = ggml_type_size(lm->embed_tokens->type);
    if (ggml_blck_size(lm->embed_tokens->type) != 1) {
        *err = "embed_tokens is block-quantized — cannot transpose";
        return false;
    }
    std::vector<uint8_t> src(ggml_nbytes(lm->embed_tokens)), dst(ggml_nbytes(st->t_embT));
    ggml_backend_tensor_get(lm->embed_tokens, src.data(), 0, src.size());
    for (int64_t v = 0; v < V; v++) {
        for (int64_t h = 0; h < H; h++) {
            memcpy(&dst[(size_t) (h * (int64_t) V + v) * ts], &src[(size_t) (v * (int64_t) H + h) * ts], ts);
        }
    }
    ggml_backend_tensor_set(st->t_embT, dst.data(), 0, dst.size());
    return true;
}

// ─── the only new autodiff plumbing (D8) ────────────────────────────────────
//
// PARAM-flag node classes in a segment graph, IN PRIORITY ORDER:
//   1. the segment's checkpoint input  -> g_out  (its dL/dX; NOT in param_slot)
//   2. a LoRA A/B tensor               -> o->acc[param_slot[nd]]
//   3. the surrogate loss node         -> t_one  (NOT o->t_lossgrad — D9)
//
// `ckpt_in` carries the PARAM flag AND is absent from param_slot, so the order
// of the first two branches is load-bearing: swapped, lm_optim's GGML_ASSERT
// equivalent fires on every segment.
static void lm_ckpt_fill_gacc(const LmOptim * o, ggml_cgraph * gf, ggml_tensor * ckpt_in, ggml_tensor * g_out,
                              ggml_tensor * t_one, std::vector<ggml_tensor *> * gacc) {
    const int n = ggml_graph_n_nodes(gf);
    gacc->assign((size_t) n, nullptr);
    for (int i = 0; i < n; i++) {
        ggml_tensor * nd = ggml_graph_node(gf, i);
        if (nd == ckpt_in) {
            (*gacc)[(size_t) i] = g_out;
            continue;
        }
        if (nd->flags & GGML_TENSOR_FLAG_PARAM) {
            auto it = o->param_slot.find(nd);
            GGML_ASSERT(it != o->param_slot.end());
            (*gacc)[(size_t) i] = o->acc[(size_t) it->second];
            continue;
        }
        if (nd->flags & GGML_TENSOR_FLAG_LOSS) {
            (*gacc)[(size_t) i] = t_one;
        }
    }
}

// ─── one micro-step ─────────────────────────────────────────────────────────

struct LmCkptRun {
    Qwen3LM *            lm    = nullptr;
    LmOptim *            opt   = nullptr;
    ggml_backend_sched_t sched = nullptr;
    LmCkptState *        st    = nullptr;

    ggml_tensor * t_tok = nullptr;
    ggml_tensor * t_pos = nullptr;
    ggml_tensor * t_msk = nullptr;
    ggml_tensor * t_gs  = nullptr;  // [1] per-chunk upstream scalar
    ggml_tensor * t_one = nullptr;  // [1] == 1.0f, the segment surrogate's loss grad

    int grad_accum = 1;

    // Self-test only (§3.6): the un-chunked CE head. Needs a full [V, s_tr]
    // label buffer, which is exactly why D4 exists — never a production path.
    bool          naive_head    = false;
    ggml_tensor * t_lab_full    = nullptr;
    // Self-test only: cast embed_tokens / embT to F32 in the head so the
    // chunked and naive heads are numerically comparable (T11 isolates
    // chunking from the BF16 GEMM). Production leaves the head in the base's
    // own dtype, exactly as Side-Step does.
    bool          head_f32_embed = false;
};

static inline void lm_ckpt_upload_mask(LmCkptRun & r, int S) {
    if (S == r.st->last_mask_S) {
        return;
    }
    std::vector<float> m;
    lm_causal_mask(S, &m);
    ggml_backend_tensor_set(r.t_msk, m.data(), 0, m.size() * sizeof(float));
    r.st->last_mask_S = S;
}

static inline LmLayerOpts lm_ckpt_layer_opts(const LmCkptState & st) {
    LmLayerOpts o;
    o.cast_weights    = true;
    o.attn_head_block = st.cfg.attn_head_block;
    o.attn_zero       = st.t_zero_attn;
    return o;
}

// P5 (production): per-chunk graphs, no autodiff, disjoint t_G column writes.
static bool lm_ckpt_head_chunked(LmCkptRun & r, const LmSample & s, bool count_loss, double * ce_out) {
    LmCkptState &         st   = *r.st;
    const Qwen3LMConfig & c    = r.lm->cfg;
    const int             H    = c.hidden_size, V = c.vocab_size;
    const int             s_tr = s.s_tr, CH = st.cfg.chunk;
    const int             GA   = std::max(1, r.grad_accum);

    double ce = 0.0;
    for (int i = 0; i < s_tr; i += CH) {
        const int Sc = std::min(CH, s_tr - i);

        // D9: gs = Sc / (s_tr * grad_accum). ggml_cross_entropy_loss_back
        // computes (softmax - onehot) * gs / nrows with nrows == Sc, i.e.
        // (softmax - onehot) / (s_tr * grad_accum) — exactly Side-Step's
        // (loss / (n_tok * grad_accum)).backward(). The trunk surrogate's loss
        // gradient is therefore 1.0, NOT 1/grad_accum.
        const float gs = (float) Sc / ((float) s_tr * (float) GA);
        ggml_backend_tensor_set(r.t_gs, &gs, 0, sizeof(float));

        LmLabelGuard guard(st.t_labc, s.targets.data() + i, Sc, V);

        ggml_init_params ip  = { st.arena.size(), st.arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 1024, /*grads=*/false);

        const size_t  col = (size_t) (s.n_masked - 1 + i);  // [P] L6a offset
        ggml_tensor * hd  = ggml_cont(ctx, ggml_view_2d(ctx, st.t_H, H, Sc, st.t_H->nb[1], col * st.t_H->nb[1]));
        ggml_tensor * emb = r.head_f32_embed ? qwen3_f32(ctx, r.lm->embed_tokens) : r.lm->embed_tokens;
        ggml_tensor * ebt = r.head_f32_embed ? qwen3_f32(ctx, st.t_embT) : st.t_embT;
        ggml_tensor * lg  = ggml_mul_mat(ctx, emb, hd);  // [V, Sc]
        ggml_tensor * lb  = ggml_view_2d(ctx, st.t_labc, V, Sc, st.t_labc->nb[1], 0);
        ggml_tensor * lc  = ggml_cross_entropy_loss(ctx, lg, lb);  // scalar, mean over Sc rows
        ggml_set_output(lc);
        ggml_tensor * dl = ggml_cross_entropy_loss_back(ctx, r.t_gs, lg, lb);  // (grad, logits, labels)
        ggml_tensor * dh = ggml_mul_mat(ctx, ebt, dl);                         // [H, Sc]
        ggml_tensor * gv = ggml_view_2d(ctx, st.Gh[0], H, Sc, st.Gh[0]->nb[1], col * st.Gh[0]->nb[1]);
        ggml_build_forward_expand(gf, lc);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, dh, gv));

        ggml_backend_sched_reset(r.sched);
        const bool ok = ggml_backend_sched_graph_compute(r.sched, gf) == GGML_STATUS_SUCCESS;
        if (ok && count_loss) {
            float lv = 0.0f;
            ggml_backend_tensor_get(lc, &lv, 0, sizeof(float));
            ce += (double) lv * (double) Sc / (double) s_tr;
        }
        ggml_free(ctx);
        if (!ok) {
            return false;
        }
    }
    if (count_loss && ce_out) {
        *ce_out = ce;
    }
    return true;
}

// §3.6, SELF-TEST ONLY: one grads = true graph over the whole trained span, with
// t_H temporarily ggml_set_param'd so its gradient lands in t_G. Isolates
// "checkpointing" from "chunked loss" in the verification ladder.
static bool lm_ckpt_head_naive(LmCkptRun & r, const LmSample & s, bool count_loss, double * ce_out) {
    LmCkptState &         st   = *r.st;
    const Qwen3LMConfig & c    = r.lm->cfg;
    const int             H    = c.hidden_size, V = c.vocab_size;
    const int             s_tr = s.s_tr;
    GGML_ASSERT(r.t_lab_full != nullptr);

    ggml_set_param(st.t_H);

    ggml_init_params ip  = { st.arena.size(), st.arena.data(), true };
    ggml_context *   ctx = ggml_init(ip);
    ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 4096, /*grads=*/true);

    const size_t  col = (size_t) (s.n_masked - 1);
    ggml_tensor * hd  = ggml_cont(ctx, ggml_view_2d(ctx, st.t_H, H, s_tr, st.t_H->nb[1], col * st.t_H->nb[1]));
    // The cast is MANDATORY here, not optional: mul_mat's activation backward is
    // ggml_out_prod(src0, ...), which is F32-only.
    ggml_tensor * logits = ggml_mul_mat(ctx, qwen3_f32(ctx, r.lm->embed_tokens), hd);
    ggml_tensor * labv   = ggml_view_2d(ctx, r.t_lab_full, V, s_tr, r.t_lab_full->nb[1], 0);
    ggml_tensor * loss   = ggml_cross_entropy_loss(ctx, logits, labv);
    ggml_set_loss(loss);
    ggml_set_output(loss);
    ggml_build_forward_expand(gf, loss);

    lm_ckpt_fill_gacc(r.opt, gf, st.t_H, st.Gh[0], r.opt->t_lossgrad, &st.gacc);
    ggml_build_backward_expand(ctx, gf, st.gacc.data());

    bool ok = false;
    {
        LmLabelGuard guard(r.t_lab_full, s.targets.data(), s_tr, V);
        ggml_backend_sched_reset(r.sched);
        ok = ggml_backend_sched_graph_compute(r.sched, gf) == GGML_STATUS_SUCCESS;
        if (ok && count_loss && ce_out) {
            float ce = 0.0f;
            ggml_backend_tensor_get(loss, &ce, 0, sizeof(float));
            *ce_out = (double) ce;
        }
    }
    ggml_free(ctx);
    st.t_H->flags &= ~(int32_t) GGML_TENSOR_FLAG_PARAM;
    return ok;
}

// Build (but do not run) one P7 segment to size the scheduler.
static int lm_ckpt_probe_segment_nodes(LmCkptRun & r, int S) {
    LmCkptState &         st = *r.st;
    const Qwen3LMConfig & c  = r.lm->cfg;
    const int             H  = c.hidden_size;
    const int             l  = st.cfg.layer_hi - 1;

    const LmLayerOpts opts = lm_ckpt_layer_opts(st);
    ggml_init_params  ip   = { st.arena.size(), st.arena.data(), true };
    ggml_context *    ctx  = ggml_init(ip);
    ggml_cgraph *     gf   = ggml_new_graph_custom(ctx, 8192, /*grads=*/true);

    ggml_tensor * pos_v = ggml_view_1d(ctx, r.t_pos, S, 0);
    ggml_tensor * mask  = ggml_view_2d(ctx, r.t_msk, S, S, (size_t) S * sizeof(float), 0);
    ggml_tensor * X     = ggml_view_2d(ctx, st.C[(size_t) l], H, S, st.C[(size_t) l]->nb[1], 0);
    ggml_tensor * Y     = lm_train_layer(ctx, c, &r.lm->layers[l], X, pos_v, mask, S, opts);
    Y                   = lm_rms(ctx, Y, r.lm->final_norm, c.rms_norm_eps);
    ggml_tensor * dY    = ggml_view_2d(ctx, st.Gh[0], H, S, st.Gh[0]->nb[1], 0);
    ggml_tensor * Lsur  = ggml_sum(ctx, ggml_mul(ctx, Y, dY));
    ggml_set_loss(Lsur);
    ggml_set_output(Lsur);
    ggml_build_forward_expand(gf, Lsur);
    lm_ckpt_fill_gacc(r.opt, gf, st.C[(size_t) l], st.Gh[1], r.t_one, &st.gacc);
    ggml_build_backward_expand(ctx, gf, st.gacc.data());
    const int n = ggml_graph_n_nodes(gf);
    ggml_free(ctx);
    return n;
}

static bool lm_ckpt_micro_step(LmCkptRun & r, const LmSample & s, bool count_loss, double * ce_out) {
    Qwen3LM &             lm = *r.lm;
    LmCkptState &         st = *r.st;
    const Qwen3LMConfig & c  = lm.cfg;
    const int             H  = c.hidden_size;
    const int             S  = (int) s.tokens.size();
    const int             Lo = st.cfg.layer_lo, Hi = st.cfg.layer_hi;

    GGML_ASSERT(S <= st.cfg.s_max && Hi > Lo);

    // ── P0: inputs ───────────────────────────────────────────────────────
    lm_ckpt_upload_mask(r, S);
    ggml_backend_tensor_set(r.t_tok, s.tokens.data(), 0, (size_t) S * 4);
    for (int i = 0; i < S; i++) {
        st.pos_scratch[(size_t) i] = i;
    }
    ggml_backend_tensor_set(r.t_pos, st.pos_scratch.data(), 0, (size_t) S * 4);

    const LmLayerOpts opts = lm_ckpt_layer_opts(st);

    auto run_graph = [&](ggml_cgraph * gf) -> bool {
        ggml_backend_sched_reset(r.sched);
        return ggml_backend_sched_graph_compute(r.sched, gf) == GGML_STATUS_SUCCESS;
    };

    // ── P1: embedding -> C[Lo] ───────────────────────────────────────────
    {
        ggml_init_params ip  = { st.arena.size(), st.arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 64, /*grads=*/false);
        ggml_tensor *    tv  = ggml_view_1d(ctx, r.t_tok, S, 0);
        // get_rows on a BF16 source yields F32 (exact upcast) and needs no
        // backward: embed_tokens is frozen and GET_ROWS' row indices are on
        // ggml's "not differentiable" list.
        ggml_tensor * h0 = ggml_get_rows(ctx, lm.embed_tokens, tv);
        ggml_build_forward_expand(gf,
                                  ggml_cpy(ctx, h0, ggml_view_2d(ctx, st.C[(size_t) Lo], H, S,
                                                                 st.C[(size_t) Lo]->nb[1], 0)));
        const bool ok = run_graph(gf);
        ggml_free(ctx);
        if (!ok) {
            return false;
        }
    }

    // ── P2: forward-collect, l = Lo .. Hi-2 ──────────────────────────────
    for (int l = Lo; l < Hi - 1; l++) {
        ggml_init_params ip  = { st.arena.size(), st.arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 2048, /*grads=*/false);
        ggml_tensor *    pv  = ggml_view_1d(ctx, r.t_pos, S, 0);
        ggml_tensor *    mv  = ggml_view_2d(ctx, r.t_msk, S, S, (size_t) S * sizeof(float), 0);
        ggml_tensor *    X   = ggml_view_2d(ctx, st.C[(size_t) l], H, S, st.C[(size_t) l]->nb[1], 0);
        // D13: SAME opts as the P7 recompute. Not an optimisation opportunity.
        ggml_tensor * Y = lm_train_layer(ctx, c, &lm.layers[l], X, pv, mv, S, opts);
        ggml_build_forward_expand(
            gf, ggml_cpy(ctx, Y, ggml_view_2d(ctx, st.C[(size_t) (l + 1)], H, S, st.C[(size_t) (l + 1)]->nb[1], 0)));
        const bool ok = run_graph(gf);
        ggml_free(ctx);
        if (!ok) {
            return false;
        }
    }

    // ── P3: tail forward (layer Hi-1 + final norm) -> t_H ────────────────
    {
        ggml_init_params ip  = { st.arena.size(), st.arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 2048, /*grads=*/false);
        ggml_tensor *    pv  = ggml_view_1d(ctx, r.t_pos, S, 0);
        ggml_tensor *    mv  = ggml_view_2d(ctx, r.t_msk, S, S, (size_t) S * sizeof(float), 0);
        ggml_tensor *    X   = ggml_view_2d(ctx, st.C[(size_t) (Hi - 1)], H, S, st.C[(size_t) (Hi - 1)]->nb[1], 0);
        ggml_tensor *    Y   = lm_train_layer(ctx, c, &lm.layers[Hi - 1], X, pv, mv, S, opts);
        ggml_tensor *    hN  = lm_rms(ctx, Y, lm.final_norm, c.rms_norm_eps);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, hN, ggml_view_2d(ctx, st.t_H, H, S, st.t_H->nb[1], 0)));
        const bool ok = run_graph(gf);
        ggml_free(ctx);
        if (!ok) {
            return false;
        }
    }

    // ── P4: t_G := 0 (masked columns must stay exactly zero) ─────────────
    ggml_backend_buffer_clear(st.buf_gh0, 0);

    // ── P5: CE head ──────────────────────────────────────────────────────
    if (r.naive_head) {
        if (!lm_ckpt_head_naive(r, s, count_loss, ce_out)) {
            return false;
        }
    } else {
        if (!lm_ckpt_head_chunked(r, s, count_loss, ce_out)) {
            return false;
        }
    }

    // ── P6/P7: backward segments, l = Hi-1 .. Lo ─────────────────────────
    int cur = 0;  // Gh[0] == t_G, already seeded by the head
    for (int l = Hi - 1; l >= Lo; l--) {
        const int nxt = 1 - cur;
        // MANDATORY: ggml_acc_or_set accumulates INTO a supplied grad_acc, so
        // the destination must start at zero.
        ggml_backend_buffer_clear(nxt == 0 ? st.buf_gh0 : st.buf_gh1, 0);

        ggml_init_params ip  = { st.arena.size(), st.arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 8192, /*grads=*/true);

        ggml_tensor * pv = ggml_view_1d(ctx, r.t_pos, S, 0);
        ggml_tensor * mv = ggml_view_2d(ctx, r.t_msk, S, S, (size_t) S * sizeof(float), 0);
        ggml_tensor * X  = ggml_view_2d(ctx, st.C[(size_t) l], H, S, st.C[(size_t) l]->nb[1], 0);
        ggml_tensor * Y  = lm_train_layer(ctx, c, &lm.layers[l], X, pv, mv, S, opts);
        if (l == Hi - 1) {
            Y = lm_rms(ctx, Y, lm.final_norm, c.rms_norm_eps);
        }
        ggml_tensor * dY   = ggml_view_2d(ctx, st.Gh[cur], H, S, st.Gh[cur]->nb[1], 0);
        ggml_tensor * Lsur = ggml_sum(ctx, ggml_mul(ctx, Y, dY));  // L' = SUM(Y (.) dY)
        ggml_set_loss(Lsur);
        ggml_set_output(Lsur);
        ggml_build_forward_expand(gf, Lsur);

        lm_ckpt_fill_gacc(r.opt, gf, st.C[(size_t) l], st.Gh[nxt], r.t_one, &st.gacc);
        ggml_build_backward_expand(ctx, gf, st.gacc.data());

        const bool ok = run_graph(gf);
        ggml_free(ctx);
        if (!ok) {
            return false;
        }
        cur = nxt;
    }
    // Gh[cur] now holds dL/d(embedding output) and is DISCARDED: embed_tokens is
    // frozen and carries no LoRA site.
    return true;
}
