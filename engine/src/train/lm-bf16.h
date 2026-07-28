#pragma once
// lm-bf16.h — Lever A: `--weights bf16`, engine-side backward surgery.
//
// docs/plans/2026-07-28-lm-speed-levers.md §3 (S2/S3/S4/S18)
//
// WHAT THIS DOES
//
//   Shipped low-VRAM path, per projection, per segment:
//       forward   mul_mat(cast(W_bf16 -> F32), x)     <- the per-segment F32 window
//       backward  out_prod(W_f32, transpose(grad))    <- F32-only => cublasSgemm/TF32
//
//   Under --weights bf16:
//       forward   mul_mat(W_bf16, x)                  <- cublasGemmEx(CUDA_R_16BF)
//       backward  mul_mat(Wt_bf16, grad)              <- same BF16 kernel
//
//   The backward form is produced by running ggml_build_backward_expand()
//   COMPLETELY UNCHANGED and then rewriting, in place, every GGML_OP_OUT_PROD
//   node whose src[0] is one of the segment's BF16 base weights:
//
//       node->op     = GGML_OP_MUL_MAT
//       node->src[0] = Wt      (in-graph cont(transpose(W)), same dtype)
//       node->src[1] = grad    (recovered from the TRANSPOSE view's src[0])
//
//   The rewrite is shape-exact, which is the whole reason it is legal:
//       out_prod(W[K,N], gT[S,N]) -> [K,S]   ==   mul_mat(Wt[N,K], g[N,S]) -> [K,S]
//
// WHY IT IS LEGAL (verified against this tree's ggml, b677b63c)
//
//   * ggml.c:6598-6604 — mul_mat's src1 (activation) gradient is EXACTLY
//     `ggml_out_prod(src0, ggml_transpose(grad))`. [F]
//   * ggml.c:3846-3849 — ggml_transpose keeps `src[0] = a`, so the untransposed
//     gradient is recoverable from the view. [F]
//   * ggml.c:7069 — the `src->type == F32 || F16` assert applies only to sources
//     that NEED gradients. A frozen (non-PARAM) BF16 weight is exempt, which is
//     why ggml_build_backward_expand does not choke on the BF16 forward. [F]
//   * ggml-cuda/cpy.cu:509 — BF16 -> BF16 copy is implemented, so the in-graph
//     cont(transpose(W)) runs on the GPU. [F]
//
//   HOT-Step already ships this exact technique for the output head: lm-ckpt.h
//   keeps `t_embT` in the base's own BF16 and uses ggml_mul_mat(t_embT, dl)
//   instead of out_prod (1,060 MiB instead of 2,120 MiB at 4B). Lever A
//   generalises that proven in-tree pattern to the 7 per-layer projections.
//
// WHERE Wt LIVES (S4)
//
//   IN-GRAPH, exactly where the F32 cast it replaces lives today — NOT in a
//   persistent bank. cont(transpose(W)) costs ~0.041 ms against a ~0.9 ms GEMM
//   [M], and gallocr frees it with the segment. A persistent BF16 transposed
//   bank would be a second 7,991 MiB at 4B, which is unaffordable, not merely
//   unnecessary. spike-bf16layer.h's GbWtBank is a spike convenience and is
//   deliberately NOT ported.
//
// NUMERICS (S6)
//
//   This lever CHANGES THE TRAINED WEIGHTS: the activation gradient is
//   BF16-rounded at every layer. That is deliberate, is recorded in
//   lm_train_log.json's config.weights, and is gated by T14/T15/T17.
//
// THE TRIPWIRE (S18)
//
//   A `--weights bf16` run that silently fell back to the F32 cast would be a
//   TF32 run wearing a BF16 label. The surgery depends on the SHAPE of ggml's
//   MUL_MAT backward, which a future upstream sync could reformulate (the
//   `mul_mat(cont(transpose(src0)), grad)` form sits commented out right beside
//   it at ggml.c:6600-6604). So every segment asserts EXACTLY 7 rewrites, 0
//   skipped and 0 residual base-weight out_prod, and a miss is a GGML_ABORT,
//   never a warning.
//
//   >>> Add `ace-train train-lm --self-test` (gate T16) to the upstream-sync
//   >>> checklist alongside engine/verify-hooks.ps1. It is a 30-second check
//   >>> that turns a silent regression into a loud one.

// DEVIATION vs plan §1.1's "includes lm-graph.h and ggml*.h only". The
// dependency runs the OTHER way: LmLayerOpts (in lm-graph.h) has to hold an
// `LmWtCollect *`, and lm_linear() has to dereference it, so lm-graph.h needs
// the complete type. This header therefore includes only qwen3-lm.h (for
// Qwen3LM/Qwen3Layer) and lm-graph.h includes THIS. That is a strictly smaller
// include set than the plan assumed and keeps every prohibition intact: no
// spike-*.h, no dit-*.h, no model-store.h, no pipeline-*.h.
#include "qwen3-lm.h"

#include <cstring>
#include <map>
#include <vector>

// ─── the per-graph W -> Wt map ──────────────────────────────────────────────
//
// One of these lives on the stack of each P7 segment build. `map` de-duplicates
// (a weight used twice in one segment gets one transpose); `nodes` preserves
// creation order so lm_bf16_expand_wt() can put them into the graph before the
// loss root — see §3.2, the one thing that is easy to get wrong.
struct LmWtCollect {
    std::map<ggml_tensor *, ggml_tensor *> map;    // W -> cont(transpose(W))
    std::vector<ggml_tensor *>             nodes;  // in creation order
};

// §3.2 step 2. MUST be called AFTER the layer subgraph is created in `ctx` and
// BEFORE ggml_build_forward_expand(gf, Lsur).
//
// ggml_build_forward_expand appends a subtree's nodes at the END of the node
// array in call order. The rewritten backward node consumes Wt, so Wt must sit
// at a LOWER node index than its consumer. Expanding Wt after the backward was
// built would place it after its own consumer and compute garbage — silently,
// because the shapes would still line up.
static inline void lm_bf16_expand_wt(ggml_cgraph * gf, const LmWtCollect & wc) {
    for (size_t i = 0; i < wc.nodes.size(); i++) {
        ggml_build_forward_expand(gf, wc.nodes[i]);
    }
}

// ─── the surgery (§3.3) ─────────────────────────────────────────────────────
//
// Returns the number of nodes rewritten. `skipped` counts nodes that matched a
// collected weight but whose operand shape did NOT match the expected pattern —
// that is the "upstream changed the backward" signal and must be zero.
static inline int lm_bf16_rewrite_outprod(ggml_cgraph * gf, const LmWtCollect & wc, int * skipped) {
    int            n     = 0;
    ggml_tensor ** nodes = ggml_graph_nodes(gf);
    const int      cnt   = ggml_graph_n_nodes(gf);

    for (int i = 0; i < cnt; i++) {
        ggml_tensor * nd = nodes[i];
        if (nd->op != GGML_OP_OUT_PROD) {
            continue;
        }
        std::map<ggml_tensor *, ggml_tensor *>::const_iterator it = wc.map.find(nd->src[0]);
        if (it == wc.map.end()) {
            continue;  // LoRA A/B out_prods: F32, tiny, and NOT ours. Leave alone.
        }

        ggml_tensor * Wt = it->second;
        ggml_tensor * gT = nd->src[1];

        // ggml.c:6598-6604 builds src[1] as ggml_transpose(grad); ggml.c:3848
        // keeps src[0] == grad. Anything else means the backward was rewritten
        // upstream and this lever is no longer valid.
        if (!gT || gT->op != GGML_OP_TRANSPOSE || !gT->src[0]) {
            (*skipped)++;
            continue;
        }
        ggml_tensor * g = gT->src[0];

        // out_prod(W[K,N], gT[S,N]) -> [K,S]  ==  mul_mat(Wt[N,K], g[N,S]) -> [K,S]
        if (Wt->ne[0] != g->ne[0] || nd->ne[0] != Wt->ne[1] || nd->ne[1] != g->ne[1] || !ggml_is_contiguous(g)) {
            (*skipped)++;
            continue;
        }

        nd->op     = GGML_OP_MUL_MAT;
        nd->src[0] = Wt;
        nd->src[1] = g;
        memset(nd->op_params, 0, sizeof(nd->op_params));
        n++;
    }
    return n;
}

// S18 helper: base-weight out_prod nodes still present after the rewrite. Must
// be 0 — a non-zero count means some projection is still on the F32/TF32 path
// while the run reports "bf16".
static inline int lm_bf16_count_base_outprod(const ggml_cgraph * gf, const LmWtCollect & wc) {
    int            n     = 0;
    ggml_tensor ** nodes = ggml_graph_nodes(const_cast<ggml_cgraph *>(gf));
    const int      cnt   = ggml_graph_n_nodes(const_cast<ggml_cgraph *>(gf));
    for (int i = 0; i < cnt; i++) {
        if (nodes[i]->op == GGML_OP_OUT_PROD && wc.map.count(nodes[i]->src[0])) {
            n++;
        }
    }
    return n;
}

// The number of projections whose backward this lever rewrites, per segment.
// Batching adds no out_prod nodes, so this is B-independent (S16) — which is
// exactly what makes it a good tripwire.
#define LM_BF16_REWRITES_PER_SEGMENT 7

// The three numbers a segment's surgery must produce. Exposed so gate T16 can
// REPORT them instead of aborting on them.
struct LmBf16Counts {
    int rewritten = 0;
    int skipped   = 0;
    int left      = 0;  // base-weight out_prod nodes still present

    bool ok() const { return rewritten == LM_BF16_REWRITES_PER_SEGMENT && skipped == 0 && left == 0; }
};

static inline LmBf16Counts lm_bf16_rewrite_segment(ggml_cgraph * gf, const LmWtCollect & wc) {
    LmBf16Counts cn;
    cn.rewritten = lm_bf16_rewrite_outprod(gf, wc, &cn.skipped);
    cn.left      = lm_bf16_count_base_outprod(gf, wc);
    return cn;
}

// §3.2 step 5, factored so P7 and lm_ckpt_probe_segment_nodes() cannot drift.
static inline void lm_bf16_finish_segment(ggml_cgraph * gf, const LmWtCollect & wc) {
    const LmBf16Counts cn      = lm_bf16_rewrite_segment(gf, wc);
    const int          n       = cn.rewritten;
    const int          skipped = cn.skipped;
    const int          left    = cn.left;
    if (!cn.ok()) {
        GGML_ABORT(
            "lm-bf16 (S18): backward surgery tripwire — rewrote %d, skipped %d, base out_prod left %d "
            "(expected %d / 0 / 0). ggml's MUL_MAT backward no longer matches the shape this lever rewrites; "
            "'--weights bf16' would silently be a TF32 run wearing a BF16 label. Re-run with "
            "'--weights f32-window' and re-check gate T16 (ace-train train-lm --self-test) after an upstream sync.",
            n, skipped, left, LM_BF16_REWRITES_PER_SEGMENT);
    }
}

// ─── base-dtype gate (§3.4) ─────────────────────────────────────────────────
//
// Every layer's 7 projections must be BF16. F16 bases are REFUSED — a parallel
// F16 route is future work, and a silent fallback is exactly the failure mode
// S18 forbids. Quantized bases are already refused upstream by
// lm_ckpt_check_base() (lm-ckpt.h:149).
static inline bool lm_bf16_base_is_bf16(const Qwen3LM & lm) {
    for (int i = 0; i < lm.cfg.n_layers; i++) {
        const Qwen3Layer & ly     = lm.layers[i];
        const ggml_tensor * ps[7] = { ly.q_proj,    ly.k_proj,  ly.v_proj, ly.o_proj,
                                      ly.gate_proj, ly.up_proj, ly.down_proj };
        for (int j = 0; j < 7; j++) {
            if (!ps[j] || ps[j]->type != GGML_TYPE_BF16) {
                return false;
            }
        }
    }
    return lm.cfg.n_layers > 0;
}

// The dtype name of the first projection, for the fatal message.
static inline const char * lm_bf16_base_proj_type_name(const Qwen3LM & lm) {
    if (lm.cfg.n_layers > 0 && lm.layers[0].q_proj) {
        return ggml_type_name(lm.layers[0].q_proj->type);
    }
    return "unknown";
}
