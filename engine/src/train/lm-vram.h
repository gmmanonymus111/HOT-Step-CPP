#pragma once
// lm-vram.h — the footprint model, the auto-fit binary search (L8), the
// high-water probe and the leak counter (L10).
//
// Footprint model (plan §3.7), everything in F32 bytes:
//
//   bytes(S, s_tr) = mirror
//                  + 3 * V * s_tr * 4                 // logits + logits-grad + labels
//                  + 4 * n_lora * 4                   // params + accs + m + v
//                  + L * ( c1*Nh*S*S + (c2f*F + c2h*H)*S ) * 4
//                  + static                           // causal mask + token/pos inputs
//   with c1 = 1.135, c2f = 2.4, c2h = 1.0
//
// c1/c2 are calibrated on the two measured 0.6B points (S=1536 -> 6,351 MB of
// activations; S=2113 -> 11,174 MB) and reproduce both by construction.
//
// DEVIATION (documented): the plan's formula omits the persistent causal-mask
// buffer. It is real (S*S*4 bytes: 17.8 MB at S=2113, 67 MB at S=4096) and is
// added here as an explicit `static` term. It is ~0.1 % of the total at the
// calibration points, so the c1/c2 fit is unaffected.
//
// docs/plans/2026-07-27-lm-trainer-implementation.md §3.7

#include "train/lm-common.h"

#include <algorithm>
#include <string>

struct LmVramModel {
    // model geometry
    int    n_layers = 0;
    int    hidden   = 0;
    int    ffn      = 0;
    int    n_heads  = 0;
    int    vocab    = 0;
    size_t mirror_bytes = 0;
    size_t lora_params  = 0;

    static constexpr double c1  = 1.135;
    static constexpr double c2f = 2.4;
    static constexpr double c2h = 1.0;
};

static double lm_vram_activation_bytes(const LmVramModel & m, int S) {
    const double dS = (double) S;
    const double per_layer =
        LmVramModel::c1 * (double) m.n_heads * dS * dS +
        (LmVramModel::c2f * (double) m.ffn + LmVramModel::c2h * (double) m.hidden) * dS;
    return (double) m.n_layers * per_layer * 4.0;
}

static double lm_vram_static_bytes(int S) {
    return (double) S * (double) S * 4.0   // causal mask [S*S] f32
           + (double) S * 4.0 * 2.0;       // tokens + positions (i32)
}

static double lm_vram_bytes(const LmVramModel & m, int S, int s_tr) {
    return (double) m.mirror_bytes                                    //
           + 3.0 * (double) m.vocab * (double) s_tr * 4.0             // logits + grad + labels
           + 4.0 * (double) m.lora_params * 4.0                       // params + acc + m + v
           + lm_vram_activation_bytes(m, S)                           //
           + lm_vram_static_bytes(S);
}

// 0.62 = 1272/2113, the measured trained fraction of a real full-song sequence.
// It only sizes the label buffer, and the graph views a prefix of it, so a low
// estimate is self-correcting while a high one merely wastes VRAM.
static inline int lm_vram_str_est(int S) {
    return (int) ((double) S * 0.62 + 0.5);
}

struct LmVramFit {
    int    max_len   = 0;
    double est_bytes = 0.0;
    size_t free_mb   = 0;
    size_t total_mb  = 0;
    bool   ok        = false;
};

// Largest S in [1024, 8192] (step 64) whose predicted footprint fits the
// budget. `user_len > 0` skips the search but still checks the budget.
static LmVramFit lm_vram_fit(const LmVramModel & m, ggml_backend_t backend, int reserve_mb, int user_len) {
    LmVramFit r;
    size_t    fb = 0, tb = 0;
    lm_vram_query(backend, &fb, &tb);
    r.free_mb  = fb / (1024 * 1024);
    r.total_mb = tb / (1024 * 1024);

    // `fb` is queried AFTER lm_build_f32_mirror() allocated the mirror and freed
    // the BF16 buffer, so the mirror is ALREADY subtracted from it. lm_vram_bytes()
    // also includes mirror_bytes, so the budget must add it back — otherwise the
    // mirror is charged twice and maxLen collapses (0.6B 2944 -> ~3200 once fixed;
    // 1.7B ~1891 -> ~2515, i.e. most of a real dataset would be skipped as
    // over-length). The plan's §3.7 pseudocode carries the same defect.
    const double budget = (double) fb + (double) m.mirror_bytes - (double) reserve_mb * 1048576.0;

    if (user_len > 0) {
        r.max_len   = user_len;
        r.est_bytes = lm_vram_bytes(m, user_len, lm_vram_str_est(user_len));
        r.ok        = r.est_bytes <= budget;
        return r;
    }

    int best = 0;
    for (int S = 1024; S <= 8192; S += 64) {
        if (lm_vram_bytes(m, S, lm_vram_str_est(S)) <= budget) {
            best = S;
        } else {
            break;  // monotonically increasing in S
        }
    }
    r.max_len   = best > 0 ? best : 1024;
    r.est_bytes = lm_vram_bytes(m, r.max_len, lm_vram_str_est(r.max_len));
    r.ok        = best > 0;
    return r;
}

// ─── low-VRAM footprint model (4B plan §3.8) ────────────────────────────────
//
// PURELY ADDITIVE. Everything above this line is byte-identical to the shipped
// header (dit-vram.h reuses LmVramTracker, and gate (e) compares the naive
// maxLen integer bit-for-bit).
//
//   resident(S) = base_bytes                       // BF16 base, never mirrored
//               + emb_t_bytes                      // [V,H] transpose for dL/dh
//               + L * H * S * 4                    // checkpoints C[0..L-1]
//               + 3 * H * S * 4                    // t_H, t_G(==Gh0), Gh1
//               + Nh * D * S * 4                   // t_zero_attn (0 when hb off)
//               + S*S*4 + 2*S*4                    // causal mask + tok/pos
//               + V * chunk * 4                    // per-chunk one-hot labels
//               + 4 * n_lora * 4                   // params + acc + m + v
//
//   seg_bwd(S)  = w_layer_f32 + 3*hb*S*S*4 + 2*(c2f*F + c2h*H)*S*4 + 2*Nh*D*S*4
//   seg_fwd(S)  = w_layer_f32 + 2*hb*S*S*4 +   (c2f*F + c2h*H)*S*4 + 2*Nh*D*S*4
//   head        = 3 * V * chunk * 4
//   bytes       = resident + max(seg_bwd, seg_fwd, head)
//
// c2f/c2h are the [M]-calibrated constants of the naive model, reused verbatim.
// The 3x/2x multipliers and the hb*S^2 term are DERIVED from the op-level
// analysis in D6 (soft_max_ext_back holds `sm`, `grad_sm` and `d_scores` live
// simultaneously), NOT fitted. Honest uncertainty band: +-15 %.
//
// DEVIATION vs plan §1.1: LmVramLowCfg carries `head_dim` and `layer_w_bytes`
// in addition to the four fields the plan lists. LmVramModel has no head_dim
// field and the plan requires it to stay byte-identical, so the geometry the
// low-VRAM terms need has to live here.

enum LmVramMode { LM_VRAM_NAIVE, LM_VRAM_LOWVRAM };

struct LmVramLowCfg {
    int    attn_head_block = 0;
    int    chunk           = 128;
    int    head_dim        = 0;
    size_t base_bytes      = 0;  // ggml_backend_buffer_get_size(lm.wctx.buffer)
    size_t emb_t_bytes     = 0;  // V * H * ggml_type_size(embed dtype)
    size_t layer_w_bytes   = 0;  // lm_layer_weight_bytes(cfg) — the F32 window

    // ── Lever A (2026-07-28 speed-levers plan §3.5) ──────────────────────
    // Default OFF, so lm_vram_lowvram_transient() below selects layer_w_bytes
    // exactly as it ships and the §6.0 estMb integer is unchanged.
    size_t layer_wt_bytes  = 0;      // lm_layer_proj_bytes(cfg, BF16) — the BF16
                                     // transposed window that replaces the F32 one
    bool   weights_bf16    = false;
};

static double lm_vram_lowvram_resident(const LmVramModel & m, const LmVramLowCfg & lc, int S) {
    const double dS = (double) S;
    const double hbz =
        (lc.attn_head_block > 0) ? (double) m.n_heads * (double) lc.head_dim * dS * 4.0 : 0.0;  // t_zero_attn
    return (double) lc.base_bytes + (double) lc.emb_t_bytes                                //
           + (double) m.n_layers * (double) m.hidden * dS * 4.0                            // C[0..L-1]
           + 3.0 * (double) m.hidden * dS * 4.0                                            // t_H, t_G, Gh1
           + hbz                                                                           //
           + lm_vram_static_bytes(S)                                                       // mask + tok/pos
           + (double) m.vocab * (double) lc.chunk * 4.0                                    // t_labc
           + 4.0 * (double) m.lora_params * 4.0;                                           // param/acc/m/v
}

static double lm_vram_lowvram_transient(const LmVramModel & m, const LmVramLowCfg & lc, int S) {
    const double dS  = (double) S;
    const double hb  = (lc.attn_head_block > 0) ? (double) lc.attn_head_block : (double) m.n_heads;
    const double s2  = hb * dS * dS * 4.0;
    const double non = (LmVramModel::c2f * (double) m.ffn + LmVramModel::c2h * (double) m.hidden) * dS * 4.0;
    const double qkv = 2.0 * (double) m.n_heads * (double) lc.head_dim * dS * 4.0;  // acc + cont churn
    // §3.5: Lever A replaces the per-segment F32 weight window with a BF16
    // TRANSPOSED window over the 7 projections only — 192.5 MiB instead of
    // 385.0 MiB at 4B, i.e. the lever is VRAM-POSITIVE.
    //
    // MEASURED, not just modelled — and the two do not agree, so read the
    // measurement. LmVramTracker::peak_mb over the §6.5 runs: 4B 12,466 ->
    // 12,363 MiB, i.e. -103 MiB against the -192.5 MiB this line predicts
    // (0.6B: 3,267 -> 3,242 = -25 vs -30). Checking the estimate against itself
    // would of course reproduce -192.5 exactly; it is circular and means nothing.
    // Second-order effect that matters if this ever feeds the auto-fit: the
    // model already under-predicts the real 4B peak by 340 MiB on the f32
    // window (est 12,126 vs peak 12,466), and enabling the lever widens that to
    // 429 MiB (est 11,934 vs peak 12,363) — ~26 % less conservative. Still
    // inside the 1,024 MiB reserve, but the headroom is smaller than the
    // arithmetic suggests.
    const double w_window = (lc.weights_bf16 && lc.layer_wt_bytes > 0) ? (double) lc.layer_wt_bytes
                                                                      : (double) lc.layer_w_bytes;
    const double seg_bwd = w_window + 3.0 * s2 + 2.0 * non + qkv;
    const double seg_fwd = w_window + 2.0 * s2 + non + qkv;
    const double head    = 3.0 * (double) m.vocab * (double) lc.chunk * 4.0;
    return std::max(seg_bwd, std::max(seg_fwd, head));
}

// `s_tr` is accepted for signature symmetry with lm_vram_bytes(); the low-VRAM
// label buffer is `chunk`-sized, so it genuinely does not depend on s_tr.
static double lm_vram_bytes_lowvram(const LmVramModel & m, const LmVramLowCfg & lc, int S, int s_tr) {
    (void) s_tr;
    return lm_vram_lowvram_resident(m, lc, S) + lm_vram_lowvram_transient(m, lc, S);
}

// Same binary search as lm_vram_fit(). `fb` is queried BEFORE any mirror exists
// but the base buffer is already allocated and therefore already subtracted from
// it, while lm_vram_bytes_lowvram() charges base_bytes — so, exactly as in the
// naive fit, the budget has to add it back.
static LmVramFit lm_vram_fit_lowvram(const LmVramModel & m, const LmVramLowCfg & lc, ggml_backend_t backend,
                                     int reserve_mb, int user_len) {
    LmVramFit r;
    size_t    fb = 0, tb = 0;
    lm_vram_query(backend, &fb, &tb);
    r.free_mb  = fb / (1024 * 1024);
    r.total_mb = tb / (1024 * 1024);

    const double budget = (double) fb + (double) lc.base_bytes - (double) reserve_mb * 1048576.0;

    if (user_len > 0) {
        r.max_len   = user_len;
        r.est_bytes = lm_vram_bytes_lowvram(m, lc, user_len, lm_vram_str_est(user_len));
        r.ok        = r.est_bytes <= budget;
        return r;
    }

    int best = 0;
    for (int S = 1024; S <= 8192; S += 64) {
        if (lm_vram_bytes_lowvram(m, lc, S, lm_vram_str_est(S)) <= budget) {
            best = S;
        } else {
            break;
        }
    }
    r.max_len   = best > 0 ? best : 1024;
    r.est_bytes = lm_vram_bytes_lowvram(m, lc, r.max_len, lm_vram_str_est(r.max_len));
    r.ok        = best > 0;
    return r;
}

// §3.2 step 6. `naive_max_len` is the integer the SHIPPED fit produces; passing
// 0 means "not evaluated" (the caller already forced a mode).
static inline LmVramMode lm_vram_pick_mode(const std::string & flag, const std::string & size_label,
                                           int naive_max_len) {
    if (flag == "on") {
        return LM_VRAM_LOWVRAM;
    }
    if (flag == "off") {
        return LM_VRAM_NAIVE;
    }
    if (size_label == "4B") {
        return LM_VRAM_LOWVRAM;
    }
    return (naive_max_len > 0 && naive_max_len < 2048) ? LM_VRAM_LOWVRAM : LM_VRAM_NAIVE;
}

// ─── leak counter (L10 / G3) ────────────────────────────────────────────────

// TRAINER-OWNED accounting, not device-wide.
//
// lm_vram_used_mb() is (total - free) for the whole device, so it carries every
// other process on the card (~3.2 GB of desktop apps on this machine). That
// makes `step.vramMb` incomparable with the trainer-only `estMb` printed beside
// it (§2.2's own example has them equal), and it lets an unrelated process move
// the L10 leak delta in either direction. Instead we sum the buffers we own plus
// the scheduler's own compute arena, which is exactly what a leak would grow.
struct LmVramTracker {
    ggml_backend_t       backend     = nullptr;
    ggml_backend_sched_t sched       = nullptr;
    size_t               fixed_bytes = 0;  // mirror + static inputs + LoRA + optimizer state
    size_t               base_mb     = 0;  // sampled right after the high-water probe
    size_t               peak_mb     = 0;
    size_t               last_mb     = 0;
    long long            max_delta   = 0;
    bool                 warned      = false;

    size_t used_mb() const {
        size_t b = fixed_bytes;
        if (sched && backend) {
            b += ggml_backend_sched_get_buffer_size(sched, backend);
        }
        return b / (1024 * 1024);
    }

    void probe_baseline(ggml_backend_t b, ggml_backend_sched_t s, size_t fixed) {
        backend     = b;
        sched       = s;
        fixed_bytes = fixed;
        base_mb     = used_mb();
        peak_mb     = base_mb;
        last_mb     = base_mb;
    }

    size_t sample() {
        last_mb = used_mb();
        if (last_mb > peak_mb) {
            peak_mb = last_mb;
        }
        const long long d = (long long) last_mb - (long long) base_mb;
        if (d > max_delta) {
            max_delta = d;
        }
        if (!warned && max_delta > 512) {
            warned = true;
            char b[192];
            snprintf(b, sizeof(b), "trainer VRAM grew %lld MB since the high-water probe — possible leak", max_delta);
            lm_log("warn", b);
        }
        return last_mb;
    }
};
