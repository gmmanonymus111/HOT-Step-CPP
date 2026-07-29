#pragma once
// dit-vram.h — the footprint model, the crop/depth auto-fit ladder and the
// card-support gate (plan §3.7, D9/D10/D11).
//
// Model (S = crop / patch, K = trained decoder layers):
//
//   arena_MB(S,K) = K*(0.3274*S + 1.139e-4*S^2) + (-0.125*S + 1.863e-4*S^2)
//   fixed_bytes(K) = mirror_bytes(K) + 16*adapter_params(K) + static input buffers
//   trainer_MB(S,K) = fixed + arena [+ lokr apply arena, LoKR only]
//
// The arena coefficients are the verifier's refit over all 12 measured points
// (the (K/32)*f(S) form has no K-independent term and under-predicts by 10-25%
// exactly in the low-K / long-crop corner the small-card map depends on).
//
// K10 ("reuse the polynomial as-is for LoKR") turned out to hold for the LoRA
// runs it was measured on and NOT for LoKR: every one of those 12 points is a
// LoRA run, whose adapter branch retains a single [rank, S] bottleneck, whereas
// the LoKR kron-matvec retains two activation-shaped tensors per site. The
// difference is thousands of MB at a long crop, which is how a LoKR run at the
// "fitted" crop ends up spilling into Windows' shared GPU memory. LoKR therefore
// adds dit_lokr_apply_arena_bytes() on top; the LoRA path is untouched.
//
// DEVIATION (E2, documented in the handoff): §3.7's fixed term is the fitted
// constant `7786 + (243.0 + state_MB_per_layer)*K`. We instead compute the mirror
// byte count EXACTLY by walking the same slot list the mirror builder walks
// (dit_mirror_bytes_for), and the optimizer state exactly from the adapter's own
// parameter count. That reproduces the fit's intent without its +-3 MB residual
// and, unlike the constant, stays correct for a base with different geometry.
// The arena term — the part that genuinely needs a fit — is used verbatim.
//
// docs/plans/2026-07-28-dit-trainer-implementation.md §3.7

#include "train/dit-adapter-lokr.h"
#include "train/dit-adapter.h"
#include "train/dit-mirror.h"
#include "train/gpu-mem.h"
#include "train/lm-common.h"
#include "train/lm-vram.h"  // LmVramTracker (leak counter), reused unchanged

#include <algorithm>
#include <string>
#include <vector>

struct DitVramModel {
    DiTGGML * m       = nullptr;
    int       n_layers = 0;
    int       patch    = 2;
    int       rank     = 16;
    bool      target_mlp = false;
    // Adapter parameterization: the params term is the only footprint that
    // differs, and the arena polynomial is reused as-is (K10).
    bool is_lokr     = false;
    int  lokr_dim    = 512;
    int  lokr_factor = 6;
    // Mirror precision: the fixed term is the mirror, so this is the single
    // largest lever in the whole model (--mirror bf16 roughly halves it).
    DitMirrorMode mirror = DIT_MIRROR_F32;
    // static input geometry
    int in_ch = 192, out_ch = 64, hidden = 2560, enc_H = 2048, enc_S = 0;
};

// arena (scheduler compute buffer), bytes.
static double dit_vram_arena_bytes(int S, int K) {
    const double dS = (double) S;
    const double mb = (double) K * (0.3274 * dS + 1.139e-4 * dS * dS) + (-0.125 * dS + 1.863e-4 * dS * dS);
    return (mb > 0.0 ? mb : 0.0) * 1048576.0;
}

// persistent input tensors we allocate ourselves, bytes.
static double dit_vram_static_bytes(const DitVramModel & vm, int crop) {
    const int    S  = crop / vm.patch;
    const double b  = (double) vm.in_ch * crop * 4.0                        // t_input
                    + (double) vm.enc_H * std::max(1, vm.enc_S) * 4.0       // t_enc
                    + (double) S * 4.0                                      // t_pos
                    + (double) vm.hidden * 7.0 * 4.0                        // t_temb + t_tproj
                    + (double) S * S * 2.0                                  // t_sa  f16
                    + (double) std::max(1, vm.enc_S) * S * 2.0              // t_ca  f16
                    + (double) vm.out_ch * crop * 4.0                       // t_vtgt
                    + (double) vm.out_ch * 4.0                              // t_cw
                    + 64.0;                                                 // scalars
    return b;
}

static size_t dit_vram_adapter_params(const DitVramModel & vm, int K) {
    return vm.is_lokr ? dit_lokr_expected_params(vm.m->cfg, vm.n_layers - K, vm.n_layers, vm.lokr_dim, vm.lokr_factor,
                                                 vm.target_mlp)
                      : dit_lora_expected_params(vm.m->cfg, vm.n_layers - K, vm.n_layers, vm.rank, vm.target_mlp);
}

// mirror + optimizer state + adapter params, bytes. K = trained depth (top-K).
static double dit_vram_fixed_bytes(const DitVramModel & vm, int K, int crop) {
    const size_t mirror = dit_mirror_bytes_for(vm.m, vm.n_layers - K, vm.mirror);
    const size_t np     = dit_vram_adapter_params(vm, K);
    return (double) mirror + 16.0 * (double) np + dit_vram_static_bytes(vm, crop);
}

static double dit_vram_total_bytes(const DitVramModel & vm, int crop, int K) {
    const int S = crop / vm.patch;
    double    b = dit_vram_fixed_bytes(vm, K, crop) + dit_vram_arena_bytes(S, K);
    if (vm.is_lokr) {
        // The intermediates the fitted polynomial never saw. Added here rather
        // than at any one call site so every consumer — the crop walk, the depth
        // ladder, the `vram` JSONL est and dit_train_log.json — sees them too.
        b += dit_lokr_apply_arena_bytes(vm.m->cfg, vm.n_layers - K, vm.n_layers, vm.lokr_dim, vm.lokr_factor,
                                        vm.target_mlp, S);
    }
    return b;
}

// ─── auto-fit ───────────────────────────────────────────────────────────────

struct DitVramFit {
    int    crop = 0, layers = 0;
    double est_bytes    = 0.0;
    size_t free_mb      = 0, total_mb = 0;
    const char * free_source = "cuda";  // which probe produced free_mb (gpu-mem.h)
    bool   ok           = false;
    bool   crop_user    = false;
    bool   layers_user  = false;
    bool   lowered      = false;  // the ladder had to reduce depth
    double floor_bytes  = 0.0;    // fixed footprint at K=2, crop_min — the refusal number
};

// Largest patch-aligned crop in [crop_min, cap] whose footprint fits `budget`.
// 0 when even crop_min does not fit.
static int dit_vram_best_crop(const DitVramModel & vm, int K, double budget, int crop_min, int cap) {
    int best = 0;
    for (int crop = crop_min; crop <= cap; crop += vm.patch) {
        if (dit_vram_total_bytes(vm, crop, K) <= budget) {
            best = crop;
        } else {
            break;  // monotonically increasing in crop
        }
    }
    return best;
}

// `user_layers` / `user_crop` are 0 for auto. `max_T` is the longest song, which
// caps the useful crop (songs shorter than the crop use their whole length).
//
// `mem` is the honest device figure (gpu-mem.h). Passing nullptr keeps the old
// cudaMemGetInfo behaviour; the trainer always passes one, because under WDDM
// that probe over-reports free VRAM by ~9 GB and the budget below is only as
// good as the number it starts from.
static DitVramFit dit_vram_fit(const DitVramModel & vm, ggml_backend_t backend, int reserve_mb, float safety,
                               int user_crop, int user_layers, int crop_min, int crop_max, int max_T,
                               const DitGpuMem * mem = nullptr) {
    DitVramFit r;
    size_t     fb = 0, tb = 0;
    if (mem) {
        fb            = mem->free;
        tb            = mem->total;
        r.free_source = mem->source();
    } else {
        lm_vram_query(backend, &fb, &tb);
    }
    r.free_mb     = fb / (1024 * 1024);
    r.total_mb    = tb / (1024 * 1024);
    r.crop_user   = user_crop > 0;
    r.layers_user = user_layers > 0;

    const double budget = ((double) fb - (double) reserve_mb * 1048576.0) * (double) (1.0f - safety);

    int cap = std::min(crop_max, max_T);
    cap -= cap % vm.patch;
    int cmin = std::max(vm.patch, crop_min);
    cmin -= cmin % vm.patch;
    if (cmin > cap) {
        cmin = cap;  // a dataset shorter than crop_min still trains on whole songs
    }

    r.floor_bytes = dit_vram_total_bytes(vm, cmin, 2);

    const int ladder[] = { vm.n_layers, 16, 8, 4, 2 };
    const int n_rungs  = (int) (sizeof(ladder) / sizeof(ladder[0]));

    // Fixed axes: honour them, but still refuse an over-budget combination.
    if (r.crop_user || r.layers_user) {
        int K    = r.layers_user ? std::min(user_layers, vm.n_layers) : vm.n_layers;
        int crop = r.crop_user ? user_crop : 0;
        if (crop > 0) {
            crop -= crop % vm.patch;
            if (crop > max_T) {
                crop = max_T - (max_T % vm.patch);
            }
        } else {
            crop = dit_vram_best_crop(vm, K, budget, cmin, cap);
        }
        // A user crop with AUTO depth still leaves one axis to fit, so walk the
        // same depth ladder instead of refusing outright: a fixed window (the
        // LoKR form's 60 s default) degrades to partial depth — which §2.8's
        // banner then reports, since layersSource stays "auto" — rather than
        // failing the run before it starts. A depth the USER pinned is never
        // lowered behind their back; that combination is still refused.
        if (crop > 0 && !r.layers_user && dit_vram_total_bytes(vm, crop, K) > budget) {
            for (int i = 1; i < n_rungs; i++) {
                if (ladder[i] >= K) {
                    continue;
                }
                if (dit_vram_total_bytes(vm, crop, ladder[i]) <= budget) {
                    K = ladder[i];
                    break;
                }
            }
        }
        r.layers     = K;
        r.crop       = crop;
        r.est_bytes  = crop > 0 ? dit_vram_total_bytes(vm, crop, K) : r.floor_bytes;
        r.ok         = crop > 0 && r.est_bytes <= budget;
        r.lowered    = K < vm.n_layers;
        return r;
    }

    // Both auto: depth ladder, re-raising the crop at every rung.
    for (int i = 0; i < n_rungs; i++) {
        const int K = ladder[i];
        if (K > vm.n_layers || (i > 0 && K >= ladder[i - 1])) {
            continue;
        }
        const int crop = dit_vram_best_crop(vm, K, budget, cmin, cap);
        if (crop > 0) {
            r.layers    = K;
            r.crop      = crop;
            r.est_bytes = dit_vram_total_bytes(vm, crop, K);
            r.ok        = true;
            r.lowered   = K < vm.n_layers;
            return r;
        }
    }
    r.layers    = 2;
    r.crop      = 0;
    r.est_bytes = r.floor_bytes;
    r.ok        = false;
    return r;
}

// D9: 12 GB does not fit at ANY depth (the K=2 floor alone exceeds the budget).
// Refuse it by name rather than letting the ladder produce a cryptic vram fatal.
#define DIT_MIN_VRAM_MB 16384

static bool dit_vram_card_supported(size_t total_mb) {
    return total_mb + 512 >= (size_t) DIT_MIN_VRAM_MB;  // 512 MB slack for reporting rounding
}
