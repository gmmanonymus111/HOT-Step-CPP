#pragma once
// sampler-dcw.h — DCW (Differential Correction in Wavelet domain) for hot-step-sampler.h
//
// Per-step DCW correction for SNR-t bias in flow matching.
// 4 modes: low, high, double, pix — each with per-mode t-modulation.
// Operates in-place on the post-step xt buffer.

#include "dcw.h"
#include "hot-step-params.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Apply DCW correction to xt in-place for all N batch items.
// Called once per solver step, after the ODE/SDE step has been applied.
// denoised = xt - vt * t_next (reconstructed from post-step xt).
static void sampler_apply_dcw(
    float *       xt,          // [N * T * Oc]  post-step latents (modified in-place)
    const float * vt,          // [N * T * Oc]  velocity field
    int           N,           // batch size
    int           T,           // time frames
    int           Oc,          // output channels
    float         t_curr,      // current timestep value
    float         t_next,      // next timestep value
    int           step         // current step index (for first-step logging)
) {
    if (!g_hotstep_params.dcw_enabled) return;

    const int n_per = T * Oc;
    const std::string & dcw_mode = g_hotstep_params.dcw_mode;
    bool is_low    = (dcw_mode == "low");
    bool is_high   = (dcw_mode == "high");
    bool is_double = (dcw_mode == "double");
    bool is_pix    = (dcw_mode == "pix");
    if (!is_low && !is_high && !is_double && !is_pix) {
        is_low = true;  // fallback to safest mode
    }

    // Per-mode effective scaler, modulated as the paper prescribes
    float s_low      = t_curr * g_hotstep_params.dcw_scaler;
    float s_high     = (1.0f - t_curr) * g_hotstep_params.dcw_scaler;
    float s_double_h = (1.0f - t_curr) * g_hotstep_params.dcw_high_scaler;
    float s_pix      = g_hotstep_params.dcw_scaler;  // constant per paper

    // Scratch buffers for DWT
    int Tl = (T + 1) / 2;
    std::vector<float> denoised(n_per);
    std::vector<float> tmp_xL(Tl * Oc), tmp_xH(Tl * Oc);
    std::vector<float> tmp_yL(Tl * Oc), tmp_yH(Tl * Oc);

    for (int b = 0; b < N; b++) {
        float *       xt_b = xt + b * n_per;
        const float * vt_b = vt + b * n_per;

        // Denoised from POST-step xt (matches upstream C++ exactly)
        for (int i = 0; i < n_per; i++) {
            denoised[i] = xt_b[i] - vt_b[i] * t_next;
        }

        if (is_low) {
            dcw_haar_low_inplace(xt_b, denoised.data(), T, Oc, s_low,
                                 tmp_xL.data(), tmp_xH.data(),
                                 tmp_yL.data(), tmp_yH.data());
        } else if (is_high) {
            dcw_haar_high_inplace(xt_b, denoised.data(), T, Oc, s_high,
                                  tmp_xL.data(), tmp_xH.data(),
                                  tmp_yL.data(), tmp_yH.data());
        } else if (is_double) {
            dcw_haar_double_inplace(xt_b, denoised.data(), T, Oc, s_low, s_double_h,
                                    tmp_xL.data(), tmp_xH.data(),
                                    tmp_yL.data(), tmp_yH.data());
        } else {  // is_pix
            dcw_pix_inplace(xt_b, denoised.data(), T, Oc, s_pix);
        }
    }

    if (step == 0) {
        fprintf(stderr, "[DCW] mode=%s scaler=%.4f high_scaler=%.4f "
                "(eff_low=%.4f eff_high=%.4f at t=%.3f)\n",
                dcw_mode.c_str(), g_hotstep_params.dcw_scaler,
                g_hotstep_params.dcw_high_scaler,
                s_low, s_double_h, t_curr);
    }
}
