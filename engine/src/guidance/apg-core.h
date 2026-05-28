#pragma once
// apg-core.h: APG (Adaptive Projected Guidance) primitives
//
// Extracted from dit-sampler.h so guidance modes can share the core functions.
// Matches Python ACE-Step-1.5 acestep/models/base/apg_guidance.py

#include <cmath>
#include <cstring>
#include <vector>

// Momentum buffer for APG running average smoothing.
// Matches Python APGMomentumBuffer with momentum=-0.75
struct APGMomentumBuffer {
    double              momentum;
    std::vector<double> running_average;
    bool                initialized;

    APGMomentumBuffer(double m = -0.75) : momentum(m), initialized(false) {}

    void update(const double * values, int n) {
        if (!initialized) {
            running_average.assign(values, values + n);
            initialized = true;
        } else {
            for (int i = 0; i < n; i++) {
                running_average[i] = values[i] + momentum * running_average[i];
            }
        }
    }
};

// Pre-allocated workspace for APG forward pass. Eliminates ~1.5MB of heap
// allocation per step (4 vectors of n doubles each).
// Allocate once before the denoising loop, reuse every step.
struct APGWorkspace {
    std::vector<double> diff;
    std::vector<double> pred_cond_d;
    std::vector<double> par;
    std::vector<double> orth;

    void resize(int n) {
        diff.resize(n);
        pred_cond_d.resize(n);
        par.resize(n);
        orth.resize(n);
    }
};

// project(v0, v1, dims=[1]): decompose v0 into parallel + orthogonal w.r.t. v1
// All math in double precision matching Python .double() calls.
// Layout: memory [T, Oc] time-major (ggml ne=[Oc, T]).
// Python dims=[1] on [B,T,C] = normalize/project per channel over T dimension.
// In memory [T, Oc] layout: for each channel c, operate over all T time frames.
static void apg_project(const double * v0, const double * v1,
                         double * out_par, double * out_orth,
                         int Oc, int T) {
    for (int c = 0; c < Oc; c++) {
        double norm2 = 0.0;
        for (int t = 0; t < T; t++) {
            norm2 += v1[t * Oc + c] * v1[t * Oc + c];
        }
        double inv_norm = (norm2 > 1e-60) ? (1.0 / sqrt(norm2)) : 0.0;

        double dot = 0.0;
        for (int t = 0; t < T; t++) {
            dot += v0[t * Oc + c] * (v1[t * Oc + c] * inv_norm);
        }

        for (int t = 0; t < T; t++) {
            int    idx    = t * Oc + c;
            double v1n    = v1[idx] * inv_norm;
            out_par[idx]  = dot * v1n;
            out_orth[idx] = v0[idx] - out_par[idx];
        }
    }
}

// APG forward matching Python apg_forward() exactly:
//   1. diff = cond - uncond
//   2. momentum.update(diff); diff = running_average
//   3. norm clip: per-channel L2 over T (dims=[1]), clip to norm_threshold=2.5
//   4. project(diff, pred_COND) -> (parallel, orthogonal)
//   5. result = pred_cond + (scale - 1) * orthogonal
// Internal computation in double precision (Python uses .double()).
//
// Overload with pre-allocated workspace (preferred — avoids heap allocs per step).
static void apg_forward(const float *       pred_cond,
                        const float *       pred_uncond,
                        float               guidance_scale,
                        APGMomentumBuffer & mbuf,
                        float *             result,
                        int                 Oc,
                        int                 T,
                        float               norm_threshold,
                        APGWorkspace &      ws) {
    int n = Oc * T;

    // 1. diff = cond - uncond (promote to double)
    for (int i = 0; i < n; i++) {
        ws.diff[i] = (double) pred_cond[i] - (double) pred_uncond[i];
    }

    // 2. momentum update, then use smoothed diff
    mbuf.update(ws.diff.data(), n);
    memcpy(ws.diff.data(), mbuf.running_average.data(), n * sizeof(double));

    // 3. norm clipping: per-channel L2 over T (dims=[1]), clip to threshold
    if (norm_threshold > 0.0f) {
        for (int c = 0; c < Oc; c++) {
            double norm2 = 0.0;
            for (int t = 0; t < T; t++) {
                norm2 += ws.diff[t * Oc + c] * ws.diff[t * Oc + c];
            }
            double norm = sqrt(norm2 > 0.0 ? norm2 : 0.0);
            double s    = (norm > 1e-60) ? fmin(1.0, (double) norm_threshold / norm) : 1.0;
            if (s < 1.0) {
                for (int t = 0; t < T; t++) {
                    ws.diff[t * Oc + c] *= s;
                }
            }
        }
    }

    // 4. project(diff, pred_COND) -> orthogonal component (double precision)
    for (int i = 0; i < n; i++) {
        ws.pred_cond_d[i] = (double) pred_cond[i];
    }
    apg_project(ws.diff.data(), ws.pred_cond_d.data(), ws.par.data(), ws.orth.data(), Oc, T);

    // 5. result = pred_cond + (scale - 1) * orthogonal (back to float)
    double w = (double) guidance_scale - 1.0;
    for (int i = 0; i < n; i++) {
        result[i] = (float) ((double) pred_cond[i] + w * ws.orth[i]);
    }
}

// Legacy overload: allocates workspace internally (for callers that don't pre-allocate).
static void apg_forward(const float *       pred_cond,
                        const float *       pred_uncond,
                        float               guidance_scale,
                        APGMomentumBuffer & mbuf,
                        float *             result,
                        int                 Oc,
                        int                 T,
                        float               norm_threshold = 2.5f) {
    APGWorkspace ws;
    ws.resize(Oc * T);
    apg_forward(pred_cond, pred_uncond, guidance_scale, mbuf, result, Oc, T, norm_threshold, ws);
}
