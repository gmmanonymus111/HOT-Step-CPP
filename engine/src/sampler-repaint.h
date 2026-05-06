#pragma once
// sampler-repaint.h — Repaint injection and boundary blend for hot-step-sampler.h
//
// Per-step repaint injection replaces preserved regions with noised source.
// Post-loop boundary blend smooths repaint zone edges in latent space.

#include <cstdio>

// Per-step repaint injection: replace preserved regions with noised source.
// Called once per solver step inside the main loop.
static void sampler_repaint_inject(
    float *       xt,                    // [N * T * Oc]  current latents (modified in-place)
    const float * noise,                 // [N * T * Oc]  original noise
    const float * repaint_src,           // [T * Oc]      clean source latent (single)
    int           N, int T, int Oc,
    int           repaint_t0,            // start frame of repaint zone
    int           repaint_t1,            // end frame of repaint zone
    float         repaint_injection_ratio,
    int           step, int num_steps,
    float         t_next                 // next timestep value
) {
    if (!repaint_src || repaint_t1 <= repaint_t0) return;

    int injection_cutoff = (int) (repaint_injection_ratio * (float) num_steps + 0.5f);
    if (step >= injection_cutoff) return;

    const int n_per = T * Oc;
    for (int b = 0; b < N; b++) {
        for (int t = 0; t < T; t++) {
            if (t < repaint_t0 || t >= repaint_t1) {
                for (int ch = 0; ch < Oc; ch++) {
                    int idx = b * n_per + t * Oc + ch;
                    xt[idx] = t_next * noise[idx] + (1.0f - t_next) * repaint_src[t * Oc + ch];
                }
            }
        }
    }
}

// Post-loop boundary blend: smooth repaint zone edges in latent space.
// Applies cosine-like ramp at [t0-cf, t0) and [t1, t1+cf) boundaries.
// Called once after the main solver loop completes.
static void sampler_repaint_blend(
    float *       output,                // [N * T * Oc]  final output latents (modified in-place)
    const float * repaint_src,           // [T * Oc]      clean source latent
    int           N, int T, int Oc,
    int           repaint_t0,
    int           repaint_t1,
    int           repaint_crossfade_frames
) {
    if (!repaint_src || repaint_t1 <= repaint_t0 || repaint_crossfade_frames <= 0) return;

    const int n_per = T * Oc;
    int cf         = repaint_crossfade_frames;
    int fade_start = repaint_t0 - cf > 0 ? repaint_t0 - cf : 0;
    int fade_end   = repaint_t1 + cf < T ? repaint_t1 + cf : T;

    for (int t = fade_start; t < fade_end; t++) {
        if (t >= repaint_t0 && t < repaint_t1) {
            continue;  // inside zone: keep generated output unchanged
        }
        float m;
        if (t < repaint_t0) {
            // left ramp: [fade_start, t0) -> 0..1 excluding endpoints
            int rl = repaint_t0 - fade_start;
            m      = (float) (t - fade_start + 1) / (float) (rl + 1);
        } else {
            // right ramp: [t1, fade_end) -> 1..0 excluding endpoints
            int rl = fade_end - repaint_t1;
            m      = (float) (fade_end - t) / (float) (rl + 1);
        }
        for (int b = 0; b < N; b++) {
            for (int ch = 0; ch < Oc; ch++) {
                int idx     = b * n_per + t * Oc + ch;
                output[idx] = m * output[idx] + (1.0f - m) * repaint_src[t * Oc + ch];
            }
        }
    }
}
