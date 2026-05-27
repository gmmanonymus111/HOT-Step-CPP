-- ============================================================================
-- SPDX-License-Identifier: GPL-3.0-or-later
-- Copyright (C) 2026 Alexander Allan (MDMAchine) -- A&E Concepts
--
-- This program is free software: you can redistribute it and/or modify
-- it under the terms of the GNU General Public License as published by
-- the Free Software Foundation, either version 3 of the License, or
-- (at your option) any later version.
--
-- This program is distributed in the hope that it will be useful,
-- but WITHOUT ANY WARRANTY; without even the implied warranty of
-- MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
-- GNU General Public License for more details: https://www.gnu.org/licenses/
-- ============================================================================

-- MD STORM Guidance v2.0
-- Sigma-Aware CFG Adaptation + Normalized Attention Guidance (NAG)
-- MDMAchine | A&E Concepts © 2026
--
-- Optimized: Strict Boundary State Resets & Engine Variable Safety
--
-- ── SYSTEM 1: SIGMA-AWARE CFG (guide()) ─────────────────────────────────────
--
--   Flat at full guidance_scale from step 0 until knee fraction of schedule.
--   Past knee: tail rolloff via power curve down to floor.
--   Generation boundary reset captures true sigma_max every new run.
--
-- ── SYSTEM 2: NAG POST-STEP SUPPRESSION (post_step()) ───────────────────────
--
--   Soft-clamps latent elements spiking above threshold relative to mean.
--   Suppresses progressive attention amplification / harmonic hum in LoRA.
--   step_index / step_idx dual-check prevents static dither mask.
--
-- ── RECOMMENDED SETTINGS FOR STORM + ACE-STEP LORA ─────────────────────────
--
--   CFG knee:       0.65  (rolloff starts in last quarter of schedule)
--   CFG tail_power: 2.5   (quadratic rolloff)
--   CFG floor:      0.60  (never drop below 60% of guidance_scale)
--   NAG clamp:      0.20  (standard LoRA hum suppression)
--   NAG threshold:  0.75  (fires on elements >x mean magnitude)
--   NAG dither:     true  (always on for audio)
--
-- ============================================================================

guidance = {
    name        = "md_storm_guidance",
    display     = "MD STORM Guidance V2",
    description = "Companion guidance plugin for STORM. (1) Sigma-aware CFG adaptation — flat at full scale until knee, then tail rolloff to prevent late-step over-sharpening and harmonic ringing. (2) NAG post-step latent suppression — soft-clamps spiking elements to suppress resonance buildup in LoRA inference. Generation-boundary safe.",
    params      = {
        -- ── CFG Adaptation ──────────────────────────────────────────────────
        { key = "cfg_adapt_enabled",  type = "toggle", label = "Sigma CFG Adaptation", default = true,  hint = "Bleed guidance scale as sigma drops. Prevents harmonic ringing and over-sharpening at late steps." },
        { key = "cfg_knee",           type = "slider", label = "Rolloff Knee",          default = 0.65,  min = 0.1, max = 0.9,  step = 0.05,  hint = "Progress fraction where rolloff begins. 0.75 = last quarter of steps. Higher = later, gentler rolloff." },
        { key = "cfg_tail_power",     type = "slider", label = "Rolloff Shape",         default = 2.5,   min = 0.5, max = 5.0,  step = 0.25,  hint = "Rolloff curve power. 1.0=linear, 2.0=quadratic (smooth), 4.0=aggressive late cliff." },
        { key = "cfg_floor",          type = "slider", label = "Guidance Floor",        default = 0.60,  min = 0.0, max = 0.95, step = 0.05,  hint = "Minimum guidance scale as fraction of set guidance_scale. 0.70 = never drop below 70%. Raise if detail collapses." },

        -- ── NAG Suppression ─────────────────────────────────────────────────
        { key = "nag_enabled",        type = "toggle", label = "NAG Suppression",       default = true,  hint = "Suppress progressive attention amplification / harmonic resonance buildup in LoRA inference. Applied per step to xt." },
        { key = "nag_clamp_intensity",type = "slider", label = "NAG Clamp Intensity",   default = 0.20,  min = 0.0, max = 1.0,  step = 0.01,  hint = "Blend strength toward normalized value for spiking elements. 0.15-0.25 recommended for audio LoRA." },
        { key = "nag_spike_threshold",type = "slider", label = "NAG Spike Threshold",   default = 0.75,  min = 0.3, max = 0.99, step = 0.01,  hint = "Relative magnitude above which suppression fires. 0.85 maps to ~6.7x mean. Lower = more aggressive." },
        { key = "nag_dither",         type = "toggle", label = "NAG Seed Dithering",    default = true,  hint = "Randomize spike mask edges to prevent sharp structural breaks at clamp boundary. Always recommended for audio." },
        { key = "nag_dither_strength",type = "slider", label = "Dither Strength",       default = 0.1,  min = 0.0, max = 0.20, step = 0.01,  hint = "Dither amplitude. 0.05 = standard. Softens mask edges without losing suppression effect." },
        { key = "nag_sigma_gate",     type = "slider", label = "NAG Sigma Gate",        default = 0.0,   min = 0.0, max = 0.8,  step = 0.05,  hint = "Only apply NAG when sigma is below this value. 0.0 = always active. 0.5 = refinement zone only." },
    },
}

-- ── Constants ────────────────────────────────────────────────────────────────

local EPSILON  = 1e-6
local _last_n  = 0
local _sigma_max = 1.0
local _sigma_min = 0.0

-- ── Helpers ──────────────────────────────────────────────────────────────────

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

-- Seeded LCG (deterministic, no math.random dependency)
local function make_rng(seed)
    local state = math.floor(seed) % 2147483647
    if state <= 0 then state = state + 2147483646 end
    return function()
        state = (state * 1664525 + 1013904223) % 2147483648
        return state / 2147483648.0
    end
end

-- ── CFG Envelope ─────────────────────────────────────────────────────────────

local function compute_cfg_envelope(sigma_curr, sigma_max, sigma_min, knee, tail_power, floor_)
    local range    = sigma_max - sigma_min + EPSILON
    local progress = clamp((sigma_max - sigma_curr) / range, 0.0, 1.0)

    -- Flat at 1.0 until knee, tail rolloff only past knee
    local envelope = 1.0
    if progress > knee then
        local tail_progress = clamp((progress - knee) / math.max(1.0 - knee, EPSILON), 0.0, 1.0)
        envelope = 1.0 - tail_progress ^ tail_power
    end

    return floor_ + (1.0 - floor_) * clamp(envelope, 0.0, 1.0)
end

-- ── NAG Suppression (in-place on FloatArray) ─────────────────────────────────

local function apply_nag_inplace(xt, n, clamp_int, spike_thr, dither, dither_str, seed)
    local mean_norm = 0.0
    for i = 0, n - 1 do mean_norm = mean_norm + math.abs(xt[i]) end
    mean_norm = (mean_norm / n) + EPSILON

    local thr = 1.0 / (1.0 - spike_thr + EPSILON)
    local rng = dither and make_rng(seed) or nil

    for i = 0, n - 1 do
        local rel_mag = math.abs(xt[i]) / mean_norm
        local spike   = (rel_mag > thr) and 1.0 or 0.0

        if dither and rng ~= nil then
            spike = clamp(spike - rng() * dither_str, 0.0, 1.0)
        end

        -- Skip blend entirely on non-spiking elements
        if spike > 0.0 then
            local sign        = (xt[i] >= 0.0) and 1.0 or -1.0
            local norm_target = sign * mean_norm
            local blend       = spike * clamp_int
            xt[i] = xt[i] * (1.0 - blend) + norm_target * blend
        end
    end
end

-- ── guide() — sigma-aware CFG ────────────────────────────────────────────────

function guide(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
    local n        = Oc * T
    local cur_step = step_index or step_idx or 0
    local sigma    = t_curr or 0.5

    -- Strict generation boundary reset: captures true sigma_max every new run.
    -- Fires on step 0 OR if latent size changed (model switch).
    if n ~= _last_n or cur_step == 0 then
        _last_n    = n
        _sigma_max = sigma
        _sigma_min = 0.0
    end

    local cfg_on     = (params and params.cfg_adapt_enabled)
    if cfg_on == nil then cfg_on = true end
    local knee       = (params and params.cfg_knee)       or 0.75
    local tail_power = (params and params.cfg_tail_power) or 2.0
    local floor_     = (params and params.cfg_floor)      or 0.70

    local effective_scale = guidance_scale

    if cfg_on then
        local envelope = compute_cfg_envelope(sigma, _sigma_max, _sigma_min, knee, tail_power, floor_)
        effective_scale = guidance_scale * envelope
    end

    apg(pred_cond, pred_uncond, effective_scale, result, Oc, T, norm_threshold)
end

-- ── post_step() — NAG suppression ────────────────────────────────────────────

function post_step(xt, t, n, eval_cond, eval_uncond, vt_cond, vt_uncond)
    local nag_on = (params and params.nag_enabled)
    if nag_on == nil then nag_on = true end
    if not nag_on then return end

    local clamp_int  = (params and params.nag_clamp_intensity) or 0.20
    local spike_thr  = (params and params.nag_spike_threshold) or 0.85
    local dither     = (params and params.nag_dither)
    if dither == nil then dither = true end
    local dither_str = (params and params.nag_dither_strength) or 0.05
    local sigma_gate = (params and params.nag_sigma_gate)      or 0.0

    local sigma    = t or 0.0
    local cur_step = step_index or step_idx or 0

    if sigma_gate > 0.0 and sigma > sigma_gate then return end

    -- step-varying seed: prevents identical dither mask every step
    local seed = math.floor(42 + cur_step * 7919)

    apply_nag_inplace(xt, n, clamp_int, spike_thr, dither, dither_str, seed)
end
