-- cfg_zero_star.lua: CFG-Zero⋆ — Optimized Scale + Zero-Init
-- Paper: "CFG-Zero⋆: Improved Classifier-Free Guidance for Flow Matching Models"
--        Fan et al., 2025 (arXiv:2503.18886)
--
-- Two key improvements over vanilla CFG:
--   1. Optimized scale (s⋆): projects conditional velocity onto unconditional,
--      compensating for underfitting in the learned velocity field.
--   2. Zero-init: zeroes out the velocity for the first N steps of the ODE solver,
--      since early-step CFG predictions are often worse than doing nothing.
--
-- Formula (for steps > zero_init_steps):
--   s⋆ = dot(v_cond, v_uncond) / ||v_uncond||²
--   v_guided = s⋆·v_uncond + w·(v_cond - s⋆·v_uncond)

guidance = {
    name        = "cfg_zero_star",
    display     = "CFG-Zero⋆",
    description = "Optimized scale + zero-init (Fan et al. 2025)",
    params      = {
        { key = "zero_init_steps", type = "slider", label = "Zero-Init Steps",
          default = 1, min = 0, max = 5, step = 1,
          hint = "Number of initial ODE steps to zero out (paper recommends 1)" },
    },
}

function guide(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
    local n = Oc * T
    local zero_init_steps = (params and params.zero_init_steps) or 1

    -- Zero-init: zero out velocity for the first N steps
    if (step_idx or 0) < zero_init_steps then
        for i = 0, n - 1 do
            result[i] = 0.0
        end
        return
    end

    -- Compute optimized scale s⋆ = dot(cond, uncond) / ||uncond||²
    local dot_product = 0.0
    local squared_norm = 0.0
    for i = 0, n - 1 do
        dot_product   = dot_product   + pred_cond[i] * pred_uncond[i]
        squared_norm  = squared_norm  + pred_uncond[i] * pred_uncond[i]
    end
    local st_star = dot_product / (squared_norm + 1e-8)

    -- v_guided = st_star * v_uncond + w * (v_cond - st_star * v_uncond)
    for i = 0, n - 1 do
        local scaled_uncond = st_star * pred_uncond[i]
        result[i] = scaled_uncond + guidance_scale * (pred_cond[i] - scaled_uncond)
    end
end
