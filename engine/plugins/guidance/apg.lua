-- apg.lua: Adaptive Perpendicular Guidance (default)
-- Routes through apg_forward C helper for momentum + perpendicular projection.
-- This plugin declares the params; the actual APG math is in the C++ helper
-- because it's shared infrastructure used by all guidance modes.

guidance = {
    name        = "apg",
    display     = "APG",
    description = "Adaptive perpendicular guidance (default)",
    params      = {
        { key = "momentum", type = "slider", label = "Momentum",
          default = 0.75, min = -1, max = 1, step = 0.01,
          hint = "APG momentum coefficient (negative = adaptive)" },
        { key = "norm_threshold", type = "slider", label = "Norm Threshold",
          default = 2.5, min = 0, max = 10, step = 0.1,
          hint = "APG norm clipping threshold" },
    },
}

-- guide() receives: pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold
-- The C++ wrapper calls apg_forward for us since APG core is shared C++ infrastructure.
-- This Lua function is the fallback / reference implementation.
function guide(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
    local n = Oc * T
    -- Simple CFG: result = pred_uncond + scale * (pred_cond - pred_uncond)
    -- Full APG with momentum is handled by the C++ wrapper
    for i = 0, n - 1 do
        result[i] = pred_uncond[i] + guidance_scale * (pred_cond[i] - pred_uncond[i])
    end
end
