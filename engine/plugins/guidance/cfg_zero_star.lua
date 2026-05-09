-- cfg_zero_star.lua: CFG-Zero⋆ — Zero-Init Guidance
-- Paper: "CFG-Zero⋆: Improved Classifier-Free Guidance for Flow Matching Models"
--        Fan et al., 2025 (arXiv:2503.18886)
--
-- The paper proposes two improvements: optimised scale (s⋆) and zero-init.
-- Since our engine uses APG (perpendicular projection + momentum), which
-- already corrects for the underfitting that s⋆ addresses, only zero-init
-- is applied here. Combining both would double-correct.
--
-- Zero-init: zeroes out velocity for the first N ODE steps, since early-step
-- CFG predictions in flow matching are often worse than doing nothing.
-- All subsequent steps use the standard APG pipeline.

guidance = {
    name        = "cfg_zero_star",
    display     = "CFG-Zero⋆",
    description = "Zero-init + APG guidance (Fan et al. 2025)",
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

    -- Standard APG for all other steps
    apg(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
end
