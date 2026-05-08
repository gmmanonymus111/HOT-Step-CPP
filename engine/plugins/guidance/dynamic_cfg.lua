-- dynamic_cfg.lua: Cosine-decaying guidance schedule
-- Full guidance early (structure), reduced guidance late (fine detail).

guidance = {
    name        = "dynamic_cfg",
    display     = "Dynamic CFG",
    description = "Cosine-decaying guidance schedule",
}

function guide(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
    local n = Oc * T
    local power = 0.5
    local progress = (step_idx or 0) / math.max((total_steps or 1) - 1, 1)
    local cos_val = math.max(math.cos(math.pi / 2 * progress), 0)
    local decay = cos_val ^ power
    local effective_scale = 1.0 + (guidance_scale - 1.0) * decay

    for i = 0, n - 1 do
        result[i] = pred_uncond[i] + effective_scale * (pred_cond[i] - pred_uncond[i])
    end
end
