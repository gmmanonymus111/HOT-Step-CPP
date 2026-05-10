-- aflops.lua: A-FloPS (1 NFE, stateful multistep)
-- Adaptive Flow Path Sampler with velocity decomposition.

solver = {
    name        = "aflops",
    display     = "A-FloPS (1 NFE)",
    description = "Exponential integrator with residual tracking",
    nfe         = 1,
    order       = 2,
    needs_model = false,
    stateful    = true,
    stochastic  = false,
}

local prev_w = nil
local prev_t = 0
local prev_t_dst = 0

local function clamp_alpha(t)
    local a = 1 - t
    return math.max(1e-6, math.min(a, 1 - 1e-6))
end

function step(xt, vt, t_curr, t_prev, n)
    if (step_index or 0) == 0 then prev_w = nil; prev_t = 0; prev_t_dst = 0 end
    local alpha_curr = clamp_alpha(t_curr)
    local alpha_prev = clamp_alpha(t_prev)
    local alpha_ratio = alpha_prev / alpha_curr
    local log_ratio = math.log(alpha_ratio)

    -- Compute residual velocity: w = v + x/(1-t)
    local w = {}
    local inv_alpha = 1 / alpha_curr
    for i = 0, n-1 do w[i] = vt[i] + xt[i] * inv_alpha end

    if prev_w then
        -- 2nd order: AB-like correction
        local dt_curr = t_curr - t_prev
        local dt_prev = prev_t - prev_t_dst
        local r = (dt_prev > 1e-8) and (dt_curr / dt_prev) or 1
        local c1 = 1 + 0.5 * r
        local c0 = 0.5 * r
        for i = 0, n-1 do
            local w_eff = c1 * w[i] - c0 * prev_w[i]
            xt[i] = alpha_ratio * xt[i] - alpha_prev * w_eff * log_ratio
        end
    else
        -- 1st order: exponential Euler
        for i = 0, n-1 do
            xt[i] = alpha_ratio * xt[i] - alpha_prev * w[i] * log_ratio
        end
    end

    prev_w = w
    prev_t = t_curr
    prev_t_dst = t_prev
end
