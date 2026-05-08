-- aflops2.lua: A-FloPS Midpoint (2 NFE, stateless)
-- Midpoint-corrected exponential integrator.

solver = {
    name        = "aflops2",
    display     = "A-FloPS Midpoint (2 NFE)",
    description = "Midpoint-corrected exponential integrator",
    nfe         = 2,
    order       = 2,
    needs_model = true,
    stateful    = false,
    stochastic  = false,
}

local function clamp_alpha(t)
    local a = 1 - t
    return math.max(1e-6, math.min(a, 1 - 1e-6))
end

function step(xt, vt, t_curr, t_prev, n, model_fn, vt_buf)
    local dt = t_curr - t_prev

    if t_curr < 1e-8 then
        for i = 0, n-1 do xt[i] = xt[i] - vt[i] * dt end
        return
    end

    local alpha_curr = clamp_alpha(t_curr)
    local alpha_prev = clamp_alpha(t_prev)

    -- Save v_curr
    local v_curr = {}
    for i = 0, n-1 do v_curr[i] = vt[i] end

    -- Euler half-step to midpoint
    local half_dt = dt * 0.5
    local t_mid = t_curr - half_dt
    local alpha_mid = clamp_alpha(t_mid)

    local x_mid = {}
    for i = 0, n-1 do x_mid[i] = xt[i] - v_curr[i] * half_dt end

    -- Evaluate at midpoint
    model_fn(xt, t_mid) -- xt used as scratch, but we need x_mid...
    -- Actually we need to pass x_mid to model_fn. Fix:
    -- Store xt, use x_mid for model_fn
    local xt_save = {}
    for i = 0, n-1 do xt_save[i] = xt[i]; xt[i] = x_mid[i] end
    model_fn(xt, t_mid)

    -- Compute w_mid from midpoint
    local inv_alpha_mid = 1 / alpha_mid
    local w_mid = {}
    for i = 0, n-1 do
        w_mid[i] = vt_buf[i] + x_mid[i] * inv_alpha_mid
    end

    -- Full step using midpoint residual
    local alpha_ratio = alpha_prev / alpha_curr
    local log_ratio = math.log(alpha_ratio)

    for i = 0, n-1 do
        xt[i] = alpha_ratio * xt_save[i] - alpha_prev * w_mid[i] * log_ratio
    end
end
