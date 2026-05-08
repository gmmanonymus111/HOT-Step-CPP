-- rfsolver.lua: RF-Solver (2 NFE) — Rectified Flow specific
-- Exploits the RF ODE structure for higher accuracy than generic midpoint.

solver = {
    name        = "rfsolver",
    display     = "RF-Solver (2 NFE)",
    description = "Rectified-flow-aware midpoint solver",
    nfe         = 2,
    order       = 2,
    needs_model = true,
    stateful    = false,
    stochastic  = false,
}

function step(xt, vt, t_curr, t_prev, n, model_fn, vt_buf)
    local dt = t_curr - t_prev

    if t_curr < 1e-8 then
        for i = 0, n-1 do xt[i] = xt[i] - vt[i] * dt end
        return
    end

    -- Save v_t
    local v_t = {}
    for i = 0, n-1 do v_t[i] = vt[i] end

    -- Half-step to midpoint
    local half_dt = dt * 0.5
    local t_mid = t_curr - half_dt
    for i = 0, n-1 do xt[i] = xt[i] - v_t[i] * half_dt end

    -- Evaluate at midpoint
    model_fn(xt, t_mid)

    -- RF-specific: reconstruct via x_0 prediction from midpoint
    for i = 0, n-1 do
        local x_0_mid = xt[i] - t_mid * vt_buf[i]
        xt[i] = x_0_mid + t_prev * vt_buf[i]
    end
end
