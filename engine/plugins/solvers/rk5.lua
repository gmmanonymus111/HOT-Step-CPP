-- rk5.lua: 5th-order Runge-Kutta (Cash-Karp, 6 NFE per step)

solver = {
    name        = "rk5",
    display     = "RK5 (6 NFE)",
    description = "5th-order Cash-Karp Runge-Kutta",
    nfe         = 6,
    order       = 5,
    needs_model = true,
    stateful    = false,
    stochastic  = false,
}

-- Cash-Karp Butcher tableau
local a2 = 1/5
local a3 = 3/10
local a4 = 3/5
local a5 = 1
local a6 = 7/8

local b21 = 1/5
local b31 = 3/40;   local b32 = 9/40
local b41 = 3/10;   local b42 = -9/10;  local b43 = 6/5
local b51 = -11/54; local b52 = 5/2;    local b53 = -70/27; local b54 = 35/27
local b61 = 1631/55296; local b62 = 175/512; local b63 = 575/13824; local b64 = 44275/110592; local b65 = 253/4096

-- 5th order weights
local c1 = 37/378; local c3 = 250/621; local c4 = 125/594; local c6 = 512/1771

function step(xt, vt, t_curr, t_prev, n, model_fn, vt_buf)
    local dt = t_curr - t_prev
    local xt_orig = {}
    local k1 = {}
    for i = 0, n-1 do xt_orig[i] = xt[i]; k1[i] = vt[i] end

    -- k2
    for i = 0, n-1 do xt[i] = xt_orig[i] - dt * b21 * k1[i] end
    model_fn(xt, t_curr - a2 * dt)
    local k2 = {}; for i = 0, n-1 do k2[i] = vt_buf[i] end

    -- k3
    for i = 0, n-1 do xt[i] = xt_orig[i] - dt * (b31*k1[i] + b32*k2[i]) end
    model_fn(xt, t_curr - a3 * dt)
    local k3 = {}; for i = 0, n-1 do k3[i] = vt_buf[i] end

    -- k4
    for i = 0, n-1 do xt[i] = xt_orig[i] - dt * (b41*k1[i] + b42*k2[i] + b43*k3[i]) end
    model_fn(xt, t_curr - a4 * dt)
    local k4 = {}; for i = 0, n-1 do k4[i] = vt_buf[i] end

    -- k5
    for i = 0, n-1 do xt[i] = xt_orig[i] - dt * (b51*k1[i] + b52*k2[i] + b53*k3[i] + b54*k4[i]) end
    model_fn(xt, t_curr - a5 * dt)
    local k5 = {}; for i = 0, n-1 do k5[i] = vt_buf[i] end

    -- k6
    for i = 0, n-1 do xt[i] = xt_orig[i] - dt * (b61*k1[i] + b62*k2[i] + b63*k3[i] + b64*k4[i] + b65*k5[i]) end
    model_fn(xt, t_curr - a6 * dt)

    -- Final 5th-order result
    for i = 0, n-1 do
        xt[i] = xt_orig[i] - dt * (c1*k1[i] + c3*k3[i] + c4*k4[i] + c6*vt_buf[i])
    end
end
