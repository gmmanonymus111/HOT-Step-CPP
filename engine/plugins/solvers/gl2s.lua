-- gl2s.lua: Gauss-Legendre 2-stage implicit Runge-Kutta (4th order, 6 NFE)
-- A-stable, symplectic. Fixed-point iteration solves the implicit system.

solver = {
    name        = "gl2s",
    display     = "Gauss-Legendre 2s (6 NFE)",
    description = "Implicit 4th-order A-stable symplectic integrator",
    nfe         = 6,
    order       = 4,
    needs_model = true,
    stateful    = false,
    stochastic  = false,
}

-- Butcher tableau
local SQRT3_6 = 0.28867513459481287
local C1  = 0.5 - SQRT3_6  -- ≈ 0.2113
local C2  = 0.5 + SQRT3_6  -- ≈ 0.7887
local A11 = 0.25
local A12 = 0.25 - SQRT3_6  -- ≈ -0.0387
local A21 = 0.25 + SQRT3_6  -- ≈ 0.5387
local A22 = 0.25
local ITERATIONS = 3

function step(xt, vt, t_curr, t_prev, n, model_fn, vt_buf)
    local dt = t_curr - t_prev

    -- Initialize k1 = k2 = vt
    local k1 = {}
    local k2 = {}
    for i = 0, n-1 do k1[i] = vt[i]; k2[i] = vt[i] end

    local xt_orig = {}
    for i = 0, n-1 do xt_orig[i] = xt[i] end

    local t1 = t_curr - C1 * dt
    local t2 = t_curr - C2 * dt

    -- Fixed-point iteration
    for iter = 1, ITERATIONS do
        -- Stage 1: x1 = xt - dt*(A11*k1 + A12*k2)
        for i = 0, n-1 do
            xt[i] = xt_orig[i] - dt * (A11 * k1[i] + A12 * k2[i])
        end
        model_fn(xt, t1)
        for i = 0, n-1 do k1[i] = vt_buf[i] end

        -- Stage 2: x2 = xt - dt*(A21*k1 + A22*k2)
        for i = 0, n-1 do
            xt[i] = xt_orig[i] - dt * (A21 * k1[i] + A22 * k2[i])
        end
        model_fn(xt, t2)
        for i = 0, n-1 do k2[i] = vt_buf[i] end
    end

    -- Final: xt = xt_orig - dt * 0.5 * (k1 + k2)
    for i = 0, n-1 do
        xt[i] = xt_orig[i] - dt * 0.5 * (k1[i] + k2[i])
    end
end
