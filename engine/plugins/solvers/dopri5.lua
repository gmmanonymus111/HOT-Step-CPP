-- dopri5.lua: Dormand-Prince 5(4) adaptive solver (7+ NFE)
-- Adaptive sub-stepping with error estimation for optimal accuracy.

solver = {
    name        = "dopri5",
    display     = "DOPRI5 (7+ NFE)",
    description = "Adaptive Dormand-Prince 5th order with error control",
    nfe         = 0, -- variable
    order       = 5,
    needs_model = true,
    stateful    = false,
    stochastic  = false,
}

-- Butcher tableau constants
local C = {1/5, 3/10, 4/5, 8/9, 1, 1}
local A = {
    {1/5},
    {3/40, 9/40},
    {44/45, -56/15, 32/9},
    {19372/6561, -25360/2187, 64448/6561, -212/729},
    {9017/3168, -355/33, 46732/5247, 49/176, -5103/18656},
    {35/384, 0, 500/1113, 125/192, -2187/6784, 11/84},
}
local B = {35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0}
local E = {
    35/384 - 1951/21600,
    0,
    500/1113 - 22642/50085,
    125/192 - 451/720,
    -2187/6784 + 12231/42400,
    11/84 - 649/6300,
    -1/60,
}

-- Generic ERK step: compute all stages, return result in xt_out
-- xt_fa is a FloatArray used as scratch space for model_fn calls
local function erk_step(x, k1, t, h, n_elem, model_fn, vt_buf, xt_fa, num_extra, a_rows, c_vals, b_vals)
    local ks = {k1}
    for s = 1, num_extra do
        local a_row = a_rows[s]
        for i = 0, n_elem-1 do
            local combo = 0
            for j = 1, #a_row do
                if a_row[j] ~= 0 then combo = combo + a_row[j] * ks[j][i] end
            end
            xt_fa[i] = x[i] - h * combo
        end
        -- model_fn expects a FloatArray, writes result into vt_buf
        model_fn(xt_fa, t - c_vals[s] * h)
        ks[s+1] = {}
        for i = 0, n_elem-1 do ks[s+1][i] = vt_buf[i] end
    end

    local result = {}
    for i = 0, n_elem-1 do
        local sol = 0
        for j = 1, #b_vals do
            if b_vals[j] ~= 0 and ks[j] then sol = sol + b_vals[j] * ks[j][i] end
        end
        result[i] = x[i] - h * sol
    end
    return result, ks
end

function step(xt, vt, t_curr, t_prev, n, model_fn, vt_buf)
    local atol = 1e-3
    local rtol = 1e-2
    local max_sub = 8
    local safety = 0.9

    local t = t_curr
    local t_end = t_prev
    local h = t - t_end

    -- Working copy (Lua tables for intermediate math)
    local x_cur = {}
    local v_cur = {}
    for i = 0, n-1 do x_cur[i] = xt[i]; v_cur[i] = vt[i] end

    local sub = 0
    while sub < max_sub and (t - t_end) > 1e-10 do
        h = math.min(h, t - t_end)
        local k1 = v_cur

        -- Full DOPRI5 step (6 extra stages for 7 total including FSAL)
        -- Pass xt as scratch FloatArray for model_fn calls
        local x_next, ks = erk_step(x_cur, k1, t, h, n, model_fn, vt_buf, xt, 6, A, C, B)

        -- Error estimate
        local err_sq_sum = 0
        for i = 0, n-1 do
            local err_i = 0
            for j = 1, 7 do
                if E[j] ~= 0 and ks[j] then err_i = err_i + E[j] * ks[j][i] end
            end
            err_i = err_i * h
            local scale = atol + rtol * math.max(math.abs(x_cur[i]), math.abs(x_next[i]))
            local ratio = err_i / scale
            err_sq_sum = err_sq_sum + ratio * ratio
        end
        local err_norm = math.sqrt(err_sq_sum / n)

        if err_norm <= 1 then
            t = t - h
            x_cur = x_next
            v_cur = ks[7] -- FSAL
            if err_norm > 1e-10 then
                h = h * math.min(5, safety * err_norm ^ (-0.2))
            else
                h = h * 5
            end
        else
            h = h * math.max(0.2, safety * err_norm ^ (-0.2))
        end
        sub = sub + 1
    end

    for i = 0, n-1 do xt[i] = x_cur[i] end
end
