-- stork4.lua: STORK 4 — Stabilized Taylor Orthogonal RK (4th order, 1 NFE)
-- Uses ROCK4 Chebyshev sub-stepping with precomputed coefficients.
-- Requires companion stork4_constants.lua.

local C = require("stork4_constants")

solver = {
    name        = "stork4",
    display     = "STORK 4",
    description = "4th-order stabilized Taylor-ROCK4 (1 NFE)",
    nfe         = 1,
    order       = 4,
    needs_model = false,
    stateful    = true,
    stochastic  = false,
    params      = {
        { key = "substeps", type = "slider", label = "Substeps",
          default = 10, min = 2, max = 50, step = 1,
          hint = "Number of ROCK4 sub-steps (more = more stable)" },
    },
}

local velocity_history = {}

local function rms(data, n)
    local sum = 0
    for i = 0, n-1 do sum = sum + data[i] * data[i] end
    return math.sqrt(sum / n)
end

local function has_nan_inf(data, n)
    for i = 0, n-1 do
        local v = data[i]
        if v ~= v or v == math.huge or v == -math.huge then return true end
    end
    return false
end

local function compute_derivatives(vt, n)
    local hist = velocity_history
    if #hist == 0 then return 0, nil, nil end
    local v_prev = hist[#hist].vt
    local h1 = hist[#hist].dt
    if h1 == 0 then return 0, nil, nil end
    local dv = {}
    for i = 0, n-1 do dv[i] = (v_prev[i] - vt[i]) / h1 end
    local vt_rms = rms(vt, n)
    local dv_rms = rms(dv, n)
    if vt_rms > 0 and dv_rms * math.abs(h1) > 5 * vt_rms then return 0, nil, nil end
    if #hist < 2 then return 1, dv, nil end
    local v_prev2 = hist[#hist - 1].vt
    local h2 = hist[#hist - 1].dt
    if h2 == 0 then return 1, dv, nil end
    local denom = h1 * h2 * (h1 + h2)
    if math.abs(denom) < 1e-30 then return 1, dv, nil end
    local coeff = 2 / denom
    local d2v = {}
    for i = 0, n-1 do
        d2v[i] = coeff * (v_prev2[i] * h1 - v_prev[i] * (h1 + h2) + vt[i] * h2)
    end
    local d2v_rms = rms(d2v, n)
    if vt_rms > 0 and d2v_rms * h1 * h1 > 5 * vt_rms then return 1, dv, nil end
    return 2, dv, d2v
end

local function taylor_approx(vt, order, diff, dv, d2v, n)
    local out = {}
    if order >= 2 and d2v then
        local half_d2 = 0.5 * diff * diff
        for i = 0, n-1 do out[i] = vt[i] + diff * dv[i] + half_d2 * d2v[i] end
    elseif order >= 1 and dv then
        for i = 0, n-1 do out[i] = vt[i] + diff * dv[i] end
    else
        for i = 0, n-1 do out[i] = vt[i] end
    end
    return out
end

local function rock4_mdegr(s)
    local mp1 = 1
    for i = 1, C.MS_LEN do
        if C.MS[i] >= s then
            return C.MS[i], i, mp1 - 1
        end
        mp1 = mp1 + C.MS[i] * 2 - 1
    end
    return C.MS[C.MS_LEN], C.MS_LEN, mp1 - 1
end

local function rock4_substep(xt, vt, s, t_curr, t_prev, deriv_order, dv, d2v, n)
    local max_rock4 = C.MS[C.MS_LEN]
    if s > max_rock4 then s = max_rock4 end
    local mdeg, mz, mr = rock4_mdegr(s)
    local dt = t_curr - t_prev

    local Y_j_2 = {}; local Y_j_1 = {}; local Y_j = {}
    for i = 0, n-1 do Y_j_2[i] = xt[i]; Y_j_1[i] = xt[i] end

    local ci1 = t_curr

    -- ROCK4 Chebyshev recurrence (1-indexed RECF)
    for j = 1, mdeg do
        if j == 1 then
            local temp1 = -dt * C.RECF[mr + 1]
            ci1 = t_curr + temp1
            for i = 0, n-1 do Y_j_1[i] = xt[i] + temp1 * vt[i] end
        else
            local diff = ci1 - t_curr
            local vel = taylor_approx(vt, deriv_order, diff, dv, d2v, n)
            local idx1 = mr + 2 * (j - 2) + 2
            local idx2 = mr + 2 * (j - 2) + 3
            local temp1 = -dt * C.RECF[idx1]
            local temp3 = -C.RECF[idx2]
            local temp2 = 1 - temp3
            for i = 0, n-1 do
                Y_j[i] = temp1 * vel[i] + temp2 * Y_j_1[i] + temp3 * Y_j_2[i]
            end
            for i = 0, n-1 do Y_j_2[i] = Y_j_1[i]; Y_j_1[i] = Y_j[i] end
            ci1 = temp1 + temp2 * ci1 + temp3 * ci1 -- simplified
        end
    end

    -- ROCK4 finishing procedure (4 stages)
    local Y_base = Y_j_1
    local fpa = C.FPA[mz]; local fpb = C.FPB[mz]

    -- F1
    local diff1 = ci1 - t_curr
    local F1 = taylor_approx(vt, deriv_order, diff1, dv, d2v, n)
    local fpa0 = -dt * fpa[1]
    local Yf = {}
    for i = 0, n-1 do Yf[i] = Y_base[i] + fpa0 * F1[i] end

    -- F2
    local diff2 = ci1 + fpa0 - t_curr
    local F2 = taylor_approx(vt, deriv_order, diff2, dv, d2v, n)
    local fpa1 = -dt * fpa[2]; local fpa2 = -dt * fpa[3]
    for i = 0, n-1 do Yf[i] = Y_base[i] + fpa1 * F1[i] + fpa2 * F2[i] end

    -- F3
    local diff3 = ci1 + fpa1 + fpa2 - t_curr
    local F3 = taylor_approx(vt, deriv_order, diff3, dv, d2v, n)
    local fpa3 = -dt * fpa[4]; local fpa4 = -dt * fpa[5]; local fpa5 = -dt * fpa[6]

    -- F4
    local diff4 = ci1 + fpa3 + fpa4 + fpa5 - t_curr
    local F4 = taylor_approx(vt, deriv_order, diff4, dv, d2v, n)
    local fpb0 = -dt * fpb[1]; local fpb1 = -dt * fpb[2]; local fpb2 = -dt * fpb[3]; local fpb3 = -dt * fpb[4]

    local result = {}
    for i = 0, n-1 do
        result[i] = Y_base[i] + fpb0*F1[i] + fpb1*F2[i] + fpb2*F3[i] + fpb3*F4[i]
    end
    return result, not has_nan_inf(result, n)
end

local function update_history(vt, n, dt)
    local rec = {vt = {}, dt = dt}
    for i = 0, n-1 do rec.vt[i] = vt[i] end
    table.insert(velocity_history, rec)
    while #velocity_history > 3 do table.remove(velocity_history, 1) end
end

function step(xt, vt, t_curr, t_prev, n)
    local dt = t_curr - t_prev
    if step_index == 0 then
        velocity_history = {}
        for i = 0, n-1 do xt[i] = xt[i] - vt[i] * dt end
        update_history(vt, n, dt)
        return
    end
    local deriv_order, dv, d2v = compute_derivatives(vt, n)
    local s = math.max((params and params.substeps) or 10, 2)
    local success = false; local result
    while s >= 2 do
        result, success = rock4_substep(xt, vt, s, t_curr, t_prev, deriv_order, dv, d2v, n)
        if success then break end
        s = math.floor(s / 2)
    end
    if success then
        for i = 0, n-1 do xt[i] = result[i] end
    else
        for i = 0, n-1 do xt[i] = xt[i] - vt[i] * dt end
    end
    update_history(vt, n, dt)
end
