-- stork2.lua: STORK 2 — Stabilized Taylor Orthogonal RK (2nd order, 1 NFE)
-- Uses RKG2 Chebyshev sub-stepping with velocity derivatives from history.

solver = {
    name        = "stork2",
    display     = "STORK 2",
    description = "2nd-order stabilized Taylor-Chebyshev (1 NFE)",
    nfe         = 1,
    order       = 2,
    needs_model = false,
    stateful    = true,
    stochastic  = false,
    params      = {
        { key = "substeps", type = "slider", label = "Substeps",
          default = 10, min = 2, max = 50, step = 1,
          hint = "Number of Chebyshev sub-steps (more = more stable)" },
    },
}

local velocity_history = {}  -- {vt={}, dt=float}

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

local function taylor_approx(vt, deriv_order, diff, dv, d2v, n)
    local out = {}
    if deriv_order >= 2 and d2v then
        local half_d2 = 0.5 * diff * diff
        for i = 0, n-1 do out[i] = vt[i] + diff * dv[i] + half_d2 * d2v[i] end
    elseif deriv_order >= 1 and dv then
        for i = 0, n-1 do out[i] = vt[i] + diff * dv[i] end
    else
        for i = 0, n-1 do out[i] = vt[i] end
    end
    return out
end

local function rkg2_b(j)
    if j <= 0 then return 1 end
    if j == 1 then return 1/3 end
    return 4 * (j - 1) * (j + 4) / (3 * j * (j + 1) * (j + 2) * (j + 3))
end

local function rkg2_substep(xt, vt, s, t_curr, t_prev, deriv_order, dv, d2v, n)
    local dt = t_curr - t_prev
    local Y_j_2 = {}; local Y_j_1 = {}; local Y_j = {}
    for i = 0, n-1 do Y_j_2[i] = xt[i]; Y_j_1[i] = xt[i]; Y_j[i] = xt[i] end
    local s2ps = s * s + s - 2
    for j = 1, s do
        if j == 1 then
            local mu_t = 6 / ((s + 4) * (s - 1))
            for i = 0, n-1 do Y_j[i] = Y_j_1[i] - dt * mu_t * vt[i] end
        else
            local frac = (j == 2) and (4 / (3 * s2ps)) or ((j-1)*(j-1)+(j-1)-2) / s2ps
            local bj = rkg2_b(j); local bj1 = rkg2_b(j-1); local bj2 = rkg2_b(j-2)
            local mu = (2*j+1) * bj / (j * bj1)
            local nu = -(j+1) * bj / (j * bj2)
            local mu_t = mu * 6 / ((s+4)*(s-1))
            local gamma_t = -mu_t * (1 - j*(j+1)*bj1/2)
            local diff = -frac * dt
            local vel = taylor_approx(vt, deriv_order, diff, dv, d2v, n)
            for i = 0, n-1 do
                Y_j[i] = mu*Y_j_1[i] + nu*Y_j_2[i] + (1-mu-nu)*xt[i]
                        - dt*mu_t*vel[i] - dt*gamma_t*vt[i]
            end
        end
        for i = 0, n-1 do Y_j_2[i] = Y_j_1[i]; Y_j_1[i] = Y_j[i] end
    end
    return Y_j, not has_nan_inf(Y_j, n)
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
        result, success = rkg2_substep(xt, vt, s, t_curr, t_prev, deriv_order, dv, d2v, n)
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
