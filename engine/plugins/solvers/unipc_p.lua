-- unipc_p.lua: UniPC Predictor only (1 NFE, stateful)
-- Same as UniPC but without the corrector step.

solver = {
    name        = "unipc_p",
    display     = "UniPC Predictor (1 NFE)",
    description = "UniPC predictor-only (no corrector, 1 NFE)",
    nfe         = 1,
    order       = 2,
    needs_model = false,
    stateful    = true,
    stochastic  = false,
}

-- Shares the same logic as unipc.lua but with use_corrector = false
-- For brevity, we duplicate the core with corrector disabled.

local history = {}
local max_order = 2

local function lambda(t)
    t = math.max(t, 1e-7); t = math.min(t, 1 - 1e-7)
    return math.log((1 - t) / t)
end

local function expm1(x) return math.exp(x) - 1 end

local function solve_1x1(R, b) return {b[1] / (math.abs(R[1]) > 1e-12 and R[1] or 1)} end
local function solve_2x2(R, b)
    local det = R[1]*R[4] - R[2]*R[3]
    if math.abs(det) < 1e-12 then return {0, 0} end
    local inv = 1 / det
    return {(R[4]*b[1] - R[2]*b[2]) * inv, (R[1]*b[2] - R[3]*b[1]) * inv}
end
local function solve(K, R, b)
    if K == 1 then return solve_1x1(R, b) else return solve_2x2(R, b) end
end

function step(xt, vt, t_curr, t_prev, n)
    -- Reset state on first step of a new generation
    if (step_index or 0) == 0 then history = {} end

    local D_n = {}
    for i = 0, n-1 do D_n[i] = xt[i] - t_curr * vt[i] end

    local lam_curr = lambda(t_curr)
    local lam_next = lambda(t_prev)
    local h = lam_next - lam_curr
    local alpha_next = 1 - t_prev
    local sigma_next = t_prev
    local sigma_curr = math.max(t_curr, 1e-7)
    local hh = -h
    local h_phi_1 = expm1(hh)

    local avail = #history
    local order = math.min(max_order, avail + 1)
    local n_D1 = order - 1

    local rks = {}
    for i = 1, n_D1 do
        local hist_idx = avail - i + 1
        rks[i] = (lambda(history[hist_idx].t) - lam_curr) / h
    end

    local d1 = {}
    for i = 1, n_D1 do
        local D_hist = history[avail - i + 1].model_output
        local rk_inv = (math.abs(rks[i]) > 1e-12) and (1 / rks[i]) or 0
        d1[i] = {}
        for j = 0, n-1 do d1[i][j] = (D_hist[j] - D_n[j]) * rk_inv end
    end

    local sigma_ratio = (math.abs(sigma_curr) > 1e-7) and (sigma_next / sigma_curr) or 0
    for i = 0, n-1 do
        xt[i] = sigma_ratio * xt[i] - alpha_next * h_phi_1 * D_n[i]
    end

    if n_D1 > 0 then
        local rhos_p = (order == 2) and {0.5} or solve(n_D1, {1}, {0.5})
        for i = 0, n-1 do
            local pred = 0
            for k = 1, n_D1 do pred = pred + rhos_p[k] * d1[k][i] end
            xt[i] = xt[i] - alpha_next * hh * pred
        end
    end

    table.insert(history, {model_output = D_n, t = t_curr})
    while #history > max_order do table.remove(history, 1) end
end
