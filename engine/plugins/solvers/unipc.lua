-- unipc.lua: UniPC (Unified Predictor-Corrector, 2 NFE)
-- B(h)1 variant with data prediction in log-SNR space.

solver = {
    name        = "unipc",
    display     = "UniPC (2 NFE)",
    description = "Unified predictor-corrector in log-SNR space",
    nfe         = 2,
    order       = 2,
    needs_model = true,
    stateful    = true,
    stochastic  = false,
}

local history = {}  -- {model_output={}, t=float}
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
    if K == 1 then return solve_1x1(R, b)
    elseif K == 2 then return solve_2x2(R, b)
    else return {0} end
end

local function bh1_update(xt, vt, t_curr, t_next, n, model_fn, vt_buf, use_corrector)
    -- Data prediction: D_n = x - t * v
    local D_n = {}
    for i = 0, n-1 do D_n[i] = xt[i] - t_curr * vt[i] end

    local lam_curr = lambda(t_curr)
    local lam_next = lambda(t_next)
    local h = lam_next - lam_curr
    local alpha_next = 1 - t_next
    local sigma_next = t_next
    local sigma_curr = math.max(t_curr, 1e-7)
    local hh = -h
    local B_h = hh
    local h_phi_1 = expm1(hh)

    local avail = #history
    local order = math.min(max_order, avail + 1)
    local K = order
    local n_D1 = order - 1

    -- h_phi_k sequence
    local h_phi_k_vals = {h_phi_1}
    local fact = 1; local hpk = h_phi_1
    for k = 1, order do
        hpk = hpk / hh - 1 / fact
        h_phi_k_vals[k+1] = hpk
        fact = fact * (k + 1)
    end

    -- rks, R matrix, b vector
    local rks = {}
    for i = 1, n_D1 do
        local hist_idx = avail - i + 1
        local lam_hist = lambda(history[hist_idx].t)
        rks[i] = (lam_hist - lam_curr) / h
    end
    rks[n_D1 + 1] = 1

    local R_mat = {}
    for row = 1, K do
        for col = 1, K do
            R_mat[(row-1)*K + col] = rks[col] ^ (row - 1)
        end
    end

    local b_vec = {}
    fact = 1; hpk = h_phi_1
    for i = 1, K do
        hpk = hpk / hh - 1 / fact
        b_vec[i] = hpk * fact / B_h
        fact = fact * (i + 1)
    end

    -- D1 differences
    local d1 = {}
    for i = 1, n_D1 do
        local hist_idx = avail - i + 1
        local D_hist = history[hist_idx].model_output
        local rk_inv = (math.abs(rks[i]) > 1e-12) and (1 / rks[i]) or 0
        d1[i] = {}
        for j = 0, n-1 do d1[i][j] = (D_hist[j] - D_n[j]) * rk_inv end
    end

    -- Base term
    local sigma_ratio = (math.abs(sigma_curr) > 1e-7) and (sigma_next / sigma_curr) or 0
    local x_t_ = {}
    for i = 0, n-1 do
        x_t_[i] = sigma_ratio * xt[i] - alpha_next * h_phi_1 * D_n[i]
    end

    -- Predictor
    if n_D1 > 0 then
        local rhos_p
        if order == 2 then rhos_p = {0.5}
        else
            local Kp = K - 1; local R_p = {}
            for row = 1, Kp do for col = 1, Kp do R_p[(row-1)*Kp+col] = R_mat[(row-1)*K+col] end end
            rhos_p = solve(Kp, R_p, b_vec)
        end
        for i = 0, n-1 do
            local pred = 0
            for k = 1, n_D1 do pred = pred + rhos_p[k] * d1[k][i] end
            xt[i] = x_t_[i] - alpha_next * B_h * pred
        end
    else
        for i = 0, n-1 do xt[i] = x_t_[i] end
    end

    -- Corrector
    if use_corrector and model_fn then
        model_fn(xt, t_next)
        local D_corr_diff = {}
        for i = 0, n-1 do
            local D_corr = xt[i] - t_next * vt_buf[i]
            D_corr_diff[i] = D_corr - D_n[i]
        end
        local rhos_c
        if order == 1 then rhos_c = {0.5}
        else rhos_c = solve(K, R_mat, b_vec) end

        for i = 0, n-1 do
            local corr = 0
            for k = 1, n_D1 do corr = corr + rhos_c[k] * d1[k][i] end
            corr = corr + rhos_c[K] * D_corr_diff[i]
            xt[i] = x_t_[i] - alpha_next * B_h * corr
        end
    end

    -- Update history
    table.insert(history, {model_output = D_n, t = t_curr})
    while #history > max_order do table.remove(history, 1) end
end

function step(xt, vt, t_curr, t_prev, n, model_fn, vt_buf)
    -- Reset state on first step of a new generation
    if (step_index or 0) == 0 then history = {} end

    bh1_update(xt, vt, t_curr, t_prev, n, model_fn, vt_buf, true)
end
