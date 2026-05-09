-- smc_cfg.lua: SMC-CFG — Sliding Mode Control Guidance
-- Paper: "CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance"
--        Han et al., 2025 (arXiv:2603.03281)
--
-- Reinterprets CFG as a feedback control system and applies Sliding Mode
-- Control (SMC) to stabilize guidance, especially at high scales.
--
-- Key idea: define a sliding surface s(t) = ė(t) + λ·e(t) over the
-- semantic error e = v_cond - v_uncond, then apply a switching control
-- term Δe = -k·sign(s) that enforces convergence to a stable manifold.
--
-- Formula:
--   e(t) = v_cond - v_uncond
--   ė(t) ≈ (e(t) - e(t-1)) / Δt       (finite difference)
--   s(t) = ė(t) + λ·e(t)               (sliding surface)
--   Δe   = -k · sign(s)                 (switching control)
--   v_guided = v_uncond + w · (e + Δe)
--
-- Stateful: stores previous error vector across steps.

guidance = {
    name        = "smc_cfg",
    display     = "SMC-CFG",
    description = "Sliding mode control guidance (Han et al. 2025)",
    params      = {
        { key = "lambda", type = "slider", label = "λ (Surface Slope)",
          default = 0.5, min = 0.01, max = 2.0, step = 0.01,
          hint = "Controls the sliding surface shape. Higher = faster convergence" },
        { key = "k", type = "slider", label = "k (Switching Gain)",
          default = 0.1, min = 0.01, max = 1.0, step = 0.01,
          hint = "Force toward the sliding surface. Too high = vibrations" },
    },
}

-- Stateful: previous error buffer
local prev_error = nil
local prev_n = 0

local function sign(x)
    if x > 0 then return 1.0
    elseif x < 0 then return -1.0
    else return 0.0
    end
end

function guide(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
    local n = Oc * T
    local lam = (params and params.lambda) or 0.5
    local k   = (params and params.k) or 0.1

    -- Compute semantic error e(t) = cond - uncond
    local error_now = {}
    for i = 0, n - 1 do
        error_now[i] = pred_cond[i] - pred_uncond[i]
    end

    -- First step: no previous error available, use standard CFG with APG
    if prev_error == nil or prev_n ~= n then
        -- Fall back to APG for the first step
        apg(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
        -- Store error for next step
        prev_error = error_now
        prev_n = n
        return
    end

    -- Compute ė ≈ (e_now - e_prev) / dt  (dt is a global from C++)
    -- Note: dt can be negative for reverse-time flow; use absolute
    local dt_abs = math.abs(dt or 1.0)
    if dt_abs < 1e-8 then dt_abs = 1e-8 end
    local inv_dt = 1.0 / dt_abs

    -- Compute sliding surface and switching control, element-wise
    for i = 0, n - 1 do
        local e_dot = (error_now[i] - prev_error[i]) * inv_dt
        local s = e_dot + lam * error_now[i]
        local delta_e = -k * sign(s)
        -- v_guided = v_uncond + w * (e + Δe)
        result[i] = pred_uncond[i] + guidance_scale * (error_now[i] + delta_e)
    end

    -- Store for next step
    prev_error = error_now
    prev_n = n
end
