-- sgm_uniform.lua: SGM Uniform (Karras) — uniform in σ^(1/ρ) space

scheduler = {
    name        = "sgm_uniform",
    display     = "SGM-Uniform (Karras)",
    description = "Karras σ-ramp (ρ=7), front-loads structural steps",
}

function schedule(output, num_steps, shift)
    local t_max = 0.999
    local t_min = 0.001
    local sigma_max = t_max / (1 - t_max)
    local sigma_min = t_min / (1 - t_min)
    local rho = 7.0

    local inv_rho = 1.0 / rho
    local s_max = sigma_max ^ inv_rho
    local s_min = sigma_min ^ inv_rho

    for i = 0, num_steps - 1 do
        local frac = i / num_steps
        local sigma = (s_max + frac * (s_min - s_max)) ^ rho
        output[i] = sigma / (1 + sigma)
    end
    clamp(output, num_steps)
    apply_shift(output, num_steps, shift)
end

function apply_shift(ts, n, shift)
    if shift == 1.0 then return end
    for i = 0, n - 1 do
        local t = ts[i]; ts[i] = shift * t / (1.0 + (shift - 1.0) * t)
    end
end

function clamp(ts, n)
    for i = 0, n - 1 do
        if ts[i] < 1e-6 then ts[i] = 1e-6 end
        if ts[i] > 1.0 then ts[i] = 1.0 end
    end
end
