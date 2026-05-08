-- ddim_uniform.lua: DDIM Uniform — log-SNR uniform (S-shaped)

scheduler = {
    name        = "ddim_uniform",
    display     = "DDIM Uniform",
    description = "Log-SNR uniform (S-shaped)",
}

function schedule(output, num_steps, shift)
    local t_max = 0.9986
    local t_min = 0.0014
    local logit_max = math.log(t_max / (1 - t_max))
    local logit_min = math.log(t_min / (1 - t_min))

    for i = 0, num_steps - 1 do
        local frac = i / num_steps
        local logit_t = logit_max + (logit_min - logit_max) * frac
        output[i] = 1.0 / (1.0 + math.exp(-logit_t))
    end
    clamp(output, num_steps)
    apply_shift(output, num_steps, shift)
end

function apply_shift(ts, n, shift)
    if shift == 1.0 then return end
    for i = 0, n - 1 do
        local t = ts[i]
        ts[i] = shift * t / (1.0 + (shift - 1.0) * t)
    end
end

function clamp(ts, n)
    for i = 0, n - 1 do
        if ts[i] < 1e-6 then ts[i] = 1e-6 end
        if ts[i] > 1.0 then ts[i] = 1.0 end
    end
end
