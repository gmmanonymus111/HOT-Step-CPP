-- linear_quadratic.lua: Linear start, quadratic finish

scheduler = {
    name        = "linear_quadratic",
    display     = "Linear-Quadratic",
    description = "Linear start, quadratic finish",
}

function schedule(output, num_steps, shift)
    local crossover = 0.5
    local n_linear = math.max(math.floor(num_steps * crossover), 1)
    local n_quad = num_steps - n_linear
    local t_cross = 1.0 - crossover

    for i = 0, n_linear - 1 do
        output[i] = 1.0 - i * crossover / n_linear
    end
    for i = 0, n_quad - 1 do
        local frac = (i + 1) / n_quad
        output[n_linear + i] = t_cross * (1.0 - frac * frac)
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
