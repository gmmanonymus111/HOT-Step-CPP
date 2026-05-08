-- cosine.lua: Cosine scheduler — half-cosine S-curve

scheduler = {
    name        = "cosine",
    display     = "Cosine",
    description = "Cosine annealing — balanced S-curve",
}

function schedule(output, num_steps, shift)
    for i = 0, num_steps - 1 do
        local frac = i / num_steps
        output[i] = 0.5 * (1.0 + math.cos(math.pi * frac))
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
