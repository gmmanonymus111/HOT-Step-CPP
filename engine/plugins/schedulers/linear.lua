-- linear.lua: Linear (uniform) scheduler — the ACE-Step default

scheduler = {
    name        = "linear",
    display     = "Linear",
    description = "Uniform spacing (default)",
}

function schedule(output, num_steps, shift)
    for i = 0, num_steps - 1 do
        output[i] = 1.0 - i / num_steps
    end
    apply_shift(output, num_steps, shift)
end

-- Standard shift warp: t' = shift*t / (1 + (shift-1)*t)
function apply_shift(ts, n, shift)
    if shift == 1.0 then return end
    for i = 0, n - 1 do
        local t = ts[i]
        ts[i] = shift * t / (1.0 + (shift - 1.0) * t)
    end
end

-- Clamp to [1e-6, 1.0]
function clamp(ts, n)
    for i = 0, n - 1 do
        if ts[i] < 1e-6 then ts[i] = 1e-6 end
        if ts[i] > 1.0 then ts[i] = 1.0 end
    end
end
