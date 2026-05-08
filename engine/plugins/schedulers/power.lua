-- power.lua: Power-law scheduler with configurable exponent

scheduler = {
    name        = "power",
    display     = "Power (p=2)",
    description = "Power-law t^p, front-loaded",
    params      = {
        { key = "exponent", type = "slider", label = "Exponent",
          default = 2.0, min = 0.5, max = 5.0, step = 0.1,
          hint = "Higher values front-load more steps at high noise" },
    },
}

function schedule(output, num_steps, shift)
    local p = (params and params.exponent) or 2.0
    for i = 0, num_steps - 1 do
        local frac = i / num_steps
        output[i] = (1.0 - frac) ^ p
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
