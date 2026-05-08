-- beta57.lua: Beta(0.5, 0.7) distribution scheduler
-- Requires beta_math companion for the inverse CDF computation.

local beta_math = require("beta_math")

scheduler = {
    name        = "beta57",
    display     = "Beta 57",
    description = "Beta(0.5,0.7) — smooth S-curve from RES4LYF",
}

function schedule(output, num_steps, shift)
    local alpha = 0.5
    local beta  = 0.7

    for i = 0, num_steps - 1 do
        local u = (i + 0.5) / num_steps
        local t = 1.0 - beta_math.ppf(u, alpha, beta)
        output[i] = t
    end
    -- Sort descending
    local vals = {}
    for i = 0, num_steps - 1 do vals[i+1] = output[i] end
    table.sort(vals, function(a,b) return a > b end)
    for i = 0, num_steps - 1 do output[i] = vals[i+1] end
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
