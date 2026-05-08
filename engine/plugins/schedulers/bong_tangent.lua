-- bong_tangent.lua: Tangent-based scheduler, concentrates at high noise

scheduler = {
    name        = "bong_tangent",
    display     = "Tangent",
    description = "Front-loaded (structural focus)",
}

function schedule(output, num_steps, shift)
    local scale = 1.5
    for i = 0, num_steps - 1 do
        local frac = (i + 0.5) / num_steps
        local angle = frac * math.pi / 2.0
        local tan_val = math.tan(angle)
        output[i] = 1.0 - (2.0 / math.pi) * math.atan(tan_val * scale)
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
