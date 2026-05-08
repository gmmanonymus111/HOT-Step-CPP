-- beta_math.lua: Beta distribution math helpers (companion data file)
-- Provides regularized incomplete beta function and its inverse (ppf).
-- Ported from engine/src/schedulers/scheduler-implementations.h

local M = {}

-- Log-gamma (uses Lua's built-in math library)
local function lgamma(x)
    -- Lanczos approximation for log-gamma
    if x <= 0 then return 0 end
    local g = 7
    local c = {
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    }
    if x < 0.5 then
        return math.log(math.pi / math.sin(math.pi * x)) - lgamma(1 - x)
    end
    x = x - 1
    local a = c[1]
    local t = x + g + 0.5
    for i = 2, #c do
        a = a + c[i] / (x + i - 1)
    end
    return 0.5 * math.log(2 * math.pi) + (x + 0.5) * math.log(t) - t + math.log(a)
end

-- Log of beta function: B(a,b) = Gamma(a)*Gamma(b)/Gamma(a+b)
local function lbeta(a, b)
    return lgamma(a) + lgamma(b) - lgamma(a + b)
end

-- Regularized incomplete beta function via continued fraction (Lentz's method)
local function betainc(a, b, x)
    if x <= 0 then return 0 end
    if x >= 1 then return 1 end

    -- Use symmetry for convergence
    if x > (a + 1) / (a + b + 2) then
        return 1 - betainc(b, a, 1 - x)
    end

    local ln_pre = a * math.log(x) + b * math.log(1 - x) - lbeta(a, b)
    local qab = a + b
    local qap = a + 1
    local qam = a - 1
    local c = 1
    local d = 1 - qab * x / qap
    if math.abs(d) < 1e-30 then d = 1e-30 end
    d = 1 / d
    local h = d

    for m = 1, 200 do
        local m2 = 2 * m
        -- Even numerator
        local aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d; if math.abs(d) < 1e-30 then d = 1e-30 end
        c = 1 + aa / c; if math.abs(c) < 1e-30 then c = 1e-30 end
        d = 1 / d; h = h * d * c

        -- Odd numerator
        aa = -((a + m) * (qab + m) * x) / ((a + m2) * (qap + m2))
        d = 1 + aa * d; if math.abs(d) < 1e-30 then d = 1e-30 end
        c = 1 + aa / c; if math.abs(c) < 1e-30 then c = 1e-30 end
        d = 1 / d
        local del = d * c; h = h * del

        if math.abs(del - 1) < 3e-14 then break end
    end

    return math.exp(ln_pre) * h / a
end

-- Beta PDF
local function beta_pdf(x, a, b)
    if x <= 0 or x >= 1 then return 0 end
    return math.exp((a - 1) * math.log(x) + (b - 1) * math.log(1 - x) - lbeta(a, b))
end

-- Inverse CDF (ppf) via Newton's method
function M.ppf(p, a, b)
    if p <= 0 then return 0 end
    if p >= 1 then return 1 end

    -- Initial guess
    local mu = a / (a + b)
    local var = a * b / ((a + b)^2 * (a + b + 1))
    local sigma = math.sqrt(var)
    local x = mu + sigma * (2 * p - 1)
    if x < 0.001 then x = 0.001 end
    if x > 0.999 then x = 0.999 end

    -- Newton-Raphson
    for _ = 1, 50 do
        local F = betainc(a, b, x) - p
        local f = beta_pdf(x, a, b)
        if math.abs(f) < 1e-30 then break end
        local dx = -F / f
        x = x + dx
        if x < 1e-10 then x = 1e-10 end
        if x > 1 - 1e-10 then x = 1 - 1e-10 end
        if math.abs(dx) < 1e-12 then break end
    end
    return x
end

return M
