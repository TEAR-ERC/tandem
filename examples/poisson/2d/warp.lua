function partialAnnulus(x, y)
    local r = 0.5 * (x + 1.0)
    phi = 0.5 * math.pi * y
    return r * math.cos(phi), r * math.sin(phi)
end
