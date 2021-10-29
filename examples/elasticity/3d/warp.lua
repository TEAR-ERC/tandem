function warp(x, y, z)
    return x + 0.1 * math.sin(3.0 * math.pi * y),
           y + 0.1 * math.sin(3.0 * math.pi * z),
           z + 0.1 * math.sin(3.0 * math.pi * x)
end
