local BP5 = require "BP5"

bp5_exact = BP5.new({eps=0.0})
bp5_outside = BP5.new({eps=1e-3})
bp5_inside = BP5.new({eps=-1e-3})

