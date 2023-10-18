kₐ = 2
kᵢ = 5
g = RegularFactorGraph(kₐ, kᵢ)
bp = rand_bp(g, 2)
bethe_free_energy(bp)