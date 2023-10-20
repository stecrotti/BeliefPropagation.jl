J  = 1.0
h = -0.2

kₐ = 2
kᵢ = 3
g = RegularFactorGraph(kₐ, kᵢ)
ψ = [IsingCoupling(J)]
ϕ = [IsingField(h)]
bp = BP(g, ψ, 2; ϕ)
iterate!(bp; maxiter=100, tol=1e-8)

using Graphs
g2 = pairwise_interaction_graph(IndexedGraph(complete_graph(4)))
bp2 = BP(g2, fill(IsingCoupling(J), 6), fill(2, 4); ϕ = fill(IsingField(h), 4))
iterate!(bp2; maxiter=100, tol=1e-8)

@test all(beliefs(bp)[1] ≈ beliefs(bp2)[i] for i in 1:4)
@test all(factor_beliefs(bp)[1] ≈ factor_beliefs(bp2)[ij] for ij in 1:6)
@test bethe_free_energy(bp)*4 ≈ bethe_free_energy(bp2)

reset!(bp); reset!(bp2)
iterate_ms!(bp; maxiter=100, tol=1e-8)
iterate_ms!(bp2; maxiter=100, tol=1e-8)

@test all(beliefs_ms(bp)[1] ≈ beliefs_ms(bp2)[i] for i in 1:4)
@test all(factor_beliefs_ms(bp)[1] ≈ factor_beliefs_ms(bp2)[ij] for ij in 1:6)