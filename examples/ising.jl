using BeliefPropagation
using BeliefPropagation.Models, SparseArrays, IndexedGraphs, Random

A = reduce(hcat, [1,1,0,0,0,0][randperm(6)] for _ in 1:10) |> permutedims |> sparse 
g = FactorGraph(A)
ψ = fill(IsingCoupling(1.0), nfactors(g))
ϕ = fill(IsingField(0.5), nvariables(g))
qs = fill(2, nv(g))
bp = BP(g, ψ, qs; ϕ)
iterate!(bp)
reduce.(-, beliefs(bp))