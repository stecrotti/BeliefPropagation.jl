using BeliefPropagation
# using Test
# using Aqua

# @testset "BeliefPropagation.jl" begin
#     @testset "Code quality (Aqua.jl)" begin
#         Aqua.test_all(BeliefPropagation; ambiguities = false,)
#         Aqua.test_ambiguities(BeliefPropagation)
#     end
# end

using BeliefPropagation.Models, SparseArrays, IndexedGraphs, Random
A = reduce(hcat, [1,1,0,0,0,0][randperm(6)] for _ in 1:10) |> permutedims |> sparse 
g = FactorGraph(A)
ψ = fill(IsingCoupling(1.0), nfactors(g))
ϕ = fill(IsingField(0.5), nvariables(g))
qs = fill(2, nv(g))
bp = BP(g, ψ, qs; ϕ)
iterate!(bp)