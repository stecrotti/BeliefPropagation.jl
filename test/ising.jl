using BeliefPropagation.Models

J = randn(1)
h = randn(2)
β = randn()
g = IndexedGraph(path_graph(2))
ising = Ising(g, J, h, β)
bp = BP(ising)
iterate!(bp, maxiter=2)
m = reduce.(-, beliefs(bp))
pb = only(factor_beliefs(bp))
c = pb[1,1] + pb[2,2] - pb[1,2] - pb[2,1]

@testset "Ising 2 spins" begin
    @test m[1] ≈ tanh(β * h[1] + atanh(tanh(β * h[2])*tanh(β*J[1])))
    @test m[2] ≈ tanh(β * h[2] + atanh(tanh(β * h[1])*tanh(β*J[1])))
    @test c ≈ tanh(β * J[1] + atanh(tanh(β * h[1])*tanh(β*h[2])))
end


