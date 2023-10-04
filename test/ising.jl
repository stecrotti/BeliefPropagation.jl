using BeliefPropagation.Models

@testset "Ising 2 spins" begin
    J = randn(1)
    h = randn(2)
    β = rand()
    g = IndexedGraph(path_graph(2))
    ising = Ising(g, J, h, β)
    bp = BP(ising)
    iterate!(bp, maxiter=2)
    b = beliefs(bp)
    m = reduce.(-, b)
    pb = only(factor_beliefs(bp))
    c = pb[1,1] + pb[2,2] - pb[1,2] - pb[2,1]

    @test m[1] ≈ tanh(β * h[1] + atanh(tanh(β * h[2])*tanh(β*J[1])))
    @test m[2] ≈ tanh(β * h[2] + atanh(tanh(β * h[1])*tanh(β*J[1])))
    @test c ≈ tanh(β * J[1] + atanh(tanh(β * h[1])*tanh(β*h[2])))
end

@testset "Ising random tree" begin
    N = 10
    g = prufer_decode(rand(1:N, N-2)) |> IndexedGraph
    J = randn(ne(g))
    h = randn(nv(g))
    β = rand()
    ising = Ising(g, J, h, β)
    bp = BP(ising)
    iterate!(bp; maxiter=100)
end

