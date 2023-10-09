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
    Z = exp(-bethe_free_energy(bp))

    @test m[1] ≈ tanh(β * h[1] + atanh(tanh(β * h[2])*tanh(β*J[1])))
    @test m[2] ≈ tanh(β * h[2] + atanh(tanh(β * h[1])*tanh(β*J[1])))
    @test c ≈ tanh(β * J[1] + atanh(tanh(β * h[1])*tanh(β*h[2])))
    @test Z ≈ 4*(cosh(β*J[1])*cosh(β*h[1])*cosh(β*h[2]) + sinh(β*J[1])*sinh(β*h[1])*sinh(β*h[2]))
end

@testset "Ising random tree" begin
    rng = MersenneTwister(0)
    N = 8
    g = prufer_decode(rand(rng, 1:N, N-2)) |> IndexedGraph
    J = randn(rng, ne(g))
    h = randn(rng, nv(g))
    β = rand(rng)
    ising = Ising(g, J, h, β)
    bp = BP(ising)
    iterate!(bp; maxiter=100)
    b = beliefs(bp)
    b_ex = exact_marginals(ising)
    @test b ≈ b_ex
    pb = factor_beliefs(bp)
    pb_ex = exact_pair_marginals(ising)
    @test pb ≈ pb_ex
    e = avg_energy(bp)
    e_ex = exact_avg_energy(ising) * β
    @test e ≈ e_ex
    z = exp(-bethe_free_energy(bp))
    z_ex = exact_normalization(ising)
end

@testset "Ising random tree - maxsum" begin
    rng = MersenneTwister(0)
    N = 10
    g = prufer_decode(rand(rng, 1:N, N-2)) |> IndexedGraph
    J = randn(rng, ne(g))
    h = randn(rng, nv(g))
    β = rand(rng)
    ising = Ising(g, J, h, β)
    bp = BP(ising)
    iterate_ms!(bp; maxiter=100)
    e = avg_energy(avg_energy_ms, bp)
    e_ex = minimum_energy(ising) * β
    @test e ≈ e_ex
end