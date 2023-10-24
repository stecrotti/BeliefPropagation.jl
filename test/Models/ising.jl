@testset "Ising 2 spins" begin
    J = randn(1)
    h = randn(2)
    β = rand()
    g = IndexedGraph(path_graph(2))
    ising = Ising(g, J, h, β)
    bp = BP(ising)
    iterate!(bp, maxiter=3, tol=0)
    b = beliefs(bp)
    m = reduce.(-, b)
    pb = only(factor_beliefs(bp))
    c = pb[1,1] + pb[2,2] - pb[1,2] - pb[2,1]
    bfe = bethe_free_energy(bp)
    z = exp(-bfe)

    @test m[1] ≈ tanh(β * h[1] + atanh(tanh(β * h[2])*tanh(β*J[1])))
    @test m[2] ≈ tanh(β * h[2] + atanh(tanh(β * h[1])*tanh(β*J[1])))
    @test c ≈ tanh(β * J[1] + atanh(tanh(β * h[1])*tanh(β*h[2])))
    @test z ≈ 4*(cosh(β*J[1])*cosh(β*h[1])*cosh(β*h[2]) + 
                    sinh(β*J[1])*sinh(β*h[1])*sinh(β*h[2]))
end

@testset "Ising random tree" begin
    rng = MersenneTwister(0)
    N = 8
    g = prufer_decode(rand(rng, 1:N, N-2)) |> IndexedGraph
    T = BigFloat
    J = randn(rng, ne(g))
    h = randn(rng, nv(g))
    β = rand(rng)
    ising = Ising(g, J, h, β)
    bp = BP(ising)
    f = zeros(N)
    iterate!(bp; maxiter=20, f, tol=0)
    b = beliefs(bp)
    b_ex = exact_marginals(bp)
    @test b ≈ b_ex
    pb = factor_beliefs(bp)
    pb_ex = exact_factor_marginals(bp)
    @test pb ≈ pb_ex
    e = avg_energy(bp)
    e_ex = exact_avg_energy(bp)
    @test e ≈ e_ex
    bfe = bethe_free_energy(bp)
    z = exp(-bfe)
    z_ex = exact_normalization(bp)
    @test sum(f) ≈ bethe_free_energy(bp)
end

@testset "Ising random tree - maxsum" begin
    rng = MersenneTwister(0)
    N = 9
    g = prufer_decode(rand(rng, 1:N, N-2)) |> IndexedGraph
    J = randn(rng, ne(g))
    h = randn(rng, nv(g))
    β = rand(rng)
    ising = Ising(g, J, h, β)
    bp = BP(ising)
    f = zeros(N)
    iterate_ms!(bp; maxiter=20, f, tol=0)
    e = avg_energy(avg_energy_ms, bp)
    e_ex = exact_minimum_energy(bp)
    @test e ≈ e_ex
    @test e ≈ sum(f)
end

@testset "Ising random tree - fast version" begin
    rng = MersenneTwister(1)
    N = 20
    g = prufer_decode(rand(rng, 1:N, N-2)) |> IndexedGraph
    J = randn(rng, ne(g))
    h = randn(rng, nv(g))
    β = rand(rng)
    ising = Ising(g, J, h, β)
    bp = fast_ising_bp(ising)
    f = zeros(N)
    iterate!(bp; maxiter=50, f, tol=1e-10)
    b = beliefs(bp)
    fb = factor_beliefs(bp)
    bp_slow = BP(ising)
    iterate!(bp_slow; maxiter=50, tol=1e-10)
    b_slow = beliefs(bp_slow)
    fb_slow = factor_beliefs(bp_slow)
    @test b ≈ b_slow
    @test fb ≈ fb_slow
    bfe = bethe_free_energy(bp)
    bfe_slow = bethe_free_energy(bp_slow)
    @test bfe ≈ bfe_slow
    @test sum(f) ≈ bfe
end

@testset "pspin random tree" begin
    n = 10
    g = rand_tree_factor_graph(n)
    J = randn(nfactors(g))
    h = randn(nvariables(g))
    ψ = IsingCoupling.(J)
    ϕ = IsingField.(h)
    bp = BP(g, ψ, fill(2, nvariables(g)); ϕ)
    f = zeros(n)
    iterate!(bp; maxiter=10, tol=0.0, f)
    test_observables(bp)
    bp_fast = fast_ising_bp(g, ψ, ϕ)
    iterate!(bp_fast; maxiter=10, tol=0.0)
    test_observables(bp_fast)
    @test sum(f) ≈ bethe_free_energy(bp)

    @testset "Generic BP factor" begin
        ψ_generic = [BPFactor(ψ[a], fill(2, degree(g, factor(a)))) for a in factors(g)]
        ϕ_generic = [BPFactor(ϕ[i], (2,)) for i in variables(g)]
        bp_generic = BP(g, ψ_generic, fill(2, nvariables(g)); ϕ = ϕ_generic)
        f = zeros(n)
        iterate!(bp_generic; maxiter=10, tol=0.0, f)
        test_observables(bp_generic)
        @test sum(f) ≈ bethe_free_energy(bp_generic)
    end

    ms = bp
    ms_fast = bp_fast
    iterate_ms!(ms; maxiter=10, tol=0.0)
    f = zeros(n)
    iterate_ms!(ms_fast; maxiter=10, tol=0.0, f)
    @test beliefs_ms(ms) ≈ beliefs_ms(ms_fast)
    @test factor_beliefs_ms(ms) ≈ factor_beliefs_ms(ms_fast)
    @test sum(f) ≈ bethe_free_energy_ms(ms_fast)
end

@testset "Ising infinite random regular" begin
    J  = 1.0
    h = -0.2

    kₐ = 2
    kᵢ = 3
    g = RegularFactorGraph(kₐ, kᵢ)
    ψ = [IsingCoupling(J)]
    ϕ = [IsingField(h)]
    bp = BP(g, ψ, 2; ϕ)
    bp = fast_ising_bp(g, ψ, ϕ)
    iterate!(bp; maxiter=100, tol=1e-12)

    using Graphs
    g2 = pairwise_interaction_graph(IndexedGraph(complete_graph(4)))
    bp2 = fast_ising_bp(g2, fill(IsingCoupling(J), 6), fill(IsingField(h), 4))
    iterate!(bp2; maxiter=100, tol=1e-12)

    @test all(beliefs(bp)[1] ≈ beliefs(bp2)[i] for i in 1:4)
    @test all(factor_beliefs(bp)[1] ≈ factor_beliefs(bp2)[ij] for ij in 1:6)
    @test bethe_free_energy(bp)*4 ≈ bethe_free_energy(bp2)
    @test avg_energy(bp)*4 ≈ avg_energy(bp2)

    reset!(bp); reset!(bp2)
    iterate_ms!(bp; maxiter=100, tol=1e-8)
    iterate_ms!(bp2; maxiter=100, tol=1e-8)

    @test all(beliefs_ms(bp)[1] ≈ beliefs_ms(bp2)[i] for i in 1:4)
    @test all(factor_beliefs_ms(bp)[1] ≈ factor_beliefs_ms(bp2)[ij] for ij in 1:6)
    @test bethe_free_energy_ms(bp)*4 ≈ bethe_free_energy_ms(bp2)
end