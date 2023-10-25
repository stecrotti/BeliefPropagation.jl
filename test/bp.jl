rng = MersenneTwister(0)

@testset "Basic example - no ϕ factors" begin
    A = [0 1 1 0;
         1 0 0 0;
         0 0 1 1]
    g = FactorGraph(A)
    states = [3, 2, 2, 4]
    ψ₁ = BPFactor([1.1 0.3;
               0.5 2.5])
    ψ₂ = BPFactor([1.0, 0.3, 1.0])
    ψ₃ = BPFactor([0.8 0.0 0.1 0.9;
                0.0 2.9 0.7 1.1])
    ψ = [ψ₁, ψ₂, ψ₃]
    bp = BP(g, ψ, states)
    f = zeros(2)
    iterate!(bp; maxiter=2, tol=1e-12, f)
    f_bethe = bethe_free_energy(bp)
    z_bethe = exp(-f_bethe)
    z = sum(ψ₁([x₂,x₃]) * ψ₂([x₁]) * ψ₃([x₃, x₄])
        for x₁ in 1:3, x₂ in 1:2, x₃ in 1:2, x₄ in 1:4)
    @test z_bethe ≈ z
    p43 = 1/z * sum((x₄==3) * ψ₁([x₂,x₃]) * ψ₂([x₁]) * ψ₃([x₃, x₄])
        for x₁ in 1:3, x₂ in 1:2, x₃ in 1:2, x₄ in 1:4)
    @test beliefs(bp)[4][3] ≈ p43
end

@testset "Isolated nodes" begin
    g = FactorGraph([0 1 1 0 0;
                     1 0 0 0 0;
                     0 0 1 1 0;
                     0 0 0 0 0])
    nfact, nvar = size(adjacency_matrix(g))
    qs = rand(rng, 2:2, nvar)
    bp = rand_bp(rng, g, qs)
    f = zeros(2)
    iterate!(bp; maxiter=50, f, damp=0.0, tol=0)
    b = beliefs(bp)
    @test b ≈ exact_marginals(bp)
    bf = factor_beliefs(bp)
    @test bf ≈ exact_factor_marginals(bp)
    e = avg_energy(bp)
    @test e ≈ exact_avg_energy(bp)
    bfe = bethe_free_energy(bp)
    @test bfe ≈ sum(f)
    @test exp(-bfe) ≈ exact_normalization(bp)

    bp = rand_bp(rng, g, qs)
    f = zeros(nvar)
    iterate_ms!(bp; maxiter=10, f, tol=0)
    bfe = bethe_free_energy_ms(bp)
    @test bfe ≈ sum(f)
    @test bfe ≈ exact_minimum_energy(bp)
end

@testset "Reinforcement" begin
    g = FactorGraph([0 1 1 0 0;
                    1 0 0 0 0;
                    0 0 1 1 0;
                    0 0 0 0 0])
    qs = rand(rng, 2:4, nvariables(g))
    bp = rand_bp(rng, g, qs)
    iterate!(bp; maxiter=50, rein=0, tol=0)
    iterate!(bp; maxiter=50, rein=10, tol=0)
    b_bp = beliefs(bp)

    reset!(bp)
    iterate_ms!(bp; maxiter=10)
    b_ms = beliefs(bp)
    @test all(argmax(bi1) == argmax(bi2) for (bi1, bi2) in zip(b_bp, b_ms))
end

@testset "Tree factor graph" begin
    n = 10
    g = rand_tree_factor_graph(n)
    qs = rand(rng, 2:4, nvariables(g))
    bp = rand_bp(rng, g, qs)
    f = zeros(n)
    iterate!(bp; maxiter=100, tol=0.0, f)
    b = beliefs(bp)
    test_observables(bp)
    @test sum(f) ≈ bethe_free_energy(bp)
end

@testset "Convergence of beliefs" begin
    n = 10
    g = rand_tree_factor_graph(n)
    qs = rand(rng, 2:4, nvariables(g))
    bp = rand_bp(rng, g, qs)
    iterate!(bp; maxiter=100, check_convergence=belief_convergence(1e-12))
    b = beliefs(bp)
    test_observables(bp)
end