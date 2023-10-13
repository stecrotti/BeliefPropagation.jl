rng = MersenneTwister(0)

@testset "Isolated nodes" begin
    g = FactorGraph([0 1 1 0 0;
                     1 0 0 0 0;
                     0 0 1 1 0;
                     0 0 0 0 0])
    nfact, nvar = size(adjacency_matrix(g))
    qs = rand(rng, 2:2, nvar)
    bp = rand_bp(rng, g, qs)
    f = zeros(nvar)
    iterate!(bp; maxiter=50, f, damp=0.2, tol=0)
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

@testset "On a tree" begin
    n = 10
    g = rand_tree_factor_graph(n)
    qs = rand(rng, 2:4, nvariables(g))
    bp = rand_bp(rng, g, qs)
    iterate!(bp; maxiter=100, tol=0.0)
    b = beliefs(bp)
    test_observables(bp)
end