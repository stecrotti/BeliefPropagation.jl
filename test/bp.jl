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
    iterate!(bp; maxiter=100, f, damp=0.2)
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
    iterate_ms!(bp; maxiter=10, f)
    bfe = bethe_free_energy_ms(bp)
    # @test bfe ≈ sum(f)
    @test bfe ≈ exact_minimum_energy(bp)
end