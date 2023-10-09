rng = MersenneTwister(0)

@testset "Isolated nodes" begin
    g = FactorGraph([0 1 1 0 0;
                     1 0 0 0 0;
                     0 0 1 1 0;
                     0 0 0 0 0])
    nfact, nvar = size(adjacency_matrix(g))
    qs = rand(rng, 2:2, nvar)
    bp = rand_bp(rng, g, qs)
    iterate!(bp; maxiter=5)
    @test bethe_free_energy(bp) ≈ bethe_free_energy_slow(bp)
    bp = rand_bp(rng, g, qs)
    iterate_ms!(bp; maxiter=5)
end