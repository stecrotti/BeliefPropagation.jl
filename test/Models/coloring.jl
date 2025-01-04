@testset "Coloring factor" begin
    β = 100
    # Soft coloring is equivalent to Coloring in the large β limit
    @test all(
        isapprox(ColoringCoupling()((xᵢ, xⱼ)), SoftColoringCoupling(β)((xᵢ, xⱼ)); atol=1e-14)
        for xᵢ in 1:5 for xⱼ in 1:5
    )          
end


@testset "Coloring random tree" begin
    rng = MersenneTwister(0)
    N = 8
    t = prufer_decode(rand(rng, 1:N, N-2))
    g = pairwise_interaction_graph(IndexedGraph(t))
    k = 4   # number of colors
    states = fill(k, nv(t))
    ψ = fill(ColoringCoupling(), ne(t))
    ϕ = [rand_factor(k) for _ in vertices(t)]
    bp = BP(g, ψ, states; ϕ)
    iterate!(bp; maxiter=20, tol=0, damp=0.9)
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
    bfe_beliefs = BeliefPropagation.bethe_free_energy_bp_beliefs(bp)
    @test bfe ≈ bfe_beliefs
    z = exp(-bfe)
    z_ex = exact_normalization(bp)
    @test z ≈ z_ex
    test_za(bp)
end

@testset "Soft coloring random tree" begin
    rng = MersenneTwister(0)
    N = 8
    t = prufer_decode(rand(rng, 1:N, N-2))
    g = pairwise_interaction_graph(IndexedGraph(t))
    k = 4   # number of colors
    states = fill(k, nv(t))
    β = 1
    ψ = fill(SoftColoringCoupling(β), ne(t))
    ϕ = [rand_factor(k) for _ in vertices(t)]
    bp = BP(g, ψ, states; ϕ)
    iterate!(bp; maxiter=20, tol=0, damp=0.9)
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
    bfe_beliefs = BeliefPropagation.bethe_free_energy_bp_beliefs(bp)
    @test bfe ≈ bfe_beliefs
    z = exp(-bfe)
    z_ex = exact_normalization(bp)
    @test z ≈ z_ex
    test_za(bp)
end

@testset "Decimation" begin
    rng = MersenneTwister(0)
    N = 8
    A = sparse([2, 4, 7, 1, 6, 1, 5, 4, 6, 8, 3, 5, 1, 5], [1, 1, 1, 2, 3, 4, 4, 5, 5, 5, 6, 6, 7, 8], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 8, 8)
    g = pairwise_interaction_graph(IndexedGraph(A))
    k = 3   # number of colors
    states = fill(k, N)
    ψ = fill(ColoringCoupling(), 7)
    ϕ = [
        TabulatedBPFactor([1.0, 0.0, 0.0]),
        TabulatedBPFactor([0.0, 1.0, 0.0]),
        TabulatedBPFactor([0.8576486278354781, 0.933376113472654, 0.9887502004400418]),
        TabulatedBPFactor([0.28765908697080933, 0.22925096087844, 0.6898492319049271]),
        TabulatedBPFactor([0.9346852188449652, 0.667819817743093, 0.8451887825155666]),
        TabulatedBPFactor([0.19522313463573093, 0.5485804533857773, 0.9148128306625107]),
        TabulatedBPFactor([0.2564514957015539, 0.07406713024633105, 0.13299843191690675]),
        TabulatedBPFactor([0.6550227672869462, 0.10474144135869445, 0.35157484078739665])
    ]
    bp = BP(g, ψ, states; ϕ)
    iterate_ms!(bp)
    b_ms = argmax.(beliefs_ms(bp))
    reset!(bp)
    maxiter = 100
    tol = 1e-8
    cb = Decimation(bp, maxiter, tol)
    iterate!(bp; callbacks = [cb])
    b_dec = argmax.(beliefs_ms(bp))
    @test b_ms == b_dec
end