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
    t = prufer_decode(rand(rng, 1:N, N-2))
    g = pairwise_interaction_graph(IndexedGraph(t))
    k = 3   # number of colors
    states = fill(k, nv(t))
    ψ = fill(ColoringCoupling(), ne(t))
    ϕ = vcat([BPFactor([1.0, 0, 0])],
        [BPFactor([0, 1.0, 0])],
        [rand_factor(k) for _ in 3:N]
    )
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