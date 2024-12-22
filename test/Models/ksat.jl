rng = MersenneTwister(1)

@testset "ksat random tree" begin
    n = 3
    g = rand_tree_factor_graph(rng, n)
    ψ = [KSATClause(bitrand(rng, degree(g, factor(a)))) for a in factors(g)]
    bp = BP(g, ψ, fill(2, nvariables(g)))
    test_za(bp)
    iterate!(bp; maxiter=10, tol=0.0)
    test_observables_bp(bp)

    @testset "Generic BP factor" begin
        ψ_generic = [BPFactor(bp.ψ[a], fill(2, degree(g, factor(a)))) for a in factors(g)]
        ϕ_generic = [BPFactor(bp.ϕ[i], (2,)) for i in variables(g)]
        bp_generic = BP(g, ψ_generic, fill(2, nvariables(g)); ϕ = ϕ_generic)
        iterate!(bp_generic; maxiter=10, tol=0.0)
        test_observables_bp(bp_generic)
    end
end

@testset "ksat random graph" begin
    n = 10
    m = 3
    k = 3
    g = rand_regular_factor_graph(rng, n, m, k)
    ψ = [KSATClause(bitrand(rng, degree(g, factor(a)))) for a in factors(g)]
    ϕ = [BPFactor(1.0 .+ 1e-4 * randn(2)) for _ in variables(g)]
    bp = BP(g, ψ, fill(2, nvariables(g)); ϕ)
    test_za(bp)
    iterate!(bp; maxiter=100, tol=1e-12, rein=1e-2)
    xstar = argmax.(beliefs(bp))
    nunsat = sum(1 - Int(bp.ψ[a](xstar[i] for i in neighbors(bp.g, factor(a)))) for a in factors(bp.g))
end