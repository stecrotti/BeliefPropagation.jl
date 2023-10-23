"""
    eachstate(bp::BP)

Return a lazy iterator to the joint configuration of all variables.
"""
function eachstate(bp::BP)
    return Iterators.product((1:nstates(bp, i) for i in variables(bp.g))...)
end

function exact_normalization(bp::BP)
    return sum(evaluate(bp, x) for x in eachstate(bp))
end

function exact_prob(bp::BP; Z = exact_normalization(bp))
    p = [evaluate(bp, x) / Z for x in eachstate(bp)]
    return p
end

function exact_marginals(bp::BP; p_exact = exact_prob(bp))
    dims = 1:ndims(p_exact)
    return map(variables(bp.g)) do i
        vec(sum(p_exact; dims=dims[Not(i)]))
    end
end

function exact_factor_marginals(bp::BP; p_exact = exact_prob(bp))
    dims = 1:ndims(p_exact)
    return map(factors(bp.g)) do a
        ∂a = neighbors(bp.g, factor(a))
        dropdims(sum(p_exact; dims=dims[Not(∂a)]); dims=tuple(dims[Not(∂a)]...))
    end
end

function exact_avg_energy(bp::BP; p_exact = exact_prob(bp))
    k = keys(p_exact)
    sum(energy(bp, Tuple(k[x])) * p_exact[x] for x in eachindex(p_exact))
end

function exact_minimum_energy(bp::BP)
    return minimum(energy(bp, x) for x in eachstate(bp))
end

function rand_bp(rng::AbstractRNG, g::FactorGraph, qs)
    ψ = [rand_factor(rng, [qs[i] for i in neighbors(g,factor(a))]) for a in factors(g)] 
    return BP(g, ψ, qs)  
end
function rand_bp(rng::AbstractRNG, g::RegularFactorGraph, q)
    ψ = [rand_factor(rng, fill(q, degree(g, factor(1))))] 
    return BP(g, ψ, q)  
end
rand_bp(g::AbstractFactorGraph, qs) = rand_bp(default_rng(), g, qs)

function test_observables(bp::BP; kwargs...)
    @test isapprox(beliefs(bp), exact_marginals(bp); kwargs...)
    @test isapprox(factor_beliefs(bp), exact_factor_marginals(bp); kwargs...)
    @test isapprox(exp(-bethe_free_energy(bp)), exact_normalization(bp))
end