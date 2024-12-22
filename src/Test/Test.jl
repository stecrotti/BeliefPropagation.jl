module Test

using BeliefPropagation: BPFactor, TabulatedBPFactor, BP
using BeliefPropagation: beliefs_bp, factor_beliefs_bp, bethe_free_energy_bp
using BeliefPropagation
using IndexedFactorGraphs: AbstractFactorGraph, FactorGraph, InfiniteRegularFactorGraph,
    factor, factors, variables, neighbors, edge_indices

using IndexedGraphs: degree
using Test: @test
using Random: AbstractRNG, default_rng
using InvertedIndices: Not

export exact_normalization, exact_prob, exact_marginals, exact_factor_marginals,
    exact_avg_energy, exact_minimum_energy
export make_generic
export rand_factor, rand_bp
export test_observables_bp, test_za, test_observables_bp_generic

"""
    TabulatedBPFactor(f::BPFactor, states)

Construct a `TabulatedBPFactor` out of any `BPFactor`. Used mostly for tests.
"""
function BeliefPropagation.TabulatedBPFactor(f::BPFactor, states)
    values = [f(x) for x in Iterators.product((1:q for q in states)...)]
    return TabulatedBPFactor(values)
end
BeliefPropagation.BPFactor(f::BPFactor, states) = TabulatedBPFactor(f, states)

"""
    make_generic(bp::BP)

Return the corresponding `BP` with plain `TabulatedBPFactor` as factors. Used to test specific implementations.
"""
function make_generic(bp::BP)
    (; g, ψ, ϕ) = bp
    ψ_generic = [BPFactor(ψ[a], nstates(bp, i) for i in neighbors(g, factor(a))) for a in factors(g)]
    ϕ_generic = [BPFactor(ϕ[i], (nstates(bp, i),)) for i in variables(g)]
    return BP(g, ψ_generic, [nstates(bp, i) for i in variables(g)]; ϕ = ϕ_generic)
end

function eachstate(bp::BP)
    return Iterators.product((1:nstates(bp, i) for i in variables(bp.g))...)
end

nstatestot(bp::BP) = prod(nstates(bp, i) for i in variables(bp.g); init=1.0)

"""
    exact_normalization(bp::BP)

Exhaustively compute the normalization constant.
"""
function exact_normalization(bp::BP)
    nstatestot(bp) > 10^10 && @warn "Exhaustive computations on a system of this size can take quite a while."
    return sum(evaluate(bp, x) for x in eachstate(bp))
end

"""
    exact_prob(bp::BP; Z = exact_normalization(bp))

Exhaustively compute the probability of each possible configuration of the variables.
"""
function exact_prob(bp::BP; Z = exact_normalization(bp))
    p = [evaluate(bp, x) / Z for x in eachstate(bp)]
    return p
end

"""
    exact_marginals(bp::BP; p_exact = exact_prob(bp))

Exhaustively compute marginal distributions for each variable.
"""
function exact_marginals(bp::BP; p_exact = exact_prob(bp))
    dims = 1:ndims(p_exact)
    return map(variables(bp.g)) do i
        vec(sum(p_exact; dims=dims[Not(i)]))
    end
end

"""
    exact_factor_marginals(bp::BP; p_exact = exact_prob(bp))

Exhaustively compute marginal distributions for each factor.
"""
function exact_factor_marginals(bp::BP; p_exact = exact_prob(bp))
    dims = 1:ndims(p_exact)
    return map(factors(bp.g)) do a
        ∂a = neighbors(bp.g, factor(a))
        dropdims(sum(p_exact; dims=dims[Not(∂a)]); dims=tuple(dims[Not(∂a)]...))
    end
end

"""
    exact_avg_energy(bp::BP; p_exact = exact_prob(bp))

Exhaustively compute the average energy (minus the log of the unnormalized probability weight).
"""
function exact_avg_energy(bp::BP; p_exact = exact_prob(bp))
    k = keys(p_exact)
    e = 0.0
    for x in eachindex(p_exact)
        ex = energy(bp, Tuple(k[x]))
        p = p_exact[x]
        if ex == Inf
            if p == 0
                nothing
            else
                error("Configuration with nonzero probability is giving infinite energy")
            end
        else
            e += ex * p
        end
    end
    return e
end

"""
    exact_minimum_energy(bp::BP)

Exhaustively compute the minimum energy (minus the log of the unnormalized probability weight).
"""
function exact_minimum_energy(bp::BP)
    nstatestot(bp) > 10^10 && @warn "Exhaustive computations on a system of this size can take quite a while."
    return minimum(energy(bp, x) for x in eachstate(bp))
end

"""
    rand_factor([rng,], states)

Return a random `BPFactor` whose domain is specified by the iterable `states`.
"""
function rand_factor(rng::AbstractRNG, states)
    isempty(states) && return BPFactor(zeros(0))
    values = rand(rng, states...)
    return BPFactor(values)
end
rand_factor(states) = rand_factor(default_rng(), states)

"""
    rand_bp([rng], g::FactorGraph, states)

Return a `BP` with random factors.

`states` is an iterable containing the number of values that can be taken by each variable.
"""
function rand_bp(rng::AbstractRNG, g::AbstractFactorGraph, states)
    ψ = [rand_factor(rng, [states[i] for i in neighbors(g,factor(a))]) for a in factors(g)] 
    return BP(g, ψ, states)  
end
rand_bp(g::AbstractFactorGraph, qs) = rand_bp(default_rng(), g, qs)

"""
    test_observables_bp(bp::BP; kwargs...)

Test `beliefs_bp`, `factor_beliefs_bp` and `bethe_free_energy_bp` against the same quantities computed exactly by exhaustive enumeration.
"""
function test_observables_bp(bp::BP; kwargs...)
    @test isapprox(beliefs_bp(bp), exact_marginals(bp); kwargs...)
    @test isapprox(factor_beliefs_bp(bp), exact_factor_marginals(bp); kwargs...)
    @test isapprox(exp(-bethe_free_energy_bp(bp)), exact_normalization(bp); kwargs...)
    @test isapprox(bethe_free_energy_bp(bp), BeliefPropagation.bethe_free_energy_bp_beliefs(bp); kwargs...)
    return nothing
end

"""
    test_observables_bp_generic(bp::BP, bp_generic::BP; kwargs...)

Test `beliefs_bp`, `factor_beliefs_bp` and `bethe_free_energy_bp` against the same quantities on a generic version.
See also [`make_generic`](@ref)
"""
function test_observables_bp_generic(bp::BP, bp_generic::BP; kwargs...)
    @test isapprox(beliefs_bp(bp), beliefs_bp(bp_generic); kwargs...)
    @test isapprox(factor_beliefs_bp(bp), factor_beliefs_bp(bp_generic); kwargs...)
    @test isapprox(bethe_free_energy_bp(bp), bethe_free_energy_bp(bp); kwargs...)
    @test isapprox(BeliefPropagation.bethe_free_energy_bp_beliefs(bp), 
        BeliefPropagation.bethe_free_energy_bp_beliefs(bp); kwargs...)
    return nothing
end

# factor update without autodiff
function update_f_bp_old!(bp::BP{F,FV,M,MB}, a::Integer, unew, damp::Real; extra_kwargs...) where {
            F<:BPFactor, FV<:BPFactor, M<:AbstractVector{<:Real}, MB<:AbstractVector{<:Real}}
    (; g, ψ, h) = bp
    ∂a = neighbors(g, factor(a))
    ea = edge_indices(g, factor(a))
    ψₐ = ψ[a]
    for ai in ea
        unew[ai] .= 0
    end
    # zₐ = zero(eltype(bp))
    for xₐ in Iterators.product((1:nstates(bp, i) for i in ∂a)...)
        for (i, ai) in pairs(ea)
            unew[ai][xₐ[i]] += ψₐ(xₐ) *
                prod(h[ja][xₐ[j]] for (j, ja) in pairs(ea) if j != i; init=1.0)
        end
        # zₐ += ψₐ(xₐ) * prod(h[ia][xₐ[i]] for (i, ia) in pairs(ea); init=1.0)
    end
    err = BeliefPropagation.set_messages_factor!(bp, ea, unew, damp)
    return err
end

function _rand_msgs(states)
    m = map(states) do q
        m = rand(q)
        m / sum(m)
    end
    return collect(m)
end

"""
    test_za(bp::BP)

Test a specific implementation of `compute_za` against the naive one.
"""
function test_za(bp::BP)
    (; g) = bp
    bp_generic = make_generic(bp)
    eq = map(factors(g)) do a
        states = (nstates(bp,i) for i in neighbors(g, factor(a)))
        msg_in = _rand_msgs(states)
        za = BeliefPropagation.compute_za(bp, a, msg_in)
        za_generic = BeliefPropagation.compute_za(bp_generic, a, msg_in)
        za ≈ za_generic
    end |> all
    return @test all(eq)
end

end # module