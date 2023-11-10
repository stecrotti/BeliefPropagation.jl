module Test

using BeliefPropagation: BPFactor, TabulatedBPFactor, BP
using BeliefPropagation: beliefs_bp, factor_beliefs_bp, bethe_free_energy_bp
using BeliefPropagation
using BeliefPropagation.FactorGraphs: AbstractFactorGraph, FactorGraph, RegularFactorGraph,
    factor, variable, factors, variables, neighbors

using Test: @test
using Random: AbstractRNG, default_rng
using InvertedIndices: Not

export exact_normalization, exact_prob, exact_marginals, exact_factor_marginals,
    exact_avg_energy, exact_minimum_energy
export rand_factor, rand_bp
export test_observables_bp

"""
    TabulatedBPFactor(f::BPFactor, states)

Construct a `TabulatedBPFactor` out of any `BPFactor`. Used mostly for tests.
"""
function BeliefPropagation.TabulatedBPFactor(f::BPFactor, states)
    values = [f(x) for x in Iterators.product((1:q for q in states)...)]
    return TabulatedBPFactor(values)
end
BeliefPropagation.BPFactor(f::BPFactor, states) = TabulatedBPFactor(f, states)

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
    sum(energy(bp, Tuple(k[x])) * p_exact[x] for x in eachindex(p_exact))
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

Examples
========

Create a random factor connected to three variables ``x_1 \\in \\{1,2\\}, x_2 \\in \\{1,2,3,4\\}, x_3 \\in \\{1,2,3\\}``.

```jldoctest rand_factor
julia> using BeliefPropagation.Test

julia> import Random: MersenneTwister

julia> states = (2, 4, 3);

julia> f = rand_factor(MersenneTwister(0), states);

julia> f([1, 4, 2])
0.16703619444214968

```
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
function rand_bp(rng::AbstractRNG, g::FactorGraph, states)
    ψ = [rand_factor(rng, [states[i] for i in neighbors(g,factor(a))]) for a in factors(g)] 
    return BP(g, ψ, states)  
end
function rand_bp(rng::AbstractRNG, g::RegularFactorGraph, states)
    ψ = [rand_factor(rng, fill(only(states), degree(g, factor(1))))] 
    return BP(g, ψ, q)  
end
rand_bp(g::AbstractFactorGraph, qs) = rand_bp(default_rng(), g, qs)

"""
    test_observables_bp(bp::BP

Test `beliefs_bp`, `factor_beliefs_bp` and `bethe_free_energy_bp` against the same quantities computed exactly by exhaustive enumeration.
"""
function test_observables_bp(bp::BP; kwargs...)
    @test isapprox(beliefs_bp(bp), exact_marginals(bp); kwargs...)
    @test isapprox(factor_beliefs_bp(bp), exact_factor_marginals(bp); kwargs...)
    @test isapprox(exp(-bethe_free_energy_bp(bp)), exact_normalization(bp); kwargs...)
end

function update_f_bp_old!(bp::BP{F,FV,M,MB}, a::Integer, unew, damp::Real,
        f::BeliefPropagation.BetheFreeEnergy; extra_kwargs...) where {
            F<:BPFactor, FV<:BPFactor, M<:AbstractVector{<:Real}, MB<:AbstractVector{<:Real}}
    (; g, ψ, h) = bp
    ∂a = neighbors(g, factor(a))
    ea = BeliefPropagation.FactorGraphs.edge_indices(g, factor(a))
    ψₐ = ψ[a]
    for ai in ea
        unew[ai] .= 0
    end
    zₐ = zero(eltype(bp))
    for xₐ in Iterators.product((1:nstates(bp, i) for i in ∂a)...)
        for (i, ai) in pairs(ea)
            unew[ai][xₐ[i]] += ψₐ(xₐ) *
                prod(h[ja][xₐ[j]] for (j, ja) in pairs(ea) if j != i; init=1.0)
        end
        zₐ += ψₐ(xₐ) * prod(h[ia][xₐ[i]] for (i, ia) in pairs(ea); init=1.0)
    end
    f.factors[a] -= log(zₐ)
    err = BeliefPropagation.set_messages_factor!(bp, ea, unew, damp)
    return err
end

end # module