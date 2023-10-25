"""
    BPFactor

An abstract type representing a factor.
"""
abstract type BPFactor end

"""
    UniformFactor

A type of `BPFactor` which returns the same value for any input: it behaves as if it wasn't even there.
It is used as the default for single-variable factors
"""
struct UniformFactor <: BPFactor; end
(f::UniformFactor)(x) = 1

# stores in an array `values` the result of evaluating the factor at all possible inputs 
"""
    TabulatedBPFactor

A type of `BPFactor` constructed by specifying the output to any input in a tabular fashion via an array `values`.
"""
struct TabulatedBPFactor{T,N} <: BPFactor
    values :: Array{T,N}
end

function (f::TabulatedBPFactor)(x) 
    isempty(x) && return 1.0
    return f.values[x...]
end

"""
    TabulatedBPFactor(f::BPFactor, states)

Construct a `TabulatedBPFactor` out of any `BPFactor`. Used mostly for tests.
"""
function TabulatedBPFactor(f::BPFactor, states)
    values = [f(x) for x in Iterators.product((1:q for q in states)...)]
    return TabulatedBPFactor(values)
end

# default constructors for `BPFactor`
BPFactor(values) = TabulatedBPFactor(values)
BPFactor(f::BPFactor, states) = TabulatedBPFactor(f, states)

"""
    rand_factor([rng,], states)

Return a random `BPFactor` whose domain is specified by the iterable `states`.

Examples
========

Create a random factor connected to three variables x₁ ∈ {1,2}, x₂ ∈ {1,2,3,4}, x₃ ∈ {1,2,3}.
```jldoctest random_factor
julia> using BeliefPropagation, BeliefPropagation.FactorGraphs

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