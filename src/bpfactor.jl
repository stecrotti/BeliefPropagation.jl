abstract type AbstractBPFactor end
abstract type BPFactor <: AbstractBPFactor end

# uniform external field on single variable taking one of `N` values. 
# used as default for vertex factors.
struct UniformFactor{T<:Integer} <: BPFactor
    N :: T
end
(f::UniformFactor)(::Integer) = 1 / f.N

# stores in an array `values` the result of evaluating the factor at all possible inputs 
struct TabulatedBPFactor{N,F} <: BPFactor
    values :: Array{N,F}
end

function (f::TabulatedBPFactor)(x) 
    isempty(x) && return 1.0
    return f.values[x...]
end

# for tests: convert any factor into a `TabulatedBPFactor`
# `states` is an iterable with integers
function TabulatedBPFactor(f::AbstractBPFactor, states)
    values = [f(x...) for x in Iterators.product((1:q for q in states))]
    return TabulatedBPFactor(values)
end

function random_factor(rng::AbstractRNG, states)
    isempty(states) && return TabulatedBPFactor(zeros(0))
    values = rand(rng, states...)
    return TabulatedBPFactor(values)
end
random_factor(states) = random_factor(GLOBAL_RNG, states)