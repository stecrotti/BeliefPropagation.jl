abstract type AbstractBPFactor end
abstract type BPFactor <: AbstractBPFactor end
abstract type VertexBPFactor <: AbstractBPFactor end

# uniform external field on single variable taking one of `q` values. used as default.
struct UniformVertexFactor{T<:Integer} <: VertexBPFactor
    q :: T
end
(f::UniformVertexFactor)(::Integer) = 1 / f.q

# stores in an array `values` the result of evaluating the factor at all possible inputs 
struct TabulatedBPFactor{N,F} <: BPFactor
    values :: Array{N,F}
end

(f::TabulatedBPFactor)(x) = f.values(x...)

# for tests: convert any factor into a `TabulatedBPFactor`
function TabulatedBPFactor(f::AbstractBPFactor, states)
    values = [f(x...) for x in Iterators.product((1:q for q in states))]
    return TabulatedBPFactor(values)
end