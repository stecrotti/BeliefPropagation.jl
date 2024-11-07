"""
    BPFactor

An abstract type representing a factor.
"""
abstract type BPFactor end

Base.eltype(f::BPFactor) = typeof(float(1.0))

"""
    UniformFactor

A type of `BPFactor` which returns the same value for any input: it behaves as if it wasn't even there.
It is used as the default for single-variable factors
"""
struct UniformFactor <: BPFactor; end
(f::UniformFactor)(x) = 1

"""
    TabulatedBPFactor

A type of `BPFactor` constructed by specifying the output to any input in a tabular fashion via an array `values`.
"""
struct TabulatedBPFactor{T<:Real,N} <: BPFactor
    values :: Array{T,N}
    function TabulatedBPFactor(values::Array{T,N}) where {T<:Real,N}
        any(<(0), values) && throw(ArgumentError("Factors can only take non-negative values"))
        return new{T,N}(values)
    end
end

function (f::TabulatedBPFactor)(x) 
    isempty(x) && return one(eltype(f.values))
    return f.values[x...]
end

Base.eltype(f::TabulatedBPFactor) = eltype(f.values)

# default constructors for `BPFactor`
BPFactor(values::Array{T,N}) where {T<:Real,N} = TabulatedBPFactor(values)