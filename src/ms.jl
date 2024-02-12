struct MS{T} <: Real
    val::T
    MS(v::T) where {T<:Union{Float64, Int}} = new{T}(v)
    MS{T}(v::T) where {T<:Union{Float64, Int}} = new{T}(v)
end

Base.:(+)(x::MS, y::Union{Float64, Int, MS}) = MS(max(x.val, log(y)))
Base.:(+)(y::Union{Float64, Int}, x::MS) = x + y
Base.:(*)(x::MS, y::Union{Float64, Int, MS}) = MS(x.val + log(y))
Base.:(*)(y::Union{Float64, Int}, x::MS) = x * y
Base.:(/)(x::MS, y::Union{Float64, Int, MS}) = MS(x.val - log(y))
Base.:(-)(x::MS, y::MS) = x.val - y.val
Base.:(^)(x::MS, y::Float64) = MS(x.val * y)

Base.:(<)(x::MS, y::MS) = x.val < y.val
Base.log(y::MS) = y.val
Base.one(::Type{MS{T}}) where {T<:Union{Float64, Int}} = MS(zero(T))
Base.zero(::Type{MS{T}}) where {T<:Union{Float64, Int}} = MS(typemin(T))