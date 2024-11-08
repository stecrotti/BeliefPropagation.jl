@doc raw"""
    ColoringCoupling

A type of [`BPFactor`](@ref) representing a factor in the coloring problem.
It always involves two (discrete) incident variables $x_i$, $x_j$.
The factor evaluates to
``\psi(x_i,x_j)=1-\delta(x_i,x_j)``
"""
struct ColoringCoupling  <: BPFactor; end

function (f::ColoringCoupling)(x)
    length(x) == 2 || throw(ArgumentError("ColoringCoupling is defined for 2 neighbors, got $x of length $(length(x))."))
    return float(x[1] != x[2])
end

function BeliefPropagation.compute_za(ψₐ::ColoringCoupling, 
        msg_in::AbstractVector{<:AbstractVector{<:Real}})
    length(msg_in) == 2 || throw(ArgumentError("ColoringCoupling is defined for 2 neighbors, got $(length(msg_in)) incoming messages."))
    z1 = prod(sum.(msg_in))
    z2 = sum(reduce(.*, msg_in))
    return z1 - z2
end

Base.eltype(f::ColoringCoupling) = typeof(f((1,1)))

@doc raw"""
    SoftColoringCoupling

A soft version of [`ColoringCoupling`](@ref).
It always involves two (discrete) incident variables $x_i$, $x_j$. The factor evaluates to
``\psi(x_i,x_j)=e^{-\beta\delta(x_i,x_j)}``

Fields
========

- `β`: the real parameter controlling the softness. A [`ColoringCoupling`](@ref) is recovered in the large β limit.
"""
struct SoftColoringCoupling{T<:Real} <: BPFactor
    β :: T

    function SoftColoringCoupling(β::T) where {T<:Real}
        β > 0 || throw(ArgumentError("Parameter β must be positive, got $β"))
        return new{T}(β)
    end
end

function (f::SoftColoringCoupling)(x)
    length(x) == 2 || throw(ArgumentError("SoftColoringCoupling is defined for 2 neighbors, got $x of length $(length(x))."))
    e = float(x[1] == x[2])
    return exp(-f.β * e)
end

function BeliefPropagation.compute_za(ψₐ::SoftColoringCoupling, 
        msg_in::AbstractVector{<:AbstractVector{<:Real}})
    length(msg_in) == 2 || throw(ArgumentError("SoftColoringCoupling is defined for 2 neighbors, got $(length(msg_in)) incoming messages."))
    z1 = prod(sum.(msg_in))
    z2 = (1 - exp(-ψₐ.β)) * sum(reduce(.*, msg_in))
    return z1 - z2
end

Base.eltype(f::SoftColoringCoupling) = typeof(f((1,1)))