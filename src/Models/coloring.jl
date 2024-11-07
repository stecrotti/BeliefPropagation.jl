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

struct SoftColoringCoupling{T<:Real} <: BPFactor
    β :: T
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