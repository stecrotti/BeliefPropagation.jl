# J = 1 if x appears negated, J = 0 otherwise (as in mezard montanari)
struct KSATClause{T}  <: BPFactor where {T<:AbstractVector{<:Bool}}
    J :: T 
end

Base.eltype(::KSATClause) = eltype(1.0)

function (f::KSATClause)(x) 
    isempty(x) && return 1.0
    return any(xᵢ - 1 != Jₐᵢ for (xᵢ, Jₐᵢ) in zip(x, f.J)) |> float
end

function BeliefPropagation.compute_za(ψₐ::KSATClause, 
        msg_in::AbstractVector{<:AbstractVector{<:Real}})
    isempty(msg_in) && return one(eltype(ψₐ))
    z1 = prod(sum(hᵢₐ) for (hᵢₐ, Jₐᵢ) in zip(msg_in, ψₐ.J))
    z2 = prod(hᵢₐ[Jₐᵢ+1] for (hᵢₐ, Jₐᵢ) in zip(msg_in, ψₐ.J))
    return z1 - z2
end