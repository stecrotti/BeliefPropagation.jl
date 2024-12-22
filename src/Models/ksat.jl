@doc raw"""
    KSATClause

A type of [`BPFactor`](@ref) representing a clause in a k-SAT formula.
It involves $\{0,1\}$ variables $\boldsymbol{x}_a=\{x_1, x_2, \ldots, x_k\}$.
The factor evaluates to
``\psi_a(\boldsymbol{x}_a)=1 - \prod_{i\in a}\left[1-\delta(x_i, J^a_{i})\right]``.

Fields
========

- `J`: a vector of booleans.
"""
struct KSATClause{T}  <: BPFactor where {T<:AbstractVector{<:Bool}}
    J :: T  # J = 1 if x appears negated, J = 0 otherwise (as in mezard montanari)
end

function (f::KSATClause)(x) 
    isempty(x) && return 1.0
    return any(xᵢ - 1 != Jₐᵢ for (xᵢ, Jₐᵢ) in zip(x, f.J)) |> float
end

function BeliefPropagation.compute_za(bp::BP{<:KSATClause}, a::Integer, 
        msg_in::AbstractVector{<:AbstractVector{<:Real}})
    ψₐ = bp.ψ[a]
    isempty(msg_in) && return one(eltype(ψₐ))
    z1 = prod(sum(hᵢₐ) for (hᵢₐ, Jₐᵢ) in zip(msg_in, ψₐ.J))
    z2 = prod(hᵢₐ[Jₐᵢ+1] for (hᵢₐ, Jₐᵢ) in zip(msg_in, ψₐ.J))
    return z1 - z2
end