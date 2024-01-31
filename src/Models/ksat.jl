# J = 1 if x appears negated, J = 0 otherwise (as in mezard montanari)
struct KSATClause{T}  <: BPFactor where {T<:AbstractVector{<:Bool}}
    J :: T 
end

Base.eltype(::KSATClause) = eltype(1.0)

function (f::KSATClause)(x) 
    isempty(x) && return 1.0
    return any(xᵢ - 1 != Jₐᵢ for (xᵢ, Jₐᵢ) in zip(x, f.J)) |> float
end