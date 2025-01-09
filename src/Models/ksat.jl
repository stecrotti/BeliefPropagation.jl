@doc raw"""
    KSATClause

A type of [`BPFactor`](@ref) representing a clause in a k-SAT formula.
It involves $\{0,1\}$ variables $\boldsymbol{x}_a=\{x_1, x_2, \ldots, x_k\}$.
The factor evaluates to
``\psi_a(\boldsymbol{x}_a)=1 - \prod_{i\in a}\delta(x_i, J^a_{i})``.

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
        msg_in = bp.h[edge_indices(bp.g, factor(a))])
    ψₐ = bp.ψ[a]
    isempty(msg_in) && return one(eltype(ψₐ))
    z1 = prod(sum(hᵢₐ) for (hᵢₐ, Jₐᵢ) in zip(msg_in, ψₐ.J))
    z2 = prod(hᵢₐ[Jₐᵢ+1] for (hᵢₐ, Jₐᵢ) in zip(msg_in, ψₐ.J))
    return z1 - z2
end

const BPKSAT = BP{<:KSATClause, <:BPFactor, <:Real, <:Real}


@doc raw"""
    fast_ksat_bp(g::AbstractFactorGraph, ψ::Vector{<:KSATClause}, [ϕ])

Return a specialized BP instance with `KSATClause` and messages encoded as reals instead of vectors. 
```
"""
function fast_ksat_bp(g::AbstractFactorGraph, ψ::Vector{<:KSATClause},
        ϕ::Vector{<:BPFactor}=[BPFactor(ones(2)) for _ in 1:nvariables(g)])
    T = promote_type(eltype(ψ[1]), eltype(ϕ[1]))
    all(eltype(ψₐ) == eltype(ψ[1]) for ψₐ in ψ) || @warn "Possible type issues. Check that all the factors in ψ have the same type"
    all(eltype(ϕᵢ) == eltype(ϕ[1]) for ϕᵢ in ϕ) || @warn "Possible type issues. Check that all the factors in ϕ have the same type"
    init = T(0.5)
    u = fill(init, ne(g))
    h = fill(init, ne(g))
    b = fill(init, nvariables(g))
    return BP(g, ψ, ϕ, u, h, b)
end

BeliefPropagation.nstates(bp::BPKSAT, ::Integer) = 2

Base.eltype(bp::BPKSAT) = eltype(eltype(bp.b))

function BeliefPropagation.reset!(bp::BPKSAT)
    (; u, h, b) = bp
    T = eltype(bp)
    init = T(0.5)
    u .= init
    h .= init
    b .= init
    return nothing
end
function BeliefPropagation.randomize!(rng::AbstractRNG, bp::BPKSAT)
    (; u, h, b) = bp
    rand!(rng, u)
    rand!(rng, h)
    b .= 0
    return nothing
end

function BeliefPropagation.update_v_bp!(bp::BPKSAT, i::Integer, hnew, bnew, damp::Real, rein::Real;
        extra_kwargs...)
    (; g, ϕ, u, h, b) = bp
    ei = edge_indices(g, variable(i)) 
    ϕᵢ = (ϕ[i](1) * (1-b[i])^rein) / (ϕ[i](2) * b[i]^rein)
    u_ = (1/x-1 for x in @view u[ei])
    bnew[i] = @views cavity!(hnew[ei], u_, *, one(eltype(bp)))
    f(x) = 1 / (1 + ϕᵢ*x)
    bnew[i] = f(bnew[i])
    errb = abs(bnew[i] - b[i])
    b[i] = bnew[i]
    errv = zero(eltype(bp))
    @inbounds for ia in ei
        hnew[ia] = f(hnew[ia])
        errv = max(errv, abs(hnew[ia] - h[ia]))
        h[ia] = damping(h[ia], hnew[ia], damp)
    end
    return errv, errb
end

function BeliefPropagation.update_f_bp!(bp::BPKSAT, a::Integer, unew, damp::Real;
        extra_kwargs...)
    (; g, ψ, u, h) = bp
    ea = edge_indices(g, factor(a))
    Jₐ = ψ[a].J
    h_ = (Jₐⁱ ? h[ia] : 1-h[ia] for (ia, Jₐⁱ) in zip(ea, Jₐ))
    @views cavity!(unew[ea], h_, *, one(eltype(bp)))
    errf = zero(eltype(bp))
    @inbounds for (ai, Jₐⁱ) in zip(ea, Jₐ)
        unew[ai] = (1 - Jₐⁱ * unew[ai]) / (2 - unew[ai])
        errf = max(errf, abs(unew[ai] - u[ai]))
        u[ai] = damping(u[ai], unew[ai], damp)
    end
    return errf
end

function BeliefPropagation.beliefs_bp(bp::BPKSAT)
    return map(bp.b) do bᵢ
        [1-bᵢ, bᵢ]
    end
end

function BeliefPropagation.factor_beliefs_bp(bp::BPKSAT)
    (; g, ψ, h) = bp
    return map(factors(g)) do a
        ψₐ = ψ[a]
        ∂a = neighbors(g, factor(a))
        bₐ = map(Iterators.product((1:nstates(bp, i) for i in ∂a)...)) do xₐ
            ψₐ(xₐ) * prod(xₐ[i] == 2 ? h[ia] : 1-h[ia] 
                for (i, ia) in pairs(edge_indices(g, factor(a))); init=one(eltype(bp)))
        end
        zₐ = sum(bₐ)
        bₐ ./= zₐ
        bₐ
    end
end

function BeliefPropagation.compute_zi(bp::BPKSAT, i::Integer, 
        msg_in::AbstractVector{<:Real} = bp.u[edge_indices(bp.g, variable(i))])
    zi0 = bp.ϕ[i](1) * prod(msg_in, init=one(eltype(bp)))
    zi1 = bp.ϕ[i](2) * prod((1 - u for u in msg_in), init=one(eltype(bp)))
    return zi0 + zi1
end

function BeliefPropagation.compute_za(bp::BPKSAT, a::Integer, 
        msg_in::AbstractVector{<:Real} = bp.h[edge_indices(bp.g, factor(a))])
    (; g, ψ, h) = bp
    ea = edge_indices(g, factor(a))
    Jₐ = ψ[a].J
    h_ = (Jₐⁱ ? h[ia] : 1-h[ia] for (ia, Jₐⁱ) in zip(ea, Jₐ))
    return 1 - prod(h_, init=one(eltype(bp)))
end

function BeliefPropagation.compute_zai(bp::BPKSAT, ai::Integer, uai::Real, hia::Real)
    return uai * hia + (1-uai) * (1-hia)
end