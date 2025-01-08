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

const BPKSAT = BP{<:KSATClause, <:BPFactor, <:NTuple{2,<:Real}, <:NTuple{2,<:Real}}


@doc raw"""
    fast_ksat_bp(g::AbstractFactorGraph, ψ::Vector{<:KSATClause}, [ϕ])

Return a specialized BP instance with `KSATClause` and messages encoded as tuples of two reals instead of vectors. 
```
"""
function fast_ksat_bp(g::AbstractFactorGraph, ψ::Vector{<:KSATClause},
        ϕ::Vector{<:BPFactor}=[BPFactor(ones(2)) for _ in 1:nvariables(g)])
    T = promote_type(eltype(ψ[1]), eltype(ϕ[1]))
    all(eltype(ψₐ) == eltype(ψ[1]) for ψₐ in ψ) || @warn "Possible type issues. Check that all the factors in ψ have the same type"
    all(eltype(ϕᵢ) == eltype(ϕ[1]) for ϕᵢ in ϕ) || @warn "Possible type issues. Check that all the factors in ϕ have the same type"
    u = fill((T(0.5), T(0.5)), ne(g))
    h = fill((T(0.5), T(0.5)), ne(g))
    b = fill((T(0.5), T(0.5)), nvariables(g))
    return BP(g, ψ, ϕ, u, h, b)
end

BeliefPropagation.nstates(bp::BPKSAT, ::Integer) = 2

Base.eltype(bp::BPKSAT) = eltype(eltype(eltype(bp.b)))

function BeliefPropagation.reset!(bp::BPKSAT)
    (; u, h, b) = bp
    T = eltype(bp)
    u .= ((T(0.5), T(0.5)),)
    h .= ((T(0.5), T(0.5)),)
    b .= ((T(0.5), T(0.5)),)
    return nothing
end
function BeliefPropagation.randomize!(rng::AbstractRNG, bp::BPKSAT)
    (; u, h, b) = bp
    T = eltype(bp)
    for ia in eachindex(u)
        ru = rand(rng, T); rh = rand(rng, T)
        u[ia] = (ru, 1-ru)
        h[ia] = (rh, 1-rh)
    end
    b .= ((T(0.5), T(0.5)),)
    return nothing
end

function BeliefPropagation.set_messages_variable!(bp::BPKSAT, ei, i, hnew, bnew, damp)
    (; h, b) = bp
    T = eltype(bp)
    zᵢ = sum(bnew[i])
    # there can be cases where bnew[i] is all zeros -> do not normalize
    if zᵢ != 0
        bnew[i] = bnew[i] ./ zᵢ
    end
    errb = maximum(abs, bnew[i] .- b[i])
    b[i] = bnew[i]
    errv = zero(eltype(bp))
    for ia in ei
        zᵢ₂ₐ = sum(hnew[ia])
        # there can be cases where hnew[i] is all zeros -> do not normalize
        if zᵢ != 0
            hnew[ia] = hnew[ia] ./ zᵢ₂ₐ
        end
        errv = max(errv, abs(h[ia][1] - hnew[ia][1]))
        h[ia] = damp!(h[ia], hnew[ia], damp)
    end
    return errv, errb
end

function BeliefPropagation.update_v_bp!(bp::BPKSAT, i::Integer, hnew, bnew, damp::Real, rein::Real;
        extra_kwargs...)
    (; g, ϕ, u, b) = bp
    ei = edge_indices(g, variable(i)) 
    ϕᵢ = (ϕ[i](1) * b[i][1]^rein, ϕ[i](2) * b[i][2]^rein)
    bnew[i] = @views cavity!(hnew[ei], u[ei], .*, ϕᵢ)
    errv, errb = set_messages_variable!(bp, ei, i, hnew, bnew, damp)
    return errv, errb
end

function BeliefPropagation.set_messages_factor!(bp::BPKSAT, ea, unew, damp)
    u = bp.u
    err = zero(eltype(bp))
    @inbounds for ai in ea
        unew[ai] = unew[ai] ./ sum(unew[ai])
        err = max(err, abs(unew[ai][1] - u[ai][1]))
        u[ai] = damp!(u[ai], unew[ai], damp)
    end
    return err
end

function BeliefPropagation.update_f_bp!(bp::BPKSAT, a::Integer, unew, damp::Real;
        extra_kwargs...)
    (; g, ψ, h) = bp
    ea = edge_indices(g, factor(a))
    Jₐ = ψ[a].J
    htemp = one(eltype(bp))
    @inbounds for (Jα, ia) in zip(Jₐ, ea)
        unew[ia] = (htemp, htemp)
        htemp *= h[ia][Jα+1]
    end
    htemp = one(eltype(bp))
    @inbounds for (α, ia) in Iterators.reverse(enumerate(ea))
        unew[ia] = unew[ia] .* (htemp, htemp)
        htemp *= h[ia][Jₐ[α]+1]
    end
    @inbounds for (α, ia) in enumerate(ea)
        prodh = unew[ia][1]
        u0 = (1 - prodh*(1-Jₐ[α])) / (2-prodh)
        u1 = (1 - prodh*Jₐ[α]) / (2-prodh)
        unew[ia] = (u0, u1)   
    end

    err = set_messages_factor!(bp, ea, unew, damp)
    return err
end

function BeliefPropagation.beliefs_bp(bp::BPKSAT)
    return map(bp.b) do bᵢ
        collect(bᵢ)
    end
end