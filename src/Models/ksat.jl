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

```
"""
function fast_ksat_bp(g::AbstractFactorGraph, ψ::Vector{<:KSATClause},
        ϕ::Vector{<:BPFactor}=fill(UniformFactor(), nvariables(g)))
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
    u .= (T(0.5), T(0.5))
    h .= (T(0.5), T(0.5))
    b .= (T(0.5), T(0.5))
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
    b .= (T(0.5), T(0.5))
    return nothing
end

function BeliefPropagation.set_messages_variable!(bp::BPKSAT, ei, i, hnew, bnew, damp)
    (; h, b) = bp
    zᵢ = sum(bnew[i])
    bnew[i] = bnew[i] ./ zᵢ
    errb = maximum(abs, bnew[i] .- b[i])
    b[i] = bnew[i]
    errv = zero(eltype(bp))
    for ia in ei
        hnew[ia] = hnew[ia] ./ sum(hnew[ia])
        errv = max(errv, maximum(abs, hnew[ia] .- h[ia]))
        h[ia] = damp!(h[ia], hnew[ia], damp)
    end
    return errv, errb
end

function BeliefPropagation.update_v_bp!(bp::BPKSAT, i::Integer, hnew, bnew, damp::Real, rein::Real;
        extra_kwargs...)
    (; g, ϕ, u, b) = bp
    ei = edge_indices(g, variable(i)) 
    ϕᵢ = ntuple(x -> ϕ[i](x) * b[i][x]^rein, nstates(bp, i))
    bnew[i] = @views cavity!(hnew[ei], u[ei], .*, ϕᵢ)
    errv, errb = set_messages_variable!(bp, ei, i, hnew, bnew, damp)
    return errv, errb
end

function BeliefPropagation.set_messages_factor!(bp::BPKSAT, ea, unew, damp)
    u = bp.u
    err = zero(eltype(bp))
    for ai in ea
        unew[ai] = unew[ai] ./ sum(unew[ai])
        err = max(err, maximum(abs, unew[ai] .- u[ai]))
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
    for (α, ia) in enumerate(ea)
        unew[ia] = (htemp, htemp)
        htemp *= h[ia][Jₐ[α]+1]
    end
    htemp = one(eltype(bp))
    for (α, ia) in Iterators.reverse(enumerate(ea))
        unew[ia] = unew[ia] .* (htemp, htemp)
        htemp *= h[ia][Jₐ[α]+1]
    end
    for (α, ia) in enumerate(ea)
        x = unew[ia][1]
        y = (1 - x*(1-Jₐ[α])) / (2-x)
        unew[ia] = (y, 1-y)
    end

    err = set_messages_factor!(bp, ea, unew, damp)
    return err
end

function BeliefPropagation.beliefs_bp(bp::BPKSAT)
    return map(bp.b) do bᵢ
        collect(bᵢ)
    end
end


# @doc raw"""
#     fast_ksat_bp(g::AbstractFactorGraph, ψ::Vector{<:KSATClause}, [ϕ])

# ```
# """
# function fast_ksat_bp(g::AbstractFactorGraph, ψ::Vector{<:KSATClause},
#         ϕ::Vector{<:BPFactor}=fill(UniformFactor(), nvariables(g)))
#     T = promote_type(eltype(ψ[1]), eltype(ϕ[1]))
#     all(eltype(ψₐ) == eltype(ψ[1]) for ψₐ in ψ) || @warn "Possible type issues. Check that all the factors in ψ have the same type"
#     all(eltype(ϕᵢ) == eltype(ϕ[1]) for ϕᵢ in ϕ) || @warn "Possible type issues. Check that all the factors in ϕ have the same type"

#     u = reduce(vcat, map(ψ) do ψₐ
#         [(Jₐ, T(0.5)) for Jₐ in ψₐ.J]
#     end)
#     h = deepcopy(u)
#     b = fill((T(0.5), T(0.5)), nvariables(g))
#     return BP(g, ψ, ϕ, u, h, b)
# end

# const BPKSAT = BP{<:KSATClause, <:BPFactor, <:Tuple{Bool,<:Real}, <:NTuple{2,<:Real}}

# BeliefPropagation.nstates(bp::BPKSAT, ::Integer) = 2

# Base.eltype(bp::BPKSAT) = eltype(eltype(eltype(bp.b)))

# function BeliefPropagation.reset!(bp::BPKSAT)
#     (; g, ψ, u, h, b) = bp
#     T = eltype(bp)
#     for a in factors(g)
#         for (α, ia) in enumerate(edge_indices(g, factor(a)))
#             u[ia] = h[ia] = (ψ[a].J[α], T(0.5))
#         end
#     end
#     b .= (T(0.5), T(0.5))
#     return nothing
# end
# function BeliefPropagation.randomize!(rng::AbstractRNG, bp::BPKSAT)
#     (; g, ψ, u, h, b) = bp
#     T = eltype(bp)
#     for a in factors(g)
#         for (α, ia) in enumerate(edge_indices(g, factor(a)))
#             u[ia] = h[ia] = (ψ[a].J[α], rand(rng, T))
#         end
#     end
#     b .= (T(0.5), T(0.5))
#     return nothing
# end

# function _op_ziSAT((J1, u1), (J2, u2))
#     # u3 = u1 * ((1-u2)/u2)^J2 * (u2/(1-u2))^(1-J2)
#     u3 = u1 * (J2 ? (1-u2)/u2 : u2/(1-u2))
#     return (false, u3)
# end

# function BeliefPropagation.update_v_bp!(bp::BPKSAT, i::Integer, hnew, bnew, damp::Real, rein::Real;
#         extra_kwargs...)
#     (; g, ϕ, u, h, b) = bp
#     ei = edge_indices(g, variable(i)) 

#     bnew_ = @views cavity!(hnew[ei], u[ei], _op_ziSAT, (false, one(eltype(bp))))
#     ϕᵢ = ϕ[i](1) * b[i][1]^rein, ϕ[i](2) * b[i][2]^rein
#     errv = -Inf
#     @show hnew[ei]
#     for ia in ei
#         J = h[ia][1]
#         h_ = 1 / 1 + (ϕᵢ[2-J] / ϕᵢ[J+1] * hnew[ia][2])
#         hnew[ia] = J ? (J, h_) : (J, 1-h_)
#         errv = max(errv, abs(hnew[ia][2] - h[ia][2]))
#         h[ia] = damp!(h[ia], hnew[ia], damp)
#     end
#     bnew1 = 1 / 1 + (ϕᵢ[2] / ϕᵢ[1] * bnew_[2])
#     bnew[i] = (1 - bnew1, bnew1)
#     errb = abs(bnew1 - b[i][2])
#     return errv, errb
# end

# function BeliefPropagation.update_f_bp!(bp::BPKSAT, a::Integer,
#         unew, damp::Real; extra_kwargs...)
#     (; g, ψ, u, h) = bp
#     ea = edge_indices(g, factor(a))

#     f((J1, h1), (J2, h2)) = (false, h1*(1-h2))
#     @views cavity!(unew[ea], h[ea], f, (false, one(eltype(bp))))
#     errf = -Inf
#     for ai in ea
#         unew[ai] = (u[ai][1], 1 / (2 - unew[ai][2]))
#         errf = max(errf, abs(unew[ai][2] - u[ai][2]))
#         u[ai] = damp!(u[ai], unew[ai], damp)
#     end
#     return errf
# end

# function BeliefPropagation.beliefs_bp(bp::BPKSAT)
#     return map(bp.b) do bᵢ
#         collect(bᵢ)
#     end
# end

# function BeliefPropagation.factor_beliefs_bp(bp::BPKSAT)
#     (; g, ψ, h) = bp
#     return map(factors(g)) do a
#         ψₐ = ψ[a]
#         ∂a = neighbors(g, factor(a))
#         bₐ = map(Iterators.product((1:nstates(bp, i) for i in ∂a)...)) do xₐ
#             ψₐ(xₐ) * prod(xₐ[i] == 2 ? h[ia] : 1-h[ia] 
#                 for (i, ia) in pairs(edge_indices(g, factor(a))); init=one(eltype(bp)))
#         end
#         zₐ = sum(bₐ)
#         bₐ ./= zₐ
#         bₐ
#     end
# end