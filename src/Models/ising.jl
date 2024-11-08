potts2spin(x) = 3 - 2x

@doc raw"""
    IsingCoupling

A type of [`BPFactor`](@ref) representing a factor in an Ising distribution.
It involves $\pm 1$ variables $\boldsymbol{\sigma}_a=\{\sigma_1, \sigma_2, \ldots\}$.
The factor evaluates to
``\psi(\boldsymbol{\sigma}_a)=e^{\beta J \prod_{i\in a}\sigma_i}``.
A particular case is the pairwise interaction where $a=\{i,j\}$ is a pair of vertices involved in an edge $(ij)$.

Fields
========

- `βJ`: coupling strength.
"""
struct IsingCoupling{T<:Real}  <: BPFactor 
    βJ :: T 
end

(f::IsingCoupling)(x) = isempty(x) ? one(eltype(f)) : exp(f.βJ * prod(potts2spin(xᵢ) for xᵢ in x))


@doc raw"""
    IsingField

A type of [`BPFactor`](@ref) representing a single-variable external field in an Ising distribution.
The factor evaluates to
``\psi(\sigma_i)=e^{\beta h \sigma_i}``.

Fields
========

- `βh`: field strength.
"""
struct IsingField{T<:Real}  <: BPFactor 
    βh :: T 
end

(f::IsingField)(x) = exp(f.βh * potts2spin(only(x)))

# Ising model with xᵢ ∈ {1,2} mapped onto spins {+1,-1}
struct Ising{TJ<:Real, Th<:Real, Tβ<:Real, Tg<:Integer}
    g :: IndexedGraph{Tg}
    J :: Vector{TJ}
    h :: Vector{Th}
    β :: Tβ

    function Ising(g::IndexedGraph{Tg}, J::Vector{TJ}, h::Vector{Th}, β::Tβ=1) where 
            {TJ<:Real, Th<:Real, Tβ<:Real, Tg<:Integer}
        @assert length(J) == ne(g)
        @assert length(h) == nv(g)
        @assert β ≥ 0
        new{TJ, Th, Tβ, Tg}(g, J, h, β)
    end
end

function Ising(J::AbstractMatrix{<:Real}, h::Vector{<:Real}, β::Real=1.0)
    Jvec = [J[i,j] for j in axes(J,2) for i in axes(J,1) if i < j && J[i,j]!=0]
    g = IndexedGraph(Symmetric(J, :L))
    Ising(g, Jvec, h, β)
end

function BeliefPropagation.BP(ising::Ising)
    g = pairwise_interaction_graph(ising.g)
    ψ = [IsingCoupling(ising.β * Jᵢⱼ) for Jᵢⱼ in ising.J]
    ϕ = [IsingField(ising.β * hᵢ) for hᵢ in ising.h]
    qs = fill(2, nvariables(g))
    return BP(g, ψ, qs; ϕ)
end

@doc raw"""
    fast_ising_bp(g::AbstractFactorGraph, ψ::Vector{<:IsingCoupling}, [ϕ])

Return a BP instance with Ising factors and messages and beliefs in log-ratio format:
```math
\begin{align*}
	&m_{a\to i}(\sigma_i) \propto e^{u_{a\to i}\sigma_i}\\
	&u_{a\to i} = \frac12\log\frac{m_{a\to i}(+1)}{m_{a\to i}(+1)}
\end{align*}
```
"""
function fast_ising_bp(g::AbstractFactorGraph, ψ::Vector{<:IsingCoupling},
        ϕ::Vector{<:IsingField}=fill(IsingField(0.0), nvariables(g)))
    T = promote_type(eltype(ψ[1]), eltype(ϕ[1]))
    all(eltype(ψₐ) == eltype(ψ[1]) for ψₐ in ψ) || @warn "Possible type issues. Check that all the factors in ψ have the same type"
    all(eltype(ϕᵢ) == eltype(ϕ[1]) for ϕᵢ in ϕ) || @warn "Possible type issues. Check that all the factors in ϕ have the same type"
    u = zeros(T, ne(g))
    h = zeros(T, ne(g))
    b = zeros(T, nvariables(g))
    return BP(g, ψ, ϕ, u, h, b)
end

function fast_ising_bp(ising::Ising)
    g = pairwise_interaction_graph(ising.g)
    ψ = [IsingCoupling(ising.β * Jᵢⱼ) for Jᵢⱼ in ising.J]
    ϕ = [IsingField(ising.β * hᵢ) for hᵢ in ising.h]
    return fast_ising_bp(g, ψ, ϕ)
end

const BPIsing = BP{<:IsingCoupling, <:IsingField, <:Real, <:Real}

BeliefPropagation.nstates(bp::BPIsing, ::Integer) = 2

Base.eltype(bp::BPIsing) = eltype(eltype(bp.b))

function BeliefPropagation.reset!(bp::BPIsing)
    (; u, h, b) = bp
    u .= 0
    h .= 0
    b .= 0
    return nothing
end

function BeliefPropagation.update_v_bp!(bp::BPIsing,
        i::Integer, hnew, bnew, damp::Real, rein::Real; extra_kwargs...)
    (; g, ϕ, u, h, b) = bp
    ei = edge_indices(g, variable(i)) 
    hᵢ = ϕ[i].βh + b[i]*rein
    bnew[i] = @views cavity!(hnew[ei], u[ei], +, hᵢ)
    errb = abs(bnew[i] - b[i])
    logzᵢ = log(2cosh(bnew[i]) / prod(2cosh(uai) for uai in u[ei]; init=one(eltype(bp))))
    b[i] = bnew[i]
    errv = -Inf
    for ia in ei
        errv = max(errv, abs(hnew[ia] - h[ia]))
        h[ia] = damp!(h[ia], hnew[ia], damp)
    end
    return errv, errb
end

function BeliefPropagation.update_f_bp!(bp::BPIsing, a::Integer,
        unew, damp::Real; extra_kwargs...)
    (; g, ψ, u, h) = bp
    ea = edge_indices(g, factor(a))
    Jₐ = ψ[a].βJ
    @views prodtanh = cavity!(unew[ea], tanh.(h[ea]), *, tanh(Jₐ))
    zₐ = cosh(Jₐ) * (1 + prodtanh)
    unew[ea] .= atanh.(unew[ea])
    err = -Inf
    for ai in ea
        err = max(err, abs(unew[ai] - u[ai]))
        u[ai] = damp!(u[ai], unew[ai], damp)
    end
    return err
end

function BeliefPropagation.beliefs_bp(bp::BPIsing)
    return map(bp.b) do hᵢ
        bᵢ = [exp(hᵢ), exp(-hᵢ)]
        bᵢ ./= sum(bᵢ)
        bᵢ
    end
end

function BeliefPropagation.factor_beliefs_bp(bp::BPIsing)
    (; g, ψ, h) = bp
    return map(factors(g)) do a
        ψₐ = ψ[a]
        ∂a = neighbors(g, factor(a))
        bₐ = map(Iterators.product((1:nstates(bp, i) for i in ∂a)...)) do xₐ
            ψₐ(xₐ) * prod(exp(h[ia]*potts2spin(xₐ[i])) 
                for (i, ia) in pairs(edge_indices(g, factor(a))); init=one(eltype(bp)))
        end
        zₐ = sum(bₐ)
        bₐ ./= zₐ
        bₐ
    end
end

function BeliefPropagation.compute_zi(ϕᵢ::IsingField, msg_in::AbstractVector{T}, q::Integer) where T<:Real
    bnew = sum(msg_in, init=ϕᵢ.βh) 
    return 2cosh(bnew) / prod(2cosh(uai) for uai in msg_in; init=one(T))
end

function BeliefPropagation.compute_za(ψₐ::IsingCoupling, msg_in::AbstractVector{<:Real})
    Jₐ = ψₐ.βJ
    prodtanh = prod(tanh, msg_in, init=tanh(Jₐ))
    return cosh(Jₐ) * (1 + prodtanh)
end

function BeliefPropagation.compute_zai(uai::Real, hia::Real)
    return (1 + tanh(uai)*tanh(hia)) / 2
end

function BeliefPropagation.update_v_ms!(bp::BPIsing,
        i::Integer, hnew, bnew, damp::Real, rein::Real; extra_kwargs...)
    (; g, ϕ, u, h, b) = bp
    ei = edge_indices(g, variable(i)) 
    hᵢ = ϕ[i].βh + b[i]*rein
    bnew[i] = @views cavity!(hnew[ei], u[ei], +, hᵢ)
    errb = abs(bnew[i] - b[i])
    # logzᵢ = abs(bnew[i]) - sum(abs, u[ei]; init=zero(eltype(bp)))
    b[i] = bnew[i]
    errv = -Inf
    for ia in ei
        errv = max(errv, abs(hnew[ia] - h[ia]))
        h[ia] = damp!(h[ia], hnew[ia], damp)
    end
    return errv, errb
end

function BeliefPropagation.update_f_ms!(bp::BPIsing, a::Integer,
        unew, damp::Real; extra_kwargs...)
    (; g, ψ, u, h) = bp
    ea = edge_indices(g, factor(a))
    Jₐ = ψ[a].βJ
    @views minh = cavity!(unew[ea], abs.(h[ea]), min, convert(eltype(h), abs(Jₐ)))
    signs, prodsigns = cavity(sign.(h[ea]), *, sign(Jₐ))
    unew[ea] .= signs .* unew[ea]
    # logzₐ = abs(Jₐ) - 2min(abs(Jₐ), minh)*(prodsigns!=1)
    err = -Inf
    for ai in ea
        err = max(err, abs(unew[ai] - u[ai]))
        u[ai] = damp!(u[ai], unew[ai], damp)
    end
    return err
end

function BeliefPropagation.beliefs_ms(bp::BPIsing)
    return map(bp.b) do hᵢ
        bᵢ = [σᵢ*hᵢ == abs(hᵢ) ? 1.0 : 0.0 for σᵢ in (1, -1)]
        bᵢ ./= sum(bᵢ)
    end
end

function BeliefPropagation.factor_beliefs_ms(bp::BPIsing)
    (; g, ψ, h) = bp
    return map(factors(g)) do a
        ψₐ = ψ[a]
        bₐ = map(Iterators.product((1:nstates(bp, i) for i in neighbors(g, factor(a)))...)) do xₐ
            log(ψₐ(xₐ)) + sum(h[ia]*potts2spin(xₐ[i]) for (i, ia) in pairs(edge_indices(g, factor(a))); init=0.0)
        end
        zₐ = maximum(bₐ)
        bₐ .-= zₐ
        bₐ
    end
end