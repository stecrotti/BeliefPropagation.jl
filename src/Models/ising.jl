potts2spin(x) = 3 - 2x

struct IsingCoupling{T<:Real}  <: BPFactor 
    βJ :: T 
end

function (f::IsingCoupling)(x)
    (; βJ) = f
    return exp(βJ * prod(potts2spin(xᵢ) for xᵢ in x))
end

struct IsingField{T<:Real}  <: BPFactor 
    βh :: T 
end

function (f::IsingField)(x::Integer)
    (; βh) = f
    return exp(βh * potts2spin(x))
end

# Ising model with xᵢ ∈ {1,2} mapped onto spins {+1,-1}
struct Ising{T<:Real}
    g :: IndexedGraph{Int}
    J :: Vector{T}
    h :: Vector{T}
    β :: T

    function Ising(g::IndexedGraph{Int}, J::Vector{T}, h::Vector{T}, β::T=1.0) where {T<:Real}
        @assert length(J) == ne(g)
        @assert length(h) == nv(g)
        @assert β ≥ 0
        new{T}(g, J, h, β)
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

function fast_ising_bp(g::AbstractFactorGraph, ψ::Vector{<:IsingCoupling},
        ϕ::Vector{<:IsingField}=fill(IsingField(0.0), nvariables(g)))
    u = zeros(ne(g))
    h = zeros(ne(g))
    b = zeros(nvariables(g))
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

function BeliefPropagation.reset!(bp::BPIsing)
    (; u, h, b) = bp
    u .= 0
    h .= 0
    b .= 0
    return nothing
end

function BeliefPropagation.update_v_bp!(bp::BPIsing,
        i::Integer, hnew, bnew, damp::Real, rein::Real,
        f::AtomicVector{<:Real}; extra_kwargs...)
    (; g, ϕ, u, h, b) = bp
    ei = edge_indices(g, variable(i)) 
    ∂i = neighbors(g, variable(i))
    hᵢ = ϕ[i].βh + b[i]*rein
    bnew[i] = @views cavity!(hnew[ei], u[ei], +, hᵢ)
    cout, cfull = cavity(2cosh.(u[ei]), *, 1.0)
    d = (degree(g, factor(a)) for a in ∂i)
    errv = -Inf
    for (ia, dₐ, c) in zip(ei, d, cout)
        zᵢ₂ₐ = 2cosh(hnew[ia]) / c
        f[i] -= log(zᵢ₂ₐ) * (1 - 1/dₐ)
        errv = max(errv, abs(hnew[ia] - h[ia]))
        h[ia] = damp!(h[ia], hnew[ia], damp)
    end
    errb = abs(bnew[i] - b[i])
    zᵢ = 2cosh(bnew[i]) / cfull
    f[i] -= log(zᵢ) * (1 - degree(g, variable(i)) + sum(1/dₐ for dₐ in d; init=0.0))
    b[i] = bnew[i]
    return errv, errb
end

function BeliefPropagation.update_f_bp!(bp::BPIsing, a::Integer,
        unew, damp::Real, f::AtomicVector{<:Real}; extra_kwargs...)
    (; g, ψ, u, h) = bp
    ∂a = neighbors(g, factor(a))
    ea = edge_indices(g, factor(a))
    Jₐ = ψ[a].βJ
    @views cavity!(unew[ea], tanh.(h[ea]), *, 1.0)
    unew[ea] .= atanh.(tanh(Jₐ).*unew[ea])
    dₐ = degree(g, factor(a))
    err = -Inf
    for (i, ai) in zip(∂a, ea)
        zₐ₂ᵢ = 2cosh(Jₐ)
        f[i] -= log(zₐ₂ᵢ) / dₐ
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
        bₐ = zeros((nstates(bp, i) for i in neighbors(g, factor(a)))...)
        for xₐ in keys(bₐ)
            bₐ[xₐ] = ψₐ(Tuple(xₐ)) * prod(exp(h[ia]*potts2spin(xₐ[i])) for (i, ia) in pairs(edge_indices(g, factor(a))); init=1.0)
        end
        zₐ = sum(bₐ)
        bₐ ./= zₐ
        bₐ
    end
end

function BeliefPropagation.update_v_ms!(bp::BPIsing,
    i::Integer, hnew, bnew, damp::Real, rein::Real,
    f::AtomicVector{<:Real}; extra_kwargs...)
    (; g, ϕ, u, h, b) = bp
    ei = edge_indices(g, variable(i)) 
    ∂i = neighbors(g, variable(i))
    hᵢ = ϕ[i].βh + b[i]*rein
    bnew[i] = @views cavity!(hnew[ei], u[ei], +, hᵢ)
    cout, cfull = cavity(abs.(u[ei]), +, 0.0)
    d = (degree(g, factor(a)) for a in ∂i)
    errv = -Inf
    for (ia, dₐ, c) in zip(ei, d, cout)
        fᵢ₂ₐ = abs(hnew[ia]) - c
        f[i] -= fᵢ₂ₐ * (1 - 1/dₐ)
        errv = max(errv, abs(hnew[ia] - h[ia]))
        h[ia] = damp!(h[ia], hnew[ia], damp)
    end
    errb = abs(bnew[i] - b[i])
    fᵢ = abs(bnew[i]) - cfull
    f[i] -= fᵢ * (1 - degree(g, variable(i)) + sum(1/dₐ for dₐ in d; init=0.0))
    b[i] = bnew[i]
    return errv, errb
end

function BeliefPropagation.update_f_ms!(bp::BPIsing, a::Integer,
        unew, damp::Real, f::AtomicVector{<:Real}; extra_kwargs...)
    (; g, ψ, u, h) = bp
    ∂a = neighbors(g, factor(a))
    ea = edge_indices(g, factor(a))
    Jₐ = ψ[a].βJ
    @views cavity!(unew[ea], abs.(h[ea]), min, convert(eltype(h), abs(Jₐ)))
    signs, = cavity(sign.(h[ea]), *, sign(Jₐ))
    unew[ea] .= signs .* unew[ea]
    dₐ = degree(g, factor(a))
    err = -Inf
    for (i, ai, s) in zip(∂a, ea, signs)
        fₐ₂ᵢ = abs(Jₐ)
        f[i] -= fₐ₂ᵢ / dₐ
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