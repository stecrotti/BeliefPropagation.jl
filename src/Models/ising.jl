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
    dₐ = degree(g, factor(a))
    err = -Inf
    for (i, ai) in zip(∂a, ea)
        zₐ₂ᵢ = 2cosh(Jₐ)
        unew[ai] = atanh(tanh(Jₐ)*unew[ai])
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
        ∂a = inedges(g, factor(a))
        ψₐ = ψ[a]
        bₐ = zeros((nstates(bp, src(ia)) for ia in ∂a)...)
        for xₐ in keys(bₐ)
            bₐ[xₐ] = ψₐ(Tuple(xₐ)) * prod(exp(h[idx(ia)]*potts2spin(xₐ[i])) for (i, ia) in pairs(∂a); init=1.0)
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
    ∂i = outedges(g, variable(i)) 
    hᵢ = ϕ[i].βh + b[i]*rein
    hnew[idx.(∂i)], bnew[i] = cavity(u[idx.(∂i)], +, hᵢ)
    cout, cfull = cavity(abs.(u[idx.(∂i)]), +, 0.0)
    d = (degree(g, factor(a)) for a in neighbors(g, variable(i)))
    errv = -Inf
    for ((_,_,id), dₐ, c) in zip(∂i, d, cout)
        fᵢ₂ₐ = abs(hnew[id]) - c
        f[i] -= fᵢ₂ₐ * (1 - 1/dₐ)
        errv = max(errv, abs(hnew[id] - h[id]))
        h[id] = damp!(h[id], hnew[id], damp)
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
    ∂a = inedges(g, factor(a))
    Jₐ = ψ[a].βJ
    unew[idx.(∂a)], = cavity(abs.(h[idx.(∂a)]), min, convert(eltype(h), abs(Jₐ))) 
    signs, = cavity(sign.(h[idx.(∂a)]), *, sign(Jₐ))
    dₐ = degree(g, factor(a))
    err = -Inf
    for ((i, _, id), s) in zip(∂a, signs)
        fₐ₂ᵢ = abs(Jₐ)
        unew[id] = s * unew[id]
        f[i] -= fₐ₂ᵢ / dₐ
        err = max(err, abs(unew[id] - u[id]))
        u[id] = damp!(u[id], unew[id], damp)
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
        ∂a = inedges(g, factor(a))
        ψₐ = ψ[a]
        bₐ = map(Iterators.product((1:nstates(bp, src(e)) for e in ∂a)...)) do xₐ
            log(ψₐ(xₐ)) + sum(h[idx(ia)]*potts2spin(xₐ[i]) for (i, ia) in pairs(∂a); init=0.0)
        end
        zₐ = maximum(bₐ)
        bₐ .-= zₐ
        bₐ
    end
end