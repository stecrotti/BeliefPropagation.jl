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

    function Ising(g::IndexedGraph{Int}, J::Vector{T}, h::Vector{T}, β::T) where {T<:Real}
        @assert length(J) == ne(g)
        @assert length(h) == nv(g)
        @assert β ≥ 0
        new{T}(g, J, h, β)
    end
end

function Ising(J::AbstractMatrix{F}, h::Vector{F}, β::F) where {F<:AbstractFloat}
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

function fast_ising_bp(ising::Ising{T}) where {T<:Real}
    g = pairwise_interaction_graph(ising.g)
    ψ = [IsingCoupling(ising.β * Jᵢⱼ) for Jᵢⱼ in ising.J]
    ϕ = [IsingField(ising.β * hᵢ) for hᵢ in ising.h]
    u = zeros(T, ne(g))
    h = zeros(T, ne(g))
    b = zeros(T, nvariables(g))
    return BP(g, ψ, ϕ, u, h, b)
end

const BPIsing = BP{<:IsingCoupling, <:IsingField, <:Real, <:Real}

BeliefPropagation.nstates(bp::BPIsing, ::Integer) = 2

function BeliefPropagation.update_v_bp!(bp::BPIsing,
        i::Integer, hnew, damp::Real, rein::Real,
        f::AtomicVector{<:Real}; extra_kwargs...)
    (; g, ϕ, u, h, b) = bp
    ∂i = outedges(g, variable(i)) 
    hᵢ = ϕ[i].βh + b[i]*rein
    hnew[idx.(∂i)], b[i] = cavity(u[idx.(∂i)], +, hᵢ)
    cout, cfull = cavity(2cosh.(u[idx.(∂i)]), *, 1.0)
    d = (degree(g, factor(a)) for a in neighbors(g, variable(i)))
    err = -Inf
    for ((_,_,id), dₐ, c) in zip(∂i, d, cout)
        zᵢ₂ₐ = 2cosh(hnew[id]) / c
        f[i] -= log(zᵢ₂ₐ) * (1 - 1/dₐ)
        err = max(err, abs(hnew[id] - h[id]))
        h[id] = damp!(h[id], hnew[id], damp)
    end
    zᵢ = 2cosh(b[i]) / cfull
    f[i] -= log(zᵢ) * (1 - degree(g, variable(i)) + sum(1/dₐ for dₐ in d; init=0.0))
    return err
end

function BeliefPropagation.update_f_bp!(bp::BPIsing, a::Integer,
        unew, damp::Real, f::AtomicVector{<:Real}; extra_kwargs...)
    (; g, ψ, u, h) = bp
    ∂a = inedges(g, factor(a))
    Jₐ = ψ[a].βJ
    unew[idx.(∂a)], = cavity(tanh.(h[idx.(∂a)]), *, 1.0) 
    dₐ = degree(g, factor(a))
    err = -Inf
    for (i, _, id) in ∂a
        zₐ₂ᵢ = 2cosh(Jₐ)
        unew[id] = atanh(tanh(Jₐ)*unew[id])
        f[i] -= log(zₐ₂ᵢ) / dₐ
        err = max(err, abs(unew[id] - u[id]))
        u[id] = damp!(u[id], unew[id], damp)
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
