potts2spin(x) = 3 - 2x
# spin2potts(σ) = (3-σ)/2

struct IsingCoupling{T<:Real}  <: BPFactor 
    βJ :: T 
end

function (f::IsingCoupling)(x)
    (; βJ) = f
    @assert length(x) == 2
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
struct Ising{F<:AbstractFloat}
    g :: IndexedGraph{Int}
    J :: Vector{F}
    h :: Vector{F}
    β :: F

    function Ising(g::IndexedGraph{Int}, J::Vector{F}, h::Vector{F}, β::F) where {F<:AbstractFloat}
        @assert length(J) == ne(g)
        @assert length(h) == nv(g)
        @assert β ≥ 0
        new{F}(g, J, h, β)
    end
end

function Ising(J::AbstractMatrix{F}, h::Vector{F}, β::F) where {F<:AbstractFloat}
    Jvec = [J[i,j] for j in axes(J,2) for i in axes(J,1) if i < j && J[i,j]!=0]
    g = IndexedGraph(Symmetric(J, :L))
    Ising(g, Jvec, h, β)
end

function Ising(g::IndexedGraph; J = ones(ne(g)), h = zeros(nv(g)), β = 1.0)
    Ising(g, J, h, β)
end

function energy(ising::Ising, x)
    s = 0.0
    for (i, j, id) in edges(ising.g)
        s -= potts2spin(x[i])*potts2spin(x[j])*ising.J[id]
    end
    for (xi, hi) in zip(x, ising.h)
        s -= potts2spin(xi)*hi
    end
    return s
end

function exact_normalization(ising::Ising)
    return sum(exp(-ising.β * energy(ising, x)) for x in Iterators.product(fill(1:2, nv(ising.g))...))
end

function exact_prob(ising::Ising; Z = exact_normalization(ising))
    p = [exp(-ising.β * energy(ising, x)) / Z for x in Iterators.product(fill(1:2, nv(ising.g))...)]
    return p
end

function exact_marginals(ising::Ising; p_exact = exact_prob(ising))
    dims = 1:ndims(p_exact)
    return map(dims) do i
        vec(sum(p_exact; dims=dims[Not(i)]))
    end
end

function exact_pair_marginals(ising::Ising; p_exact = exact_prob(ising))
    dims = 1:ndims(p_exact)
    return map(edges(ising.g)) do (i, j)
        Matrix(dropdims(sum(p_exact; dims=dims[Not(i, j)]), dims=tuple(dims[Not(i, j)]...)))
    end
end

function exact_avg_energy(ising::Ising; p_exact = exact_prob(ising))
    k = keys(p_exact)
    sum(energy(ising, Tuple(k[x])) * p_exact[x] for x in eachindex(p_exact))
end

function minimum_energy(ising::Ising)
    return minimum(energy(ising, x) for x in Iterators.product(fill(1:2, nv(ising.g))...))
end

function BeliefPropagation.BP(ising::Ising)
    g = pairwise_interaction_graph(ising.g)
    ψ = [IsingCoupling(ising.β * Jᵢⱼ) for Jᵢⱼ in ising.J]
    ϕ = [IsingField(ising.β * hᵢ) for hᵢ in ising.h]
    qs = fill(2, nvariables(g))
    return BP(g, ψ, qs; ϕ)
end