potts2spin(x) = 3-2x
# spin2potts(σ) = (3-σ)/2

struct IsingCoupling{T<:Real}  <: BPFactor 
    βJ :: T 
end

function (f::IsingCoupling)(x)
    (; βJ) = f
    @assert length(x) == 2
    E = - βJ * prod(potts2spin(xᵢ) for xᵢ in x)
    return 1 / (1 + exp(2E))
end

struct IsingField{T<:Real}  <: VertexBPFactor 
    βh :: T 
end

function (f::IsingField)(x::Integer)
    (; βh) = f
    E = - βh * potts2spin(x)
    return 1 / (1 + exp(2E))
end

# Ising model with xᵢ ∈ {1,2} mapped onto spins {+1,-1}
struct Ising{F<:AbstractFloat}
    g :: IndexedGraph{Int}
    J :: Vector{F}
    h :: Vector{F}
    β :: F

    function Ising(g::IndexedGraph{Int}, J::Vector{F}, h::Vector{F}, β::F) where {F<:AbstractFloat}
        N = nv(g); E = ne(g)
        @assert length(J) == E
        @assert length(h) == N
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

function energy(ising::Ising, x::AbstractVector{<:Integer})
    s = 0.0
    for (i, j, id) in edges(ising.g)
        s -= potts2spin(x[i])*potts2spin(x[j])*ising.J[id]
    end
    for (xi, hi) in zip(x, ising.h)
        s -= potts2spin(xi)*hi
    end
    return s
end

function BeliefPropagation.BP(ising::Ising)
    g = FactorGraph(ising.g)
    ψ = [IsingCoupling(ising.β * Jᵢⱼ) for Jᵢⱼ in ising.J]
    ϕ = [IsingField(ising.β * hᵢ) for hᵢ in ising.h]
    qs = fill(2, nvariables(g))
    return BP(g, ψ, qs; ϕ)
end