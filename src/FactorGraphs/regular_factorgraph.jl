"""
    RegularFactorGraph <: AbstractFactorGraph

A type to represent an infinite regular factor graph with fixed factor and variable degree
"""
struct RegularFactorGraph{T<:Integer} <: AbstractFactorGraph{T}
    kₐ :: T   # factor degree
    kᵢ :: T   # variable degree

    function RegularFactorGraph(kₐ::T, kᵢ::T) where {T<:Integer}
        kₐ > 0 || throw(ArgumentError("Factor degree must be positive, got $kₐ"))
        kᵢ > 0 || throw(ArgumentError("Factor degree must be positive, got $kᵢ"))
        return new{T}(kₐ, kᵢ)
    end
end

nvariables(::RegularFactorGraph) = 1
nfactors(::RegularFactorGraph) = 1
Graphs.ne(::RegularFactorGraph) = 1
Graphs.edges(::RegularFactorGraph) = (IndexedEdge(1,1,1) for _ in 1:1)
variables(::RegularFactorGraph) = 1:1
factors(::RegularFactorGraph) = 1:1

IndexedGraphs.degree(g::RegularFactorGraph, v::FactorGraphVertex) = length(neighbors(g, v))

function IndexedGraphs.neighbors(g::RegularFactorGraph, ::FactorGraphVertex{Factor})
    return Fill(1, g.kₐ)
end
function IndexedGraphs.neighbors(g::RegularFactorGraph, ::FactorGraphVertex{Variable})
    return Fill(1, g.kᵢ)
end
edge_indices(g::RegularFactorGraph, ::FactorGraphVertex{Factor}) = Fill(1, g.kₐ)
edge_indices(g::RegularFactorGraph, ::FactorGraphVertex{Variable}) = Fill(1, g.kᵢ)