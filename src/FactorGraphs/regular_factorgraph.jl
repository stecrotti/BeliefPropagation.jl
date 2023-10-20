struct RegularFactorGraph <: AbstractFactorGraph{Int}
    kₐ :: Int   # factor degree
    kᵢ :: Int   # variable degree
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

function IndexedGraphs.inedges(g::RegularFactorGraph, ::FactorGraphVertex{Factor})
    return (IndexedEdge(1,1,1) for _ in 1:g.kₐ)
end
function IndexedGraphs.outedges(g::RegularFactorGraph, ::FactorGraphVertex{Factor})
    return (IndexedEdge(1,1,1) for _ in 1:g.kₐ)
end
function IndexedGraphs.inedges(g::RegularFactorGraph, ::FactorGraphVertex{Variable})
    return (IndexedEdge(1,1,1) for _ in 1:g.kᵢ)
end
function IndexedGraphs.outedges(g::RegularFactorGraph, ::FactorGraphVertex{Variable})
    return (IndexedEdge(1,1,1) for _ in 1:g.kᵢ)
end

edge_indices(g::RegularFactorGraph, ::FactorGraphVertex{Factor}) = Fill(1, g.kₐ)
edge_indices(g::RegularFactorGraph, ::FactorGraphVertex{Variable}) = Fill(1, g.kᵢ)