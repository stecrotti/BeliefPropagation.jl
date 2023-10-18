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

# function IndexedGraphs.degree(g::RegularFactorGraph, a::FactorGraphVertex{Factor})
#     a.i == 1 || throw(ArgumentError("Type `RegularFactorGraph` only has one factor, got factor index $a"))
#     return g.kₐ
# end
# function IndexedGraphs.degree(g::RegularFactorGraph, i::FactorGraphVertex{Variable})
#     i.i == 1 || throw(ArgumentError("Type `RegularFactorGraph` only has one variable, got variable index $i"))
#     return g.kᵢ
# end
IndexedGraphs.degree(g::RegularFactorGraph, v::FactorGraphVertex) = length(neighbors(g, v))

function IndexedGraphs.neighbors(g::RegularFactorGraph, a::FactorGraphVertex{Factor})
    # a.i == 1 || throw(ArgumentError("Type `RegularFactorGraph` only has one factor, got factor index $a"))
    return 1:g.kₐ
end
function IndexedGraphs.neighbors(g::RegularFactorGraph, i::FactorGraphVertex{Variable})
    # i.i == 1 || throw(ArgumentError("Type `RegularFactorGraph` only has one variable, got variable index $i"))
    return 1:g.kᵢ
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