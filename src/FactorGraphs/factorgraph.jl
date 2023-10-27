const Factor = Left
const Variable = Right

"""
    FactorGraphVertex

A type to represent a vertex in a bipartite graph, to be passed as an argument to [`neighbors`](@ref), see examples therein.
It is recommended to use the [`variable`](@ref) and [`factor`](@ref) constructors.
"""
const FactorGraphVertex = BipartiteGraphVertex

"""
    factor(a::Integer)

Wraps index `a` in a container such that other functions like [`neighbors`](@ref) know that it indices a factor node.
"""
factor(a::Integer) = vertex(a, Factor)

"""
    variable(i::Integer)

Wraps index `i` in a container such that other functions like [`neighbors`](@ref) know that it indices a variable node.
"""
variable(i::Integer) = vertex(i, Variable)

abstract type AbstractFactorGraph{T} end

# treat an `AbstractFactorGraph`` object as a scalar in broadcasting
Base.broadcastable(b::AbstractFactorGraph) = Ref(b)

"""
    FactorGraph{T}

A type representing a [factor graph](https://en.wikipedia.org/wiki/Factor_graph).
"""
struct FactorGraph{T} <: AbstractFactorGraph{T}
    g :: BipartiteIndexedGraph{T}
end
"""
    FactorGraph(A::AbstractMatrix)

Construct a `FactorGraph` from adjacency matrix `A` with the convention that rows are factors, columns are variables.
"""
function FactorGraph(A::AbstractMatrix)
    A = sparse(A)
    g = BipartiteIndexedGraph(SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, fill(NullNumber(), length(A.nzval))))
	FactorGraph(g)
end

"""
    pairwise_interaction_graph(g::IndexedGraph)

Construct a factor graph whose factors are the pair-wise interactions encoded in `g`.
"""
function pairwise_interaction_graph(g::AbstractGraph)
    I = reduce(vcat, [e, e] for e in 1:ne(g))
    J = reduce(vcat, [src(e), dst(e)] for e in edges(g)) 
    K = ones(Int, 2*ne(g))
    A = sparse(I, J, K)
    FactorGraph(A)
end

function Base.show(io::IO, g::FactorGraph{T}) where T
    nfact = nfactors(g)
    nvar = nvariables(g)
    ned = ne(g)
    println(io, "FactorGraph{$T} with $nfact factors, $nvar variables and $ned edges")
end

"""
    nvariables(g::FactorGraph)

Return the number of variables vertices in `g`.
"""
nvariables(g::FactorGraph) = nv_right(g.g)

"""
    nactors(g::FactorGraph)

Return the number of actors vertices in `g`.
"""
nfactors(g::FactorGraph) = nv_left(g.g)
IndexedGraphs.nv(g::FactorGraph) = nv(g.g)
IndexedGraphs.ne(g::FactorGraph) = ne(g.g)

"""
    variables(g::FactorGraph)

Return a lazy iterator to the indices of variable vertices in `g`.
"""
variables(g::FactorGraph) = 1:nvariables(g)

"""
    factors(g::FactorGraph)

Return a lazy iterator to the indices of factor vertices in `g`.
"""
factors(g::FactorGraph) = 1:nfactors(g)

"""
    IndexedGraphs.neighbors(g::FactorGraph, v::FactorGraphVertex)

Return a lazy iterators to the neighbors of vertex `v`.

Examples
========

```jldoctest neighbors
julia> using BeliefPropagation.FactorGraphs

julia> g = FactorGraph([0 1 1 0;
                        1 0 0 0;
                        0 0 1 1])
FactorGraph{Int64} with 3 factors, 4 variables and 5 edges

julia> collect(neighbors(g, variable(3)))
2-element Vector{Int64}:
 1
 3

julia> collect(neighbors(g, factor(2)))
1-element Vector{Int64}:
 1
```
"""
function IndexedGraphs.neighbors(g::FactorGraph, a::FactorGraphVertex{Factor})
    return @view g.g.X.rowval[nzrange(g.g.X, a.i)]
end
function IndexedGraphs.neighbors(g::FactorGraph, i::FactorGraphVertex{Variable})
    return @view g.g.A.rowval[nzrange(g.g.A, i.i)]
end

"""
    IndexedGraphs.edge_indices(g::FactorGraph, v::FactorGraphVertex)

Return a lazy iterators to the indices of the edges incident on vertex `v`, with `v`.

The output of `edge_indices` does not allocate and it can be used to index external arrays of properties directly

Examples
========

```jldoctest edge_indices
julia> using BeliefPropagation.FactorGraphs, Random

julia> g = FactorGraph([0 1 1 0;
                        1 0 0 0;
                        0 0 1 1])
FactorGraph{Int64} with 3 factors, 4 variables and 5 edges

julia> edgeprops = randn(MersenneTwister(0), ne(g));

julia> indices = edge_indices(g, variable(3));

julia> edgeprops[indices]
2-element Vector{Float64}:
 -0.3530074003005963
 -0.13485387193052173
```
"""
function edge_indices(g::FactorGraph, a::FactorGraphVertex{Factor})
    return @view g.g.X.nzval[nzrange(g.g.X, a.i)]
end
function edge_indices(g::FactorGraph, i::FactorGraphVertex{Variable})
    return nzrange(g.g.A, i.i)
end

"""
    edges(g::FactorGraph)

Return a lazy iterator to the edges of `g`, with the convention that the source is the factor and the destination is the variable

```jldoctest edges
julia> using BeliefPropagation.FactorGraphs

julia> g = FactorGraph([0 1 1 0;
                        1 0 0 0;
                        0 0 1 1])
FactorGraph{Int64} with 3 factors, 4 variables and 5 edges

julia> collect(edges(g))
5-element Vector{IndexedGraphs.IndexedEdge{Int64}}:
 Indexed Edge 2 => 1 with index 1
 Indexed Edge 1 => 2 with index 2
 Indexed Edge 1 => 3 with index 3
 Indexed Edge 3 => 3 with index 4
 Indexed Edge 3 => 4 with index 5
```
"""
function IndexedGraphs.edges(g::FactorGraph)
    A = g.g.A
    return (IndexedEdge(A.rowval[k], j, k) for j=1:size(A, 2) for k=nzrange(A, j))
end

function IndexedGraphs.degree(g::FactorGraph, v::FactorGraphVertex)
    return degree(g.g, linearindex(g.g, v))
end
function IndexedGraphs.degree(g::FactorGraph, i::Integer)
    return throw(ArgumentError("Properties of a vertex of a `FactorGraph` such as degree, neighbors, etc. cannot be accessed using an integer. Use a `FactorGraphVertex` instead."))
end

IndexedGraphs.adjacency_matrix(g::FactorGraph, T::DataType=Int) = g.g.A

Graphs.is_cyclic(g::FactorGraph) = is_cyclic(g.g)