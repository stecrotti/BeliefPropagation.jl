module FactorGraphs

using IndexedGraphs:
    AbstractIndexedGraph, IndexedGraph, IndexedEdge, BipartiteIndexedGraph, 
    BipartiteGraphVertex,
    NullNumber,
    src, dst, idx, edges, nv, ne, degree, linearindex,
    nv_left, nv_right, Left, Right
using IndexedGraphs

using Graphs: is_cyclic, prufer_decode
using Graphs

using SparseArrays: sparse, SparseMatrixCSC, nzrange
using Random: AbstractRNG, default_rng
using StatsBase: sample

include("factorgraph.jl")
include("generators.jl")

export FactorGraph, nvariables, nfactors, variables, factors, factor, variable,
    pairwise_interaction_graph,
    neighbors, inedges, outedges, edges, src, dst, idx, ne, nv, degree,
    adjacency_matrix, is_cyclic
export rand_factor_graph, rand_regular_factor_graph, rand_tree_factor_graph

end