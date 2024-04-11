module FactorGraphs

using IndexedGraphs:
    IndexedGraphs,
    AbstractIndexedGraph, IndexedGraph, IndexedEdge, BipartiteIndexedGraph, 
    BipartiteGraphVertex,
    NullNumber,
    src, dst, idx, edges, nv, ne, degree, linearindex, neighbors, adjacency_matrix,
    nv_left, nv_right, vertex, Left, Right

using Graphs: Graphs, AbstractGraph, is_cyclic, prufer_decode

using FillArrays: Fill

using SparseArrays: sparse, SparseMatrixCSC, nzrange
using Random: AbstractRNG, default_rng
using StatsBase: sample

abstract type AbstractFactorGraph{T} end

include("factorgraph.jl")
include("generators.jl")
include("regular_factorgraph.jl")

export FactorGraph, nvariables, nfactors, variables, factors, factor, variable,
    pairwise_interaction_graph,
    neighbors, edges, src, dst, idx, ne, nv, degree,
    edge_indices,
    adjacency_matrix, is_cyclic
export rand_factor_graph, rand_regular_factor_graph, rand_tree_factor_graph
export RegularFactorGraph

end