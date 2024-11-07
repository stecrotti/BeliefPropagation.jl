module Models 

using IndexedFactorGraphs: AbstractFactorGraph,
    pairwise_interaction_graph, factor, variable, edge_indices,
    nvariables, factors

using BeliefPropagation: BPFactor, BP, damp!
using BeliefPropagation

using CavityTools: cavity, cavity!
using IndexedGraphs: IndexedGraphs, IndexedGraph, ne, neighbors, nv
using LinearAlgebra: Symmetric

include("ising.jl")
include("ksat.jl")
include("coloring.jl")

export IsingCoupling, IsingField, Ising
export BP, fast_ising_bp
export KSATClause
export ColoringCoupling, SoftColoringCoupling

end