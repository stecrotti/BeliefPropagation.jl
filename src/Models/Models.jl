module Models 

using IndexedFactorGraphs: AbstractFactorGraph,
    pairwise_interaction_graph, f_vertex, v_vertex, edge_indices,
    nvariables, eachfactor

using BeliefPropagation: BPFactor, BP, damping, set_messages_factor!, set_messages_variable!
using BeliefPropagation

using CavityTools: cavity, cavity!
using IndexedGraphs: IndexedGraphs, IndexedGraph, ne, neighbors, nv
using LinearAlgebra: Symmetric
using Random: AbstractRNG, rand!

include("ising.jl")
include("ksat.jl")
include("coloring.jl")

export IsingCoupling, IsingField, Ising, fast_ising_bp
export KSATClause, fast_ksat_bp
export ColoringCoupling, SoftColoringCoupling

end