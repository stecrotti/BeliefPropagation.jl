module Models 

using IndexedFactorGraphs: AbstractFactorGraph, FactorGraph,
    pairwise_interaction_graph, factor, variable, degree, edge_indices,
    nvariables, nfactors, variables, factors

using BeliefPropagation: BPFactor, BP, BetheFreeEnergy, damp!, cavity, cavity!
using BeliefPropagation

using IndexedGraphs
using InvertedIndices
using LinearAlgebra: Symmetric

include("ising.jl")

export IsingCoupling, IsingField, Ising
export BP, fast_ising_bp

end