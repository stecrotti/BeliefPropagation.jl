module Models 

using BeliefPropagation.FactorGraphs: AbstractFactorGraph, FactorGraph,
    pairwise_interaction_graph, factor, variable, degree, edge_indices

using BeliefPropagation: BPFactor, BP, BetheFreeEnergy
import BeliefPropagation: damp!, cavity, cavity!
using BeliefPropagation

using IndexedGraphs
using InvertedIndices
using LinearAlgebra: Symmetric

include("ising.jl")

export IsingCoupling, IsingField, Ising
export BP, fast_ising_bp

end