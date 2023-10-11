module Models 

using BeliefPropagation.FactorGraphs: pairwise_interaction_graph

using BeliefPropagation: BPFactor
using BeliefPropagation

using IndexedGraphs
using InvertedIndices
using LinearAlgebra: Symmetric

include("ising.jl")

export IsingCoupling, IsingField, Ising
export BP

end