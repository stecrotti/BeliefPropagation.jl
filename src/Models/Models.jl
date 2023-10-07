module Models 

using BeliefPropagation.FactorGraphs: pairwise_interaction_graph

using BeliefPropagation: BPFactor
using BeliefPropagation

using IndexedGraphs
using InvertedIndices

include("ising.jl")

export IsingCoupling, IsingField, Ising
export BP
export exact_normalization, exact_prob, exact_marginals, exact_pair_marginals, 
    exact_avg_energy, minimum_energy

end