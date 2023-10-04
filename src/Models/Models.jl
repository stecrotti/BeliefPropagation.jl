module Models 

using BeliefPropagation: BPFactor, VertexBPFactor
using BeliefPropagation

using IndexedGraphs
using InvertedIndices

include("ising.jl")

export IsingCoupling, IsingField, Ising
export BP
export exact_normalization, exact_prob, exact_marginals

end