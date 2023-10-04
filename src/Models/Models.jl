module Models 

using BeliefPropagation: BPFactor, VertexBPFactor
using BeliefPropagation

using IndexedGraphs

include("ising.jl")

export IsingCoupling, IsingField, Ising, BP

end