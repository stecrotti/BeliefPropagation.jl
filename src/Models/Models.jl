module Models 

using BeliefPropagation.FactorGraphs: pairwise_interaction_graph, factor, variable, degree

using BeliefPropagation: BPFactor, BP
import BeliefPropagation: AtomicVector, damp!, cavity
using BeliefPropagation

using IndexedGraphs
using InvertedIndices
using LinearAlgebra: Symmetric

include("ising.jl")

export IsingCoupling, IsingField, Ising
export BP, fast_ising_bp
# export update_f_bp!, update_v_bp!

end