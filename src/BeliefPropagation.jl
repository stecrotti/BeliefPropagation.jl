module BeliefPropagation

using IndexedFactorGraphs: AbstractFactorGraph, InfiniteRegularFactorGraph,
    nvariables, nfactors, variables, factors, factor, variable,
    neighbors, edge_indices, edges, dst, ne, degree
using CavityTools: cavity! 
using Base.Threads: @threads
using LogExpFunctions: xlogx, xlogy
using BlockArrays: mortar
using ForwardDiff: ForwardDiff

include("bpfactor.jl")
include("bp.jl")
include("maxsum.jl")

include("Models/Models.jl")
include("Test/Test.jl")

export BPFactor, TabulatedBPFactor, UniformFactor
export BP, reset!, nstates, evaluate, energy, energy_factors, energy_variables
export iterate!, beliefs, factor_beliefs, avg_energy, bethe_free_energy
export message_convergence, belief_convergence
export update_f_bp!, update_v_bp!, beliefs_bp, factor_beliefs_bp, avg_energy_bp
export update_f_ms!, update_v_ms!, beliefs_ms, factor_beliefs_ms, iterate_ms!,
    avg_energy_ms, bethe_free_energy_ms

end
