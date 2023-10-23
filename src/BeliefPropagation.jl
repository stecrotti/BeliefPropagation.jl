module BeliefPropagation

include("FactorGraphs/FactorGraphs.jl")

using .FactorGraphs: AbstractFactorGraph, FactorGraph, RegularFactorGraph, FactorGraphVertex,
    nvariables, nfactors, variables, factors, factor, variable,
    neighbors, edge_indices, edges, src, dst, idx, ne, nv, degree
using CavityTools: cavity, cavity! 
using Random: AbstractRNG, default_rng
using Statistics: mean
using Lazy: @forward
using Base.Threads: @threads, SpinLock

include("bpfactor.jl")
include("atomic_vector.jl")
include("bp.jl")
include("maxsum.jl")

include("Models/Models.jl")

export BPFactor, TabulatedBPFactor, rand_factor, UniformFactor
export AbstractFactorGraph, FactorGraph, variables, factors, nvariables, nfactors
export BP, reset!, nstates, evaluate, energy
export iterate!, beliefs, factor_beliefs, avg_energy, bethe_free_energy
export message_convergence, belief_convergence
export update_f_bp!, update_v_bp!, beliefs_bp, factor_beliefs_bp, avg_energy_bp
export update_f_ms!, update_v_ms!, beliefs_ms, factor_beliefs_ms, iterate_ms!,
    avg_energy_ms, bethe_free_energy_ms

end
