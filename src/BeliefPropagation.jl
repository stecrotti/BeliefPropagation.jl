module BeliefPropagation

include("FactorGraphs/FactorGraphs.jl")

using .FactorGraphs

using CavityTools: cavity 

using Random: AbstractRNG, GLOBAL_RNG

using Statistics: mean

using Lazy: @forward

using Base.Threads: @threads, SpinLock

include("bpfactor.jl")
include("atomic_vector.jl")
include("bp.jl")
include("maxsum.jl")

include("Models/Models.jl")

export BPFactor, TabulatedBPFactor, rand_factor
export FactorGraph, variables, factors, nvariables, nfactors
export BP, rand_bp, reset!, nstates
export iterate!, beliefs, factor_beliefs, avg_energy, bethe_free_energy
export update_f_bp!, update_v_bp!, beliefs_bp, factor_beliefs_bp, avg_energy_bp
export update_f_ms!, update_v_ms!, beliefs_ms, factor_beliefs_ms, iterate_ms!,
    avg_energy_ms, bethe_free_energy_ms

end
