module BeliefPropagation

include("FactorGraphs/FactorGraphs.jl")

using .FactorGraphs

using CavityTools: cavity 

using Random: AbstractRNG, GLOBAL_RNG

include("bpfactor.jl")
include("bp.jl")
include("Models/Models.jl")

export FactorGraph, variables, factors, nvariables, nfactors,
    BP, rand_bp, beliefs, factor_beliefs, iterate! 

end
