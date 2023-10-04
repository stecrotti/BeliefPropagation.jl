module BeliefPropagation

using IndexedGraphs: FactorGraph, Variable, Factor
using IndexedGraphs     # needs new release with bugfixes

using CavityTools: cavity     # needs new release with bugfixes

include("bpfactor.jl")
include("bp.jl")
include("Models/Models.jl")

export BP, beliefs, factor_beliefs, iterate! 

end
