module BeliefPropagation

using IndexedGraphs: FactorGraph, Variable, Factor
using IndexedGraphs

using CavityTools: cavity

include("bpfactor.jl")
include("bp.jl")
include("Models/Models.jl")

export BP, beliefs, iterate!

end
