using BeliefPropagation
using Test
using Aqua
using Graphs
using IndexedGraphs
using Random

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(BeliefPropagation; ambiguities = false,)
    Aqua.test_ambiguities(BeliefPropagation)
end

include("ising.jl")

nothing