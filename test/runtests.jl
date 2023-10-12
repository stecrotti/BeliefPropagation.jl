using BeliefPropagation
using Test
using Aqua
using Graphs
using IndexedGraphs
using SparseArrays
using Random: AbstractRNG
using Random
using InvertedIndices

using BeliefPropagation.FactorGraphs
using BeliefPropagation.Models


include("testutils.jl")

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(BeliefPropagation; ambiguities = false,)
    Aqua.test_ambiguities(BeliefPropagation)
end

@testset "FactorGraphs" begin
    include("FactorGraphs/factorgraph.jl")
    include("FactorGraphs/generators.jl")
end

@testset "BeliefPropagation" begin
    include("bp.jl")
end

@testset "Ising" begin
    include("Models/ising.jl")
end

nothing