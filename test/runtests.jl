using BeliefPropagation
using Test
using Aqua
using Graphs
using IndexedGraphs
using SparseArrays
using Random

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(BeliefPropagation; ambiguities = false,)
    Aqua.test_ambiguities(BeliefPropagation)
end

using BeliefPropagation.FactorGraphs
@testset "FactorGraphs" begin
    include("FactorGraphs/factorgraphs.jl")
end

@testset "BeliefPropagation" begin
    include("bp.jl")
end

using BeliefPropagation.Models
@testset "Ising" begin
    include("Models/ising.jl")
end

nothing