using BeliefPropagation
using BeliefPropagation.Models
using BeliefPropagation.Test
using Test
using Aqua
using IndexedFactorGraphs
using Graphs
using IndexedGraphs
using SparseArrays
using Random: default_rng
using Random
using InvertedIndices
using ForwardDiff

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(BeliefPropagation; ambiguities = false,)
    Aqua.test_ambiguities(BeliefPropagation)
end

@testset "BeliefPropagation" begin
    include("bp.jl")
end

@testset "Ising" begin
    include("Models/ising.jl")
end

@testset "KSAT" begin
    include("Models/ksat.jl")
end

@testset "Coloring" begin
    include("Models/coloring.jl")
end

@testset "Autodiff" begin
    include("autodiff.jl")
end

nothing