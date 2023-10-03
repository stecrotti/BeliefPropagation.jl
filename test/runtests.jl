using BeliefPropagation
using Test
using Aqua

@testset "BeliefPropagation.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(BeliefPropagation; ambiguities = false,)
    end
    # Write your tests here.
end
