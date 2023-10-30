rng = MersenneTwister(0)
m = 50
n = 100
A = sprand(rng, m, n, 0.1)
g = FactorGraph(A)

@testset "Basics" begin
    @test nfactors(g) == m
    @test nvariables(g) == n
    @test all(degree(g, factor(a)) == length(neighbors(g,factor(a))) for a in factors(g))
    @test all(degree(g, variable(i)) == length(neighbors(g,variable(i))) for i in variables(g))
    @test_throws ArgumentError degree(g, 1)
end           