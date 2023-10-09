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
    @test all(degree(g, factor(a)) == length(inedges(g,factor(a))) for a in factors(g))
    @test all(degree(g, variable(i)) == length(inedges(g,variable(i))) for i in variables(g))
    @test all(all(src(e)==a for (e,a) in zip(inedges(g, variable(i)), neighbors(g, variable(i)))) for i in variables(g))
    @test all(all(src(e)==i for (e,i) in zip(inedges(g, factor(a)), neighbors(g, factor(a)))) for a in factors(g))
    @test all(all(dst(e)==a for (e,a) in zip(outedges(g, variable(i)), neighbors(g, variable(i)))) for i in variables(g))
    @test all(all(dst(e)==i for (e,i) in zip(outedges(g, factor(a)), neighbors(g, factor(a)))) for a in factors(g))
end           
