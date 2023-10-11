rng = MersenneTwister(0)
ngraphs = 20
ns = rand(rng, 1:50, ngraphs)
ms = rand(rng, 1:50, ngraphs)
es = rand(rng, 1:50, ngraphs)

@test all(zip(ns, ms, es)) do (n, m, e)
    g = rand_factor_graph(rng, n, m, e)
    nvariables(g) == n && nfactors(g) == m && ne(g) == e
end

p = 0.1
@test all(zip(ns, ms)) do (n, m)
    g = rand_factor_graph(rng, n, m, p)
    nvariables(g) == n && nfactors(g) == m
end