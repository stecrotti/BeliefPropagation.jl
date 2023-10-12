ngraphs = 20
ns = rand(5:50, ngraphs)
ms = rand(5:50, ngraphs)
es = [rand(1:n*m) for (n, m) in zip(ns, ms)]

@test all(zip(ns, ms, es)) do (n, m, e)
    g = rand_factor_graph(n, m, e)
    nvariables(g) == n && nfactors(g) == m && ne(g) == e
end

p = 0.1
@test all(zip(ns, ms)) do (n, m)
    g = rand_factor_graph(n, m, p)
    nvariables(g) == n && nfactors(g) == m
end

k = 4
@test all(zip(ns, ms)) do (n, m)
    g = rand_regular_factor_graph(n, m, k)
    nvariables(g) == n && nfactors(g) == m && ne(g) == m * k
end