ngraphs = 20
ns = rand(5:50, ngraphs)
ms = rand(5:50, ngraphs)
es = [rand(1:n*m) for (n, m) in zip(ns, ms)]

@testset "Random factor graph - fixed # edges" begin
@test all(zip(ns, ms, es)) do (n, m, e)
    g = rand_factor_graph(n, m, e)
    nvariables(g) == n && nfactors(g) == m && ne(g) == e
end
end

@testset "Random factor graph - prob of edges" begin
p = 0.1
@test all(zip(ns, ms)) do (n, m)
    g = rand_factor_graph(n, m, p)
    nvariables(g) == n && nfactors(g) == m
end
end

@testset "Random regular factor graph" begin
k = 4
@test all(zip(ns, ms)) do (n, m)
    g = rand_regular_factor_graph(n, m, k)
    nvariables(g) == n && nfactors(g) == m && ne(g) == m * k
end
end

@testset "Random tree factor graph" begin
@test all(ns) do n
    g = rand_tree_factor_graph(n)
    nv(g) == n && !is_cyclic(g)
end
end