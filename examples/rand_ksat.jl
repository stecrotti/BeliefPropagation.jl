using BeliefPropagation, BeliefPropagation.Models
using Random, IndexedFactorGraphs

Random.seed!(0)

ns = 2 .^ (9:2:13)
αs = 3.7:0.05:4.4
nsamples = 50
nunsats = [[zeros(Int, nsamples) for _ in αs] for _ in ns]
niters = [[zeros(Int, nsamples) for _ in αs] for _ in ns]
const k = 3

for (i,n) in enumerate(ns)
    println("Size n=$n: $i of $(length(ns))")
    for (j, α) in enumerate(αs)
        m = round(Int, n*α)
        for l in 1:nsamples
            g = rand_regular_factor_graph(n, m, k)
            ψ = [KSATClause(bitrand(length(neighbors(g, factor(a))))) for a in factors(g)]
            bp = fast_ksat_bp(g, ψ)
            iters = iterate!(bp; maxiter=2000, tol=1e-6, rein=1e-3)
            xstar = argmax.(beliefs(bp))   
            nunsat = sum(!Bool(bp.ψ[a](xstar[i] for i in neighbors(bp.g, factor(a)))) 
                for a in factors(bp.g))
            nunsats[i][j][l] = nunsat
            niters[i][j][l] = iters
        end
        println("\tFinished α=$α: $j of $(length(αs))")
    end
end


using Plots, ColorSchemes, Statistics

cg = cgrad(:matter, length(ns)+1, categorical=true)
pl = plot(; xlabel="α", ylabel="SAT fraction")
for (i,n) in enumerate(ns)
    x = map.(isequal(0), nunsats[i])
    plot!(pl, αs, mean.(x), yerr = std.(x) ./ sqrt(nsamples),
        c=cg[i+1], label="n=$n", m=:o)
end

savefig(pl, (@__DIR__)*"/rand_ksat.pdf")

pl = plot(; xlabel="α", ylabel="# iterations")
for (i,n) in enumerate(ns)
    x = niters[i]
    plot!(pl, αs, mean.(x), yerr = std.(x) ./ sqrt(nsamples),
        c=cg[i+1], label="n=$n", m=:o)
end
savefig(pl, (@__DIR__)*"/rand_ksat_iters.pdf")