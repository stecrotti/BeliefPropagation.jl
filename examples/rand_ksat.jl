using BeliefPropagation, BeliefPropagation.Models
using Random, IndexedFactorGraphs

rng = MersenneTwister(0)


ns = [100, 1000, 10000]
alphas = 03.6:0.1:4.2
nsamples = 50
nunsats = [[zeros(Int, nsamples) for _ in alphas] for _ in ns]
const k = 3

for (i,n) in enumerate(ns)
    for (j, alpha) in enumerate(alphas)
        m = round(Int, n*alpha)
        for k in 1:nsamples
            g = rand_regular_factor_graph(rng, n, m, k)
            ψ = [KSATClause(bitrand(rng, length(neighbors(g, factor(a))))) for a in factors(g)]
            bp = fast_ksat_bp(g, ψ)
            iters = iterate!(bp; maxiter=1000, tol=1e-10, rein=5e-2)
            xstar = argmax.(beliefs(bp))   
            nunsat = sum(!(Bool(bp.ψ[a](xstar[i]+1 for i in neighbors(bp.g, factor(a))))) 
                for a in factors(bp.g))
            nunsats[i][j][k] = nunsat
        end
        println("\tFinished alpha $j of $(length(alphas))")
    end
    println("Finished n $i of $(length(ns))")
end


using Plots, ColorSchemes, Statistics

cg = cgrad(:matter, length(ns)+1, categorical=true)
pl = plot(; xlabel="α", ylabel="prob SAT")
for (i,n) in enumerate(ns)
    plot!(pl, alphas, mean.(isequal(0), nunsats[i]), c=cg[i+1], label="n=$n", m=:o)
end
pl