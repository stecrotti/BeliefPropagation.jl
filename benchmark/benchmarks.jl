using BenchmarkTools
using BeliefPropagation
using BeliefPropagation.FactorGraphs
using BeliefPropagation.Test
using BeliefPropagation.Models
using Random

const SUITE = BenchmarkGroup()

rng = MersenneTwister(1) 
m = 10
n = 20
ned = 50
g = rand_factor_graph(rng, m, n, ned)
qs = rand(rng, 2:6, nvariables(g))
bp = rand_bp(rng, g, qs)
maxiter = 50

SUITE["generic bp"] = BenchmarkGroup()

SUITE["generic bp"]["run bp"] = @benchmarkable iterate!(bp2; maxiter=$maxiter, tol=0.0) setup=(bp2 = deepcopy($bp))
SUITE["generic bp"]["run maxsum"] = @benchmarkable iterate_ms!(bp2; maxiter=$maxiter, tol=0.0) setup=(bp2 = deepcopy($bp))

J = randn(nfactors(g))
h = randn(nvariables(g))
ψ = IsingCoupling.(J)
ϕ = IsingField.(h)
bpising = fast_ising_bp(g, ψ, ϕ)

SUITE["ising"] = BenchmarkGroup()

SUITE["ising"]["run bp"] = @benchmarkable iterate!(bpising2; maxiter=$maxiter, tol=0.0) setup=(bpising2 = deepcopy($bpising))
SUITE["ising"]["run maxsum"] = @benchmarkable iterate_ms!(bpising2; maxiter=$maxiter, tol=0.0) setup=(bpising2 = deepcopy($bpising))
