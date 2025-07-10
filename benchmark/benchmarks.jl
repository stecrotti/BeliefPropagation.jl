using BenchmarkTools
using BeliefPropagation
using BeliefPropagation.Test
using BeliefPropagation.Models

using Random, IndexedFactorGraphs

const SUITE = BenchmarkGroup()

rng = MersenneTwister(1)
maxiter = 50 

m = 10
n = 20
ned = 50
g = rand_factor_graph(rng, m, n, ned)
qs = rand(rng, 2:6, nvariables(g))
bp = rand_bp(rng, g, qs)

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

n = 100
m = 60
k = 3
g = rand_regular_factor_graph(rng, n, m, k)
ψ = [KSATClause(bitrand(rng, degree(g, f_vertex(a)))) for a in eachfactor(g)]
bp = fast_ksat_bp(g, ψ)
bp_generic = make_generic(bp)

SUITE["ksat"]["optimized"] = @benchmarkable iterate!(bp2; maxiter=$maxiter, tol=0.0) setup=(bp2 = deepcopy($bp))
SUITE["ksat"]["generic"] = @benchmarkable iterate!(bp2; maxiter=$maxiter, tol=0.0) setup=(bp2 = deepcopy($bp_generic))