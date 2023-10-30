using BeliefPropagation: UpdateFactorBP#, UpdateFactorMaxsum

n = 10
rng = MersenneTwister(0)
g = rand_tree_factor_graph(rng, n)
qs = rand(rng, 2:4, nvariables(g))
bp = rand_bp(rng, g, qs)
f = init_free_energy(bp)

iterate!(bp; maxiter=100, tol=0.0, f, update_factor! = UpdateFactorBP())
test_observables_bp(bp)
@test sum(f) ≈ bethe_free_energy(bp)

# ms = bp
# reset!(ms)
# ms2 = deepcopy(ms)
# f = init_free_energy(ms)
# iterate!(ms; maxiter=1, f, tol=0, update_variable! = update_v_ms!,
#     update_factor! = UpdateFactorMaxsum())
# f2 = init_free_energy(ms2)
# iterate_ms!(ms2; maxiter=1, f=f2, tol=0)
# bfe = bethe_free_energy_ms(ms)
# @test bfe ≈ sum(f)
# @test bfe ≈ exact_minimum_energy(ms)