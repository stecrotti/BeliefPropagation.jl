# check autodiff through BP by computing internal energy as the derivative of the free energy

@testset "Generic BP" begin
    function free_energy_energy(β)
        rng = MersenneTwister(0)
        g = rand_tree_factor_graph(rng, 30)
        states = [rand(rng, 2:4) for _ in variables(g)]
        fₐ = [rand(rng, [states[i] for i in neighbors(g,factor(a))]...) for a in factors(g)]
        fᵢ = [rand(rng, states[i]) for i in variables(g)] 
        ψ = [BPFactor(f .^ β) for f in fₐ]
        ϕ = [BPFactor(f .^ β) for f in fᵢ]
        bp = BP(g, ψ, states; ϕ)
        f = init_free_energy(bp)
        iterate!(bp; maxiter=30, tol=1e-10, f)
        return sum(f), bethe_free_energy(bp), avg_energy(bp)
    end

    β = 1.0
    f_fast, f, e = free_energy_energy(β)
    @test f_fast ≈ f
    e_autodiff_fast = ForwardDiff.derivative(β -> free_energy_energy(β)[1], β)
    e_autodiff = ForwardDiff.derivative(β -> free_energy_energy(β)[2], β)
    @test e_autodiff_fast ≈ e
    @test e_autodiff ≈ e
end

@testset "Ising" begin
    rng = MersenneTwister(0)
    g = rand_tree_factor_graph(rng, 30)
    J = randn(rng, nfactors(g))
    h = randn(rng, nvariables(g))

    function free_energy_energy(β)
        ψ = [IsingCoupling(β * Jᵢⱼ) for Jᵢⱼ in J]
        ϕ = [IsingField(β * hᵢ) for hᵢ in h]
        bp = fast_ising_bp(g, ψ, ϕ)
        f = init_free_energy(bp)
        iterate!(bp; maxiter=30, tol=1e-10, f)
        return sum(f), bethe_free_energy(bp), avg_energy(bp)
    end

    β = 1.0
    f_fast, f, e = free_energy_energy(β)
    @test f_fast ≈ f
    e_autodiff_fast = ForwardDiff.derivative(β -> free_energy_energy(β)[1], β)
    e_autodiff = ForwardDiff.derivative(β -> free_energy_energy(β)[2], β)
    @test e_autodiff_fast ≈ e
    @test e_autodiff ≈ e
end