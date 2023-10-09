struct BP{F<:BPFactor, FV<:BPFactor, M, MB, G<:FactorGraph, T<:Real}
    g :: G               # graph
    ψ :: Vector{F}       # factors
    ϕ :: Vector{FV}      # vertex-dependent factors
    u :: Vector{M}       # messages factor -> variable
    h :: Vector{M}       # messages variable -> factor
    b :: Vector{MB}      # beliefs
    f :: Vector{T}       # free energy contributions

    function BP(g::G, ψ::Vector{F}, ϕ::Vector{FV}, u::Vector{M}, h::Vector{M}, b::Vector{MB},
        f::Vector{T}) where {G<:FactorGraph, F<:BPFactor, FV<:BPFactor, M, MB, T<:Real}

        nvar = nvariables(g)
        nfact = nfactors(g)
        nedges = ne(g)
        length(ψ) == nfact || throw(DimensionMismatch("Number of factor nodes in factor graph `g`, $nfact, does not match length of `ψ`, $(length(ψ))"))
        length(ϕ) == nvar || throw(DimensionMismatch("Number of variable nodes in factor graph `g`, $nvar, does not match length of `ϕ`, $(length(ϕ))"))
        length(u) == nedges || throw(DimensionMismatch("Number of edges in factor graph `g`, $nvar, does not match length of `u`, $(length(u))"))
        length(h) == nedges || throw(DimensionMismatch("Number of edges in factor graph `g`, $nvar, does not match length of `h`, $(length(h))"))
        length(b) == nvar || throw(DimensionMismatch("Number of variable nodes in factor graph `g`, $nvar, does not match length of `b`, $(length(b))"))
        length(f) == nvar || throw(DimensionMismatch("Number of variable nodes in factor graph `g`, $nvar, does not match length of `f`, $(length(f))"))
        new{F,FV,M,MB,G,T}(g, ψ, ϕ, u, h, b, f)
    end
end

function BP(g::FactorGraph, ψ, qs; ϕ = [UniformFactor(q) for q in qs])
    u = [ones(qs[dst(e)]) for e in edges(g)]
    h = [ones(qs[dst(e)]) for e in edges(g)]
    b = [ones(qs[i]) for i in variables(g)]
    f = zeros(nvariables(g))
    return BP(g, ψ, ϕ, u, h, b, f)
end

function rand_bp(rng::AbstractRNG, g::FactorGraph, qs)
    ψ = [random_factor(rng, [qs[i] for i in neighbors(g,factor(a))]) for a in factors(g)] 
    return BP(g, ψ, qs)  
end
rand_bp(g::FactorGraph, qs) = rand_bp(GLOBAL_RNG, g, qs)

nstates(bp::BP, i::Integer) = length(bp.b[i])
beliefs(f, bp::BP) = f(bp)
beliefs_bp(bp::BP) = bp.b
beliefs(bp::BP) = beliefs(beliefs_bp, bp)
factor_beliefs(f, bp::BP) = f(bp)
avg_energy(f, bp::BP) = f(bp)

function iterate!(bp::BP; update_variable! = update_v_bp!, update_factor! = update_f_bp!,
        maxiter=100)
    for it in 1:maxiter
        bp.f .= 0
        for i in variables(bp.g)
            update_variable!(bp, i)
        end
        for a in factors(bp.g)
            update_factor!(bp, a)
        end
    end
    return nothing
end

function update_v_bp!(bp::BP, i::Integer)
    (; g, ϕ, u, h, b, f) = bp
    ∂i = outedges(g, variable(i)) 
    ϕᵢ = [ϕ[i](x) for x in 1:nstates(bp, i)]
    msg_mult(m1, m2) = m1 .* m2
    h[idx.(∂i)], b[i] = cavity(u[idx.(∂i)], msg_mult, ϕᵢ)
    d = (degree(g, factor(a)) for a in neighbors(g, variable(i)))
    for (ia, dₐ) in zip(∂i, d)
        hᵢₐ = h[idx(ia)]
        zᵢ₂ₐ = sum(hᵢₐ)
        f[i] -= log(zᵢ₂ₐ) * (1 - 1/dₐ)
        hᵢₐ ./= zᵢ₂ₐ
    end
    zᵢ = sum(b[i])
    f[i] -= log(zᵢ) * (1 - degree(g, variable(i)) + sum(1/dₐ for dₐ in d; init=0.0))
    b[i] ./= zᵢ
    return nothing
end

function update_f_bp!(bp::BP, a::Integer)
    (; g, ψ, u, h, f) = bp
    ∂a = inedges(g, factor(a))
    ψₐ = ψ[a]
    for ai in ∂a
        u[idx(ai)] .= 0
    end
    for xₐ in Iterators.product((1:nstates(bp, src(e)) for e in ∂a)...)
        for (i, ai) in pairs(∂a)
            u[idx(ai)][xₐ[i]] += ψₐ(xₐ) * 
                prod(h[idx(ja)][xₐ[j]] for (j, ja) in pairs(∂a) if j != i; init=1.0)
        end
    end
    dₐ = degree(g, factor(a))
    for (i, _, id) in ∂a
        uₐᵢ = u[id]
        zₐ₂ᵢ = sum(uₐᵢ)
        f[i] -= log(zₐ₂ᵢ) / dₐ
        uₐᵢ ./= zₐ₂ᵢ
    end
    return nothing
end

function factor_beliefs_bp(bp::BP)
    (; g, ψ, h) = bp
    return map(factors(g)) do a
        ∂a = inedges(g, factor(a))
        ψₐ = ψ[a]
        bₐ = zeros((nstates(bp, src(ia)) for ia in ∂a)...)
        for xₐ in keys(bₐ)
            bₐ[xₐ] = ψₐ(Tuple(xₐ)) * prod(h[idx(ia)][xₐ[i]] for (i, ia) in pairs(∂a); init=1.0)
        end
        zₐ = sum(bₐ)
        bₐ ./= zₐ
        bₐ
    end
end
factor_beliefs(bp::BP) = factor_beliefs(factor_beliefs_bp, bp)

function avg_energy_bp(bp::BP; fb = factor_beliefs(bp), b = beliefs(bp))
    (; g, ψ, ϕ) = bp
    e = 0.0
    for a in factors(g)
        ∂a = inedges(g, factor(a))
        for xₐ in Iterators.product((1:nstates(bp, src(e)) for e in ∂a)...)
            e += -log(ψ[a](xₐ)) * fb[a][xₐ...]
        end
    end
    for i in variables(g)
        for xᵢ in eachindex(b[i])
            e += -log(ϕ[i](xᵢ)) * b[i][xᵢ]
        end
    end
    return e
end
avg_energy(bp::BP) = avg_energy(avg_energy_bp, bp)

bethe_free_energy(bp::BP) = sum(bp.f)