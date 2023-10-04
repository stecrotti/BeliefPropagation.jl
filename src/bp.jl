struct BP{F<:BPFactor, M, G<:FactorGraph, FV<:VertexBPFactor, MB, T<:Real}
    g :: G                              # graph
    ψ :: Vector{F}                      # factors
    ϕ :: Vector{FV}      # vertex-dependent factors
    u :: Vector{M}                      # messages factor -> variable
    h :: Vector{M}                      # messages variable -> factor
    b :: Vector{MB}                     # beliefs
    f :: Vector{T}                      # free energy contributions

    function BP(g::G, ψ::Vector{F}, ϕ::Vector{FV}, u::Vector{M}, h::Vector{M}, b::Vector{MB},
        f::Vector{T}) where {G<:FactorGraph, F<:BPFactor, FV<:VertexBPFactor, M, MB, T<:Real}

        nvar = nvariables(g)
        nfact = nfactors(g)
        nedges = ne(g)
        nvert = nv(g)
        length(ψ) == nfact || throw(DimensionMismatch("Number of factor nodes in factor graph `g`, $nfact, does not match length of `ψ`, $(length(ψ))"))
        length(ϕ) == nvar || throw(DimensionMismatch("Number of variable nodes in factor graph `g`, $nvar, does not match length of `ϕ`, $(length(ϕ))"))
        length(u) == nedges || throw(DimensionMismatch("Number of edges in factor graph `g`, $nvar, does not match length of `u`, $(length(u))"))
        length(h) == nedges || throw(DimensionMismatch("Number of edges in factor graph `g`, $nvar, does not match length of `h`, $(length(h))"))
        length(b) == nvar || throw(DimensionMismatch("Number of variable nodes in factor graph `g`, $nvar, does not match length of `b`, $(length(b))"))
        length(f) == nvert || throw(DimensionMismatch("Number of nodes in factor graph `g`, $nvert, does not match length of `f`, $(length(f))"))
        new{F,M,G,FV,MB,T}(g, ψ, ϕ, u, h, b, f)
    end
end

function BP(g, ψ, qs; ϕ = [UniformVertexFactor(q) for q in qs])
    u = [ones(qs[e.i]) for e in edges(g)]
    h = [ones(qs[e.i]) for e in edges(g)]
    b = [ones(qs[i.i]) for i in variables(g)]
    f = zeros(nv(g))
    return BP(g, ψ, ϕ, u, h, b, f)
end

nstates(bp::BP, i::Integer) = length(bp.b[i])
beliefs(bp::BP) = bp.b

function update_variable!(bp::BP, i::Integer)
    (; g, ϕ, u, h, b) = bp
    ∂i = edges(g, Variable(i))
    msg_mult(m1, m2) = m1 .* m2
    ϕᵢ = [ϕ[i](x) for x in 1:nstates(bp, i)]
    h[idx.(∂i)], b[i] = cavity(u[idx.(∂i)], msg_mult, ϕᵢ)
    for hᵢₐ in h[idx.(∂i)]
        hᵢₐ ./= sum(hᵢₐ)
    end
    b[i] ./= sum(b[i])
    return nothing
end

function update_factor!(bp::BP, a::Integer)
    (; g, ψ, u, h) = bp
    ∂a = inedges(g, Factor(a))
    ψₐ = ψ[a]
    for ai in ∂a
        u[idx(ai)] .= 0
    end
    for xₐ in Iterators.product((1:nstates(bp, e.i) for e in ∂a)...)
        for (i, ai) in pairs(∂a)
            u[idx(ai)][xₐ[i]] += ψₐ(xₐ) * 
                prod(h[idx(ja)][xₐ[j]] for (j, ja) in pairs(∂a) if j != i)
        end
    end
    for uₐᵢ in u[idx.(∂a)]
        uₐᵢ ./= sum(uₐᵢ)
    end
    return nothing
end

function iterate!(bp::BP; maxiter=100)
    for it in 1:maxiter
        for i in variables(bp.g)
            update_variable!(bp, i.i)
        end
        for a in factors(bp.g)
            update_factor!(bp, a.a)
        end
    end
    return nothing
end

function factor_beliefs(bp::BP)
    (; g, ψ, h) = bp
    return map(factors(g)) do fa
        a = fa.a
        ∂a = edges(g, Factor(a))
        ψₐ = ψ[a]
        bₐ = map(Iterators.product((1:nstates(bp, e.i) for e in ∂a)...)) do xₐ
            ψₐ(xₐ) * prod(h[idx(ia)][xₐ[i]] for (i, ia) in pairs(∂a))
        end
        zₐ = sum(bₐ)
        bₐ ./= zₐ
        bₐ
    end
end