struct BP{G, F, FV, M, MB, T}
    g :: G                              # graph
    ψ :: Vector{F}                      # factors
    ϕ :: Vector{FV}      # vertex-dependent factors
    u :: Vector{M}                      # messages factor -> variable
    h :: Vector{M}                      # messages variable -> factor
    b :: Vector{MB}                     # beliefs
    f :: Vector{T}                      # free energy contributions
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
    (; g, ψ, ϕ, u, h, b, f) = bp
    ein = inedges(g, Variable(i))
    eout = outedges(g, Variable(i))
    msg_mult(m1, m2) = m1 .* m2
    ϕᵢ = [ϕ[i](x) for x in 1:nstates(bp, i)]
    b[i] = cavity!(h[idx.(eout)], u[idx.(ein)], msg_mult, ϕᵢ)
    for hᵢₐ in h[idx.(eout)]
        hᵢₐ ./= sum(hᵢₐ)
    end
    b[i] ./= sum(b[i])
    return nothing
end

function update_factor!(bp::BP, a::Integer)
    (; g, ψ, ϕ, u, h, b, f) = bp
    ein = inedges(g, Factor(a))
    eout = outedges(g, Factor(a))
    for ai in eout
        u[idx(ai)] .= 0
    end
    for xₐ in Iterators.product((1:nstates(bp, e.i) for e in ein)...)
        for (ia, ai) in zip(ein, eout)
            u[idx(ai)] .+= ψ[a](xₐ) * 
                reduce(.*, h[idx(ja)][xₐ[j]] for (j, ja) in pairs(ein) if ja != ia)
        end
    end
    for uₐᵢ in u[idx.(eout)]
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