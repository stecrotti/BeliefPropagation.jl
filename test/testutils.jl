function bethe_free_energy_slow(bp::BP; fb = factor_beliefs(bp), b = beliefs(bp))
    (; g, ψ, ϕ) = bp
    f = 0.0
    for a in factors(g)
        ∂a = inedges(g, factor(a))
        for xₐ in Iterators.product((1:nstates(bp, src(e)) for e in ∂a)...)
            f += log(fb[a][xₐ...] / ψ[a](xₐ)) * fb[a][xₐ...]
        end
    end
    for i in variables(g)
        dᵢ = degree(g, variable(i))
        for xᵢ in eachindex(b[i])
            f += log((b[i][xᵢ])^(1-dᵢ) / ϕ[i](xᵢ)) * b[i][xᵢ]
        end
    end
    return f
end