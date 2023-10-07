function iterate_ms!(bp; kwargs...) 
    return iterate!(bp; update_variable! = update_v_ms!, update_factor! = update_f_ms!,
        kwargs...)
end

function update_v_ms!(bp::BP, i::Integer)
    (; g, ϕ, u, h, b) = bp
    ∂i = outedges(g, variable(i))
    logϕᵢ = [log(ϕ[i](x)) for x in 1:nstates(bp, i)]
    msg_sum(m1, m2) = m1 .+ m2
    h[idx.(∂i)], b[i] = cavity(u[idx.(∂i)], msg_sum, logϕᵢ)
    for hᵢₐ in h[idx.(∂i)]
        hᵢₐ .-= maximum(hᵢₐ)
    end
    zᵢ = maximum(b[i])
    b[i] .-= zᵢ
    return nothing
end

function update_f_ms!(bp::BP, a::Integer)
    (; g, ψ, u, h) = bp
    ∂a = inedges(g, factor(a))
    ψₐ = ψ[a]
    for ai in ∂a
        u[idx(ai)] .= -Inf
    end
    for xₐ in Iterators.product((1:nstates(bp, src(e)) for e in ∂a)...)
        for (i, ai) in pairs(∂a)
            u[idx(ai)][xₐ[i]] = max(u[idx(ai)][xₐ[i]],  log(ψₐ(xₐ)) + 
                sum(h[idx(ja)][xₐ[j]] for (j, ja) in pairs(∂a) if j != i) )
        end
    end
    for uₐᵢ in u[idx.(∂a)]
        uₐᵢ .-= maximum(uₐᵢ)
    end
    return nothing
end

beliefs_ms(bp) = bp.b

function factor_beliefs_ms(bp::BP)
    (; g, ψ, h) = bp
    return map(factors(g)) do a
        ∂a = inedges(g, factor(a))
        ψₐ = ψ[a]
        bₐ = map(Iterators.product((1:nstates(bp, src(e)) for e in ∂a)...)) do xₐ
            log(ψₐ(xₐ)) + sum(h[idx(ia)][xₐ[i]] for (i, ia) in pairs(∂a))
        end
        zₐ = maximum(bₐ)
        bₐ .-= zₐ
        bₐ
    end
end

function avg_energy_ms(bp::BP; fb = factor_beliefs_ms(bp), b = beliefs_ms(bp))
    (; g, ψ, ϕ) = bp
    e = 0.0
    for a in factors(g)
        ∂a = inedges(g, factor(a))
        bₐ = fb[a]
        bmax = maximum(bₐ)
        e -= mean(log(ψ[a](xₐ)) 
            for xₐ in Iterators.product((1:nstates(bp, src(e)) for e in ∂a)...)
            if bₐ[xₐ...] == bmax
        )
    end
    for i in variables(g)
        bᵢ = b[i]
        bmax = maximum(bᵢ)
        e -= mean(log(ϕ[i](xᵢ)) for xᵢ in eachindex(bᵢ) if bᵢ[xᵢ] == bmax)
    end
    return e
end