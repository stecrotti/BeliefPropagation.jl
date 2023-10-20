"""
    iterate_ms!(bp::BP; kwargs...)

Runs the max-sum algorithm (BP at zero temperature). 
"""
function iterate_ms!(bp::BP; kwargs...) 
    return iterate!(bp; update_variable! = update_v_ms!, update_factor! = update_f_ms!,
        kwargs...)
end

function update_v_ms!(bp::BP, i::Integer, hnew, bnew, damp::Real, rein::Real,
        f::AtomicVector{<:Real}; extra_kwargs...)
    (; g, ϕ, u, h, b) = bp
    ∂i = outedges(g, variable(i))
    logϕᵢ = [log(ϕ[i](x)) + b[i][x]*rein for x in 1:nstates(bp, i)]
    msg_sum(m1, m2) = m1 .+ m2
    hnew[idx.(∂i)], bnew[i] = cavity(u[idx.(∂i)], msg_sum, logϕᵢ)
    d = (degree(g, factor(a)) for a in neighbors(g, variable(i)))
    errv = -Inf
    for ((_,_,id), dₐ) in zip(∂i, d)
        fᵢ₂ₐ = maximum(hnew[id])
        f[i] -= fᵢ₂ₐ * (1 - 1/dₐ)
        hnew[id] .-= fᵢ₂ₐ
        errv = max(errv, mean(abs, hnew[id] - h[id]))
        h[id] = damp!(h[id], hnew[id], damp)
    end
    errb = mean(abs, bnew[i] - b[i])
    fᵢ  = maximum(bnew[i])
    f[i] -= fᵢ * (1 - degree(g, variable(i)) + sum(1/dₐ for dₐ in d; init=0.0))
    b[i] = bnew[i]
    b[i] .-= fᵢ
    return errv, errb
end

function update_f_ms!(bp::BP, a::Integer, unew, damp::Real, f::AtomicVector{<:Real};
        extra_kwargs...)
    (; g, ψ, u, h) = bp
    ∂a = inedges(g, factor(a))
    ψₐ = ψ[a]
    for ai in ∂a
        unew[idx(ai)] .= -Inf
    end
    for xₐ in Iterators.product((1:nstates(bp, src(e)) for e in ∂a)...)
        for (i, ai) in pairs(∂a)
            u[idx(ai)][xₐ[i]] = max(u[idx(ai)][xₐ[i]],  log(ψₐ(xₐ)) + 
                sum(h[idx(ja)][xₐ[j]] for (j, ja) in pairs(∂a) if j != i; init=0.0))
        end
    end
    dₐ = degree(g, factor(a))
    err = -Inf
    for (i, _, id) in ∂a
        fₐ₂ᵢ = maximum(unew[id])
        f[i] -= fₐ₂ᵢ / dₐ
        unew[id] .-= fₐ₂ᵢ
        err = max(err, mean(abs, unew[id] - u[id]))
        u[id] = damp!(u[id], unew[id], damp)
    end
    return err
end

function beliefs_ms(bp::BP)
    return map(bp.b) do hᵢ
        bmax = maximum(hᵢ)
        bᵢ = [hᵢx == bmax ? 1.0 : 0.0 for hᵢx in hᵢ]
        bᵢ ./= sum(bᵢ)
    end
end

function factor_beliefs_ms(bp::BP)
    (; g, ψ, h) = bp
    return map(factors(g)) do a
        ∂a = inedges(g, factor(a))
        ψₐ = ψ[a]
        bₐ = map(Iterators.product((1:nstates(bp, src(e)) for e in ∂a)...)) do xₐ
            log(ψₐ(xₐ)) + sum(h[idx(ia)][xₐ[i]] for (i, ia) in pairs(∂a); init=0.0)
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
        bₐ = fb[a]
        xmax = argmax(bₐ) |> Tuple
        e -= log(ψ[a](xmax)) 
    end
    for i in variables(g)
        bᵢ = b[i]
        xmax = argmax(bᵢ)
        e -= log(ϕ[i](xmax))
    end
    return e
end

function bethe_free_energy_ms(bp::BP; fb = factor_beliefs_ms(bp), b = beliefs_ms(bp))
    return avg_energy_ms(bp; fb, b)
end