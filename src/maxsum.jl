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
    ei = edge_indices(g, variable(i))
    ∂i = neighbors(g, variable(i))
    logϕᵢ = [log(ϕ[i](x)) + b[i][x]*rein for x in 1:nstates(bp, i)]
    msg_sum(m1, m2) = m1 .+ m2
    bnew[i] = @views cavity!(hnew[ei], u[ei], msg_sum, logϕᵢ)
    d = (degree(g, factor(a)) for a in ∂i)
    errv = typemin(eltype(bp))
    for (ia, dₐ) in zip(ei, d)
        fᵢ₂ₐ = maximum(hnew[ia])
        f[i] -= fᵢ₂ₐ * (1 - 1/dₐ)
        hnew[ia] .-= fᵢ₂ₐ
        errv = max(errv, maximum(abs, hnew[ia] - h[ia]))
        h[ia] = damp!(h[ia], hnew[ia], damp)
    end
    fᵢ  = maximum(bnew[i])
    f[i] -= fᵢ * (1 - degree(g, variable(i)) + sum(1/dₐ for dₐ in d; init=0.0))
    bnew[i] .-= fᵢ
    errb = maximum(abs, bnew[i] - b[i])
    b[i] = bnew[i]
    return errv, errb
end

function update_f_ms!(bp::BP, a::Integer, unew, damp::Real, f::AtomicVector{<:Real};
        extra_kwargs...)
    (; g, ψ, u, h) = bp
    ∂a = neighbors(g, factor(a))
    ea = edge_indices(g, factor(a))
    ψₐ = ψ[a]
    for ai in ea
        unew[ai] .= typemin(eltype(bp))
    end
    
    for xₐ in Iterators.product((1:nstates(bp, i) for i in ∂a)...)
        for (i, ai) in pairs(ea)
            unew[ai][xₐ[i]] = max(u[ai][xₐ[i]], log(ψₐ(xₐ)) + 
                sum(h[ja][xₐ[j]] for (j, ja) in pairs(ea) if j != i; init=0.0))
        end
    end
    dₐ = degree(g, factor(a))
    err = typemin(eltype(bp))
    for (i, ai) in zip(∂a, ea)
        fₐ₂ᵢ = maximum(unew[ai])
        f[i] -= fₐ₂ᵢ / dₐ
        unew[ai] .-= fₐ₂ᵢ
        err = max(err, maximum(abs, unew[ai] - u[ai]))
        u[ai] = damp!(u[ai], unew[ai], damp)
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
        ∂a = neighbors(g, factor(a))
        ea = edge_indices(g, factor(a))
        ψₐ = ψ[a]
        bₐ = map(Iterators.product((1:nstates(bp, i) for i in ∂a)...)) do xₐ
            log(ψₐ(xₐ)) + sum(h[ia][xₐ[i]] for (i, ia) in pairs(ea); init=zero(eltype(bp)))
        end
        zₐ = maximum(bₐ)
        bₐ .-= zₐ
        bₐ
    end
end

function avg_energy_ms(bp::BP; fb = factor_beliefs_ms(bp), b = beliefs_ms(bp))
    (; g, ψ, ϕ) = bp
    eₐ = eᵢ = 0.0
    for a in factors(g)
        bₐ = fb[a]
        xmax = argmax(bₐ) |> Tuple
        eₐ -= log(ψ[a](xmax)) 
    end
    eₐ *= _free_energy_correction(bp)
    for i in variables(g)
        bᵢ = b[i]
        xmax = argmax(bᵢ)
        eᵢ -= log(ϕ[i](xmax))
    end
    return eₐ + eᵢ
end

function bethe_free_energy_ms(bp::BP; fb = factor_beliefs_ms(bp), b = beliefs_ms(bp))
    return avg_energy_ms(bp; fb, b)
end