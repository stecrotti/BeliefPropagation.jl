"""
    iterate_ms!(bp::BP; kwargs...)

Runs the max-sum algorithm (BP at zero temperature). 
"""
function iterate_ms!(bp::BP; kwargs...) 
    return iterate!(bp; update_variable! = update_v_ms!, update_factor! = update_f_ms!,
        compute_fai! = compute_fai_ms!, kwargs...)
end

function update_v_ms!(bp::BP, i::Integer, hnew, bnew, damp::Real, rein::Real,
        f::BetheFreeEnergy; extra_kwargs...)
    (; g, ϕ, u, h, b) = bp
    ei = edge_indices(g, variable(i))
    logϕᵢ = [log(ϕ[i](x)) + b[i][x]*rein for x in 1:nstates(bp, i)]
    msg_sum(m1, m2) = m1 .+ m2
    bnew[i] = @views cavity!(hnew[ei], u[ei], msg_sum, logϕᵢ)
    logzᵢ = maximum(bnew[i])
    bnew[i] .-= logzᵢ
    f.variables[i] = -logzᵢ
    errb = maximum(abs, bnew[i] - b[i])
    b[i] = bnew[i]
    errv = typemin(eltype(bp))
    for ia in ei
        logzᵢ₂ₐ = maximum(hnew[ia])
        hnew[ia] .-= logzᵢ₂ₐ
        errv = max(errv, maximum(abs, hnew[ia] - h[ia]))
        h[ia] = damp!(h[ia], hnew[ia], damp)
    end
    return errv, errb
end

struct MS <: Real
    val::Float64
    MS(v::Float64) = new(v)
end

Base.:(+)(x::MS, y::MS) = MS(max(x.val, y.val))
Base.:(+)(x::MS, y::Float64) = MS(max(x.val, log(y)))
Base.:(+)(y::Float64, x::MS) = x + y
Base.:(*)(x::MS, y::MS) = MS(x.val + y.val)
Base.:(*)(x::MS, y::Float64) = MS(x.val + log(y))
Base.:(*)(y::Float64, x::MS) = x * y
Base.one(::Type{MS}) = MS(0.0)
Base.zero(::Type{MS}) = MS(-Inf)


function update_f_ms!(bp::BP, a::Integer, unew, damp::Real,
        f::BetheFreeEnergy; extra_kwargs...)
    (; g, ψ, u, h) = bp
    ea = edge_indices(g, factor(a))
    hflat = mortar([mappedarray(MS, h[ai]) for ai in ea])
    uflat = mortar([mappedarray(x->MS(x), x->x.val, unew[ai]) for ai in ea])
    ret = DiffResults.DiffResult(MS(0.0), uflat)
    ForwardDiff.gradient!(ret, hin -> compute_za(ψ[a], hin.blocks), hflat)
    logzₐ = DiffResults.value(ret).val
    f.factors[a] = -logzₐ
    err = typemin(eltype(bp))
    for ai in ea
        unew[ai] .-= maximum(unew[ai])
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
    eₐ *= _free_energy_correction_factors(bp)
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

function compute_fai_ms!(f::BetheFreeEnergy, bp::BPGeneric)
    fai = f.edges
    for (ai, uai, hia) in zip(eachindex(fai), bp.u, bp.h)
        fai[ai] = -maximum(uaix + hiax for(uaix, hiax) in zip(uai, hia)) 
    end
    return nothing
end
