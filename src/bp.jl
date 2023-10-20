"""
    BP{F<:BPFactor, FV<:BPFactor, M, MB, G<:FactorGraph}

A type representing the state of the Belief Propagation algorithm.

Fields
========

- `g`: a [`FactorGraph`](@ref)
- `ψ`: a vector of [`BPFactor`](@ref) representing the factors {ψₐ(xₐ)}ₐ
- `ϕ`: a vector of [`BPFactor`](@ref) representing the single-variable factors {ϕᵢ(xᵢ)}ᵢ
- `u`: messages from factor to variable
- `h`: messages from variable to factor
- `b`: beliefs
"""
struct BP{F<:BPFactor, FV<:BPFactor, M, MB, G<:FactorGraph}
    g :: G               # graph
    ψ :: Vector{F}       # factors
    ϕ :: Vector{FV}      # vertex-dependent factors
    u :: AtomicVector{M}       # messages factor -> variable
    h :: AtomicVector{M}       # messages variable -> factor
    b :: Vector{MB}      # beliefs

    function BP(g::G, ψ::Vector{F}, ϕ::Vector{FV}, u::Vector{M}, h::Vector{M}, 
        b::Vector{MB}) where {G<:FactorGraph, F<:BPFactor, FV<:BPFactor, M, MB}

        nvar = nvariables(g)
        nfact = nfactors(g)
        nedges = ne(g)
        length(ψ) == nfact || throw(DimensionMismatch("Number of factor nodes in factor graph `g`, $nfact, does not match length of `ψ`, $(length(ψ))"))
        length(ϕ) == nvar || throw(DimensionMismatch("Number of variable nodes in factor graph `g`, $nvar, does not match length of `ϕ`, $(length(ϕ))"))
        length(u) == nedges || throw(DimensionMismatch("Number of edges in factor graph `g`, $nvar, does not match length of `u`, $(length(u))"))
        length(h) == nedges || throw(DimensionMismatch("Number of edges in factor graph `g`, $nvar, does not match length of `h`, $(length(h))"))
        length(b) == nvar || throw(DimensionMismatch("Number of variable nodes in factor graph `g`, $nvar, does not match length of `b`, $(length(b))"))
        new{F,FV,M,MB,G}(g, ψ, ϕ, AtomicVector(u), AtomicVector(h), b)
    end
end

"""
    BP(g::FactorGraph, ψ::AbstractVector{<:BPFactor}, qs; ϕ)

Constructor for the BP type.

Arguments
========

- `g`: a [`FactorGraph`](@ref)
- `ψ`: a vector of [`BPFactor`](@ref) representing the factors {ψₐ(xₐ)}ₐ
- `states`: an iterable of integers of length equal to the number of variable nodes specifyig the number of values each variable can take 
- `ϕ`: (optional) a vector of [`BPFactor`](@ref) representing the single-variable factors {ϕᵢ(xᵢ)}ᵢ
"""
function BP(g::FactorGraph, ψ::AbstractVector{<:BPFactor}, states;
        ϕ = [UniformFactor(q) for q in states])
    length(states) == nvariables(g) || throw(ArgumentError("Length of `states` must match number of variable nodes, got $(length(states)) and $(nvariables(g))"))
    u = [1/states[dst(e)]*ones(states[dst(e)]) for e in edges(g)]
    h = [1/states[dst(e)]*ones(states[dst(e)]) for e in edges(g)]
    b = [1/states[i]*ones(states[i]) for i in variables(g)]
    return BP(g, ψ, ϕ, u, h, b)
end

# treat a BP object as a scalar in broadcasting
Base.broadcastable(b::BP) = Ref(b)

"""
    reset!(bp::BP)

Reset all messages and beliefs to zero
"""
function reset!(bp::BP)
    (; u, h, b) = bp
    for uai in u; uai .= 1 / length(uai); end
    for hia in h; hia .= 1 / length(hia); end
    for bi in b; bi .= 1 / length(bi); end
    return nothing
end

"""
    nstates(bp::BP, i::Integer)

Return the number of values taken by variable `i`.
"""
nstates(bp::BP, i::Integer) = length(bp.b[i])

"""
    beliefs([f], bp::BP)

Return single-variable beliefs {bᵢ(xᵢ)}ᵢ.
"""
beliefs(f, bp::BP) = f(bp)

"""
    factor_beliefs([f], bp::BP)

Return factor beliefs {bₐ(xₐ)}ₐ.
"""
factor_beliefs(f, bp::BP) = f(bp)

"""
    avg_energy([f], bp::BP)

Return the average energy ∑ₐ∑ₓₐbₐ(xₐ)[-logψₐ(xₐ)] + ∑ᵢ∑ₓᵢbᵢ(xᵢ)[-logϕᵢ(xᵢ)]
"""
avg_energy(f, bp::BP; kwargs...) = f(bp; kwargs...)

"""
    bethe_free_energy([f], bp::BP)

Return the bethe free energy ∑ₐ∑ₓₐbₐ(xₐ)log[bₐ(xₐ)/ψₐ(xₐ)] + ∑ᵢ∑ₓᵢbᵢ(xᵢ)log[bᵢ(xᵢ)^(1-|∂i|)/ϕᵢ(xᵢ)]
"""
bethe_free_energy(f, bp::BP; kwargs...) = f(bp; kwargs...)

beliefs_bp(bp::BP) = bp.b
beliefs(bp::BP) = beliefs(beliefs_bp, bp)

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

function bethe_free_energy_bp(bp::BP; fb = factor_beliefs(bp), b = beliefs(bp))
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
bethe_free_energy(bp::BP) = bethe_free_energy(bethe_free_energy_bp, bp)

"""
    energy(bp::BP, x)

Return the energy ∑ₐ[-logψₐ(xₐ)] + ∑ᵢ[-logϕᵢ(xᵢ)] of configuration `x`.
"""
function energy(bp::BP, x)
    (; g, ψ, ϕ) = bp
    w = 0.0
    for a in factors(g)
        ∂a = neighbors(g, factor(a))
        w += -log(ψ[a](x[∂a]))
    end
    for i in variables(g)
        w += -log(ϕ[i](x[i]))
    end
    return w
end

"""
    evaluate(bp::BP, x)

Return the unnormalized probability ∏ₐψₐ(xₐ)∏ᵢϕᵢ(xᵢ) of configuration `x`.
"""
evaluate(bp::BP, x) = exp(-energy(bp, x))


abstract type ConvergenceChecker end

struct MessageConvergence{T<:Real} <: ConvergenceChecker
    tol :: T
end
message_convergence(tol::Real) = MessageConvergence(tol)

function (check_convergence::MessageConvergence)(::BP, errv, errf, errb)
    max(maximum(errv), maximum(errf)) < check_convergence.tol
end

struct BeliefConvergence{T<:Real} <: ConvergenceChecker
    tol :: T
end
belief_convergence(tol::Real) = BeliefConvergence(tol)

function (check_convergence::BeliefConvergence)(::BP, errv, errf, errb)
    maximum(errb) < check_convergence.tol
end

"""
    iterate!(bp::BP; kwargs...)

Run BP.

Optional arguments
=================

- `update_variable!`: the function that computes and updates variable-to-factor messages
- `update_factor!`: the function that computes and updates factor-to-variable messages
- `maxiter`: maximum number of iterations
- `tol`: convergence check parameter
- `damp`: damping parameter
- `rein`: reinforcement parameter
- `f`: a vector to store on-the-fly computations of the bethe free energy
- `callback`
- `check_convergence`: a function that checks if convergence has been reached
- extra arguments to be passed to custom `update_variable!` and `update_factor!`
"""
function iterate!(bp::BP; update_variable! = update_v_bp!, update_factor! = update_f_bp!,
        maxiter=100, tol=1e-6, damp::Real=0.0, rein::Real=0.0,
        f::AbstractVector{<:Real} = zeros(nvariables(bp.g)),
        callback = (bp, errv, errf, errb, it, f) -> nothing,
        check_convergence=message_convergence(tol),
        extra_kwargs...
        )
    (; g, u, h, b) = bp
    unew = copy(u); hnew = copy(h); bnew = copy(b)
    errv = zeros(nvariables(g)); errf = zeros(nfactors(g)); errb = zeros(nvariables(g))
    ff = AtomicVector(f)
    for it in 1:maxiter
        ff .= 0
        @threads for i in variables(bp.g)
            errv[i], errb[i] = update_variable!(bp, i, hnew, bnew, damp, rein*it, ff; extra_kwargs...)
        end
        @threads for a in factors(bp.g)
            errf[a] = update_factor!(bp, a, unew, damp, ff; extra_kwargs...)
        end
        callback(bp, errv, errf, errb, it, f)
        check_convergence(bp, errv, errf, errb) && return it
    end
    return maxiter
end

function damp!(x::Real, xnew::Real, damp::Real)
    0 ≤ damp ≤ 1 || throw(ArgumentError("Damping factor must be in [0,1], got $damp"))
    damp == 0 && return xnew
    return xnew * (1-damp) + x * damp
end

function damp!(x::T, xnew::T, damp::Real) where {T<:AbstractVector}
    0 ≤ damp ≤ 1 || throw(ArgumentError("Damping factor must be in [0,1], got $damp"))
    if damp != 0
        for (xi, xinew) in zip(x, xnew)
            xinew = xinew * (1-damp) + xi * damp
        end
    end
    x, xnew = xnew, x
    return x
end

function update_v_bp!(bp::BP{F,FV,M,MB}, i::Integer, hnew, bnew, damp::Real, rein::Real,
        f::AtomicVector{<:Real}; extra_kwargs...) where {
        F<:BPFactor, FV<:BPFactor, M<:AbstractVector{<:Real}, MB<:AbstractVector{<:Real}}
    (; g, ϕ, u, h, b) = bp
    ∂i = outedges(g, variable(i)) 
    ϕᵢ = [ϕ[i](x) * b[i][x]^rein for x in 1:nstates(bp, i)]
    msg_mult(m1, m2) = m1 .* m2
    hnew[idx.(∂i)], bnew[i] = cavity(u[idx.(∂i)], msg_mult, ϕᵢ)
    d = (degree(g, factor(a)) for a in neighbors(g, variable(i)))
    errv = -Inf
    for ((_,_,id), dₐ) in zip(∂i, d)
        zᵢ₂ₐ = sum(hnew[id])
        f[i] -= log(zᵢ₂ₐ) * (1 - 1/dₐ)
        hnew[id] ./= zᵢ₂ₐ
        errv = max(errv, mean(abs, hnew[id] - h[id]))
        h[id] = damp!(h[id], hnew[id], damp)
    end
    errb = mean(abs, bnew[i] - b[i])
    zᵢ = sum(bnew[i])
    f[i] -= log(zᵢ) * (1 - degree(g, variable(i)) + sum(1/dₐ for dₐ in d; init=0.0))
    b[i] = bnew[i]
    b[i] ./= zᵢ
    return errv, errb
end

function update_f_bp!(bp::BP{F,FV,M,MB}, a::Integer, unew, damp::Real,
        f::AtomicVector{<:Real}; extra_kwargs...) where {
            F<:BPFactor, FV<:BPFactor, M<:AbstractVector{<:Real}, MB<:AbstractVector{<:Real}}
    (; g, ψ, u, h) = bp
    ∂a = inedges(g, factor(a))
    ψₐ = ψ[a]
    for ai in ∂a
        unew[idx(ai)] .= 0
    end
    for xₐ in Iterators.product((1:nstates(bp, src(e)) for e in ∂a)...)
        for (i, ai) in pairs(∂a)
            unew[idx(ai)][xₐ[i]] += ψₐ(xₐ) * 
                prod(h[idx(ja)][xₐ[j]] for (j, ja) in pairs(∂a) if j != i; init=1.0)
        end
    end
    dₐ = degree(g, factor(a))
    err = -Inf
    for (i, _, id) in ∂a
        zₐ₂ᵢ = sum(unew[id])
        f[i] -= log(zₐ₂ᵢ) / dₐ
        unew[id] ./= zₐ₂ᵢ
        err = max(err, mean(abs, unew[id] - u[id]))
        u[id] = damp!(u[id], unew[id], damp)
    end
    return err
end