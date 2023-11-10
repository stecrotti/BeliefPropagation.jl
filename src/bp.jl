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
struct BP{F<:BPFactor, FV<:BPFactor, M, MB, G<:AbstractFactorGraph}
    g :: G
    ψ :: Vector{F}
    ϕ :: Vector{FV}
    u :: Vector{M}
    h :: Vector{M}
    b :: Vector{MB}

    function BP(g::G, ψ::Vector{F}, ϕ::Vector{FV}, u::Vector{M}, h::Vector{M}, 
        b::Vector{MB}) where {G<:AbstractFactorGraph, F<:BPFactor, FV<:BPFactor, M, MB}

        nvar = nvariables(g)
        nfact = nfactors(g)
        nedges = ne(g)
        length(ψ) == nfact || throw(DimensionMismatch("Number of factor nodes in factor graph `g`, $nfact, does not match length of `ψ`, $(length(ψ))"))
        length(ϕ) == nvar || throw(DimensionMismatch("Number of variable nodes in factor graph `g`, $nvar, does not match length of `ϕ`, $(length(ϕ))"))
        length(u) == nedges || throw(DimensionMismatch("Number of edges in factor graph `g`, $nvar, does not match length of `u`, $(length(u))"))
        length(h) == nedges || throw(DimensionMismatch("Number of edges in factor graph `g`, $nvar, does not match length of `h`, $(length(h))"))
        length(b) == nvar || throw(DimensionMismatch("Number of variable nodes in factor graph `g`, $nvar, does not match length of `b`, $(length(b))"))
        new{F,FV,M,MB,G}(g, ψ, ϕ, u, h, b)
    end
end

Base.eltype(bp::BP) = eltype(eltype(bp.b))

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
function BP(g::AbstractFactorGraph, ψ::AbstractVector{<:BPFactor}, states;
        ϕ = fill(UniformFactor(), nvariables(g)))
    length(states) == nvariables(g) || throw(ArgumentError("Length of `states` must match number of variable nodes, got $(length(states)) and $(nvariables(g))"))
    T = promote_type(eltype(ψ[1]), eltype(ϕ[1]))
    all(eltype(ψₐ) == eltype(ψ[1]) for ψₐ in ψ) || @warn "Possible type issues. Check that all the factors in ψ have the same type"
    all(eltype(ϕᵢ) == eltype(ϕ[1]) for ϕᵢ in ϕ) || @warn "Possible type issues. Check that all the factors in ϕ have the same type"
    u = [fill(one(T)/states[dst(e)], states[dst(e)]) for e in edges(g)]
    h = [fill(one(T)/states[dst(e)], states[dst(e)]) for e in edges(g)]
    b = [fill(one(T)/states[i], states[i]) for i in variables(g)]
    return BP(g, ψ, ϕ, u, h, b)
end

# BP where messages are vectors of reals
const BPGeneric = BP{<:BPFactor, <:BPFactor, <:AbstractVector{<:Real}, <:AbstractVector{<:Real}, <:AbstractFactorGraph}

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

Return single-variable beliefs ``\\{b_i(x_i)\\}_i``.
"""
beliefs(f::Function, bp::BP) = f(bp)

"""
    factor_beliefs([f], bp::BP)

Return factor beliefs ``\\{b_a(\\underline{x}_a)\\}_a``.
"""
factor_beliefs(f::Function, bp::BP) = f(bp)

@doc raw"""
    avg_energy([f], bp::BP)

Return the average energy
```math
\sum_a\sum_{\underline{x}_a}b_a(\underline{x}_a) \left[-\log\psi_a(\underline{x}_a)\right] + \sum_i\sum_{x_i}b_i(x_i) \left[-\log\phi_i(x_i)\right]
```
"""
avg_energy(f::Function, bp::BP; kwargs...) = f(bp; kwargs...)

@doc raw"""
    bethe_free_energy([f], bp::BP)

Return the bethe free energy
```math
\sum_a\sum_{\underline{x}_a}b_a(\underline{x}_a) \left[-\log\frac{b_a(\underline{x}_a)}{\psi_a(\underline{x}_a)}\right] + \sum_i\sum_{x_i}b_i(x_i) \left[-\log\frac{b_i(x_i)^{1-\lvert\partial i\rvert}}{\phi_i(x_i)}\right]
```
"""
bethe_free_energy(f::Function, bp::BP; kwargs...) = f(bp; kwargs...)

beliefs_bp(bp::BP) = bp.b
beliefs(bp::BP) = beliefs(beliefs_bp, bp)

function factor_beliefs_bp(bp::BP)
    (; g, ψ, h) = bp
    return map(factors(g)) do a
        ∂a = neighbors(g, factor(a))
        ea = edge_indices(g, factor(a))
        ψₐ = ψ[a]
        bₐ = map(Iterators.product((1:nstates(bp, i) for i in ∂a)...)) do xₐ
            ψₐ(xₐ) * prod(h[ia][xₐ[i]...] for (i, ia) in pairs(ea); init=one(eltype(bp)))
        end
        zₐ = sum(bₐ)
        bₐ ./= zₐ
        bₐ
    end
end
factor_beliefs(bp::BP) = factor_beliefs(factor_beliefs_bp, bp)

function avg_energy_bp(bp::BP; fb = factor_beliefs(bp), b = beliefs(bp))
    (; g, ψ, ϕ) = bp
    eₐ = eᵢ = 0.0
    for a in factors(g)
        ∂a = neighbors(g, factor(a))
        for xₐ in Iterators.product((1:nstates(bp, i) for i in ∂a)...)
            eₐ += -xlogy(fb[a][xₐ...], ψ[a](xₐ))
        end
    end
    eₐ *= _free_energy_correction_factors(bp)
    for i in variables(g)
        for xᵢ in eachindex(b[i])
            eₐ += -xlogy(b[i][xᵢ], ϕ[i](xᵢ))
        end
    end
    return eₐ + eᵢ
end
avg_energy(bp::BP) = avg_energy(avg_energy_bp, bp)

_free_energy_correction_factors(bp::BP{F, FV, M, MB, G}) where {F, FV, M, MB, G} = 1
_free_energy_correction_edges(bp::BP{F, FV, M, MB, G}) where {F, FV, M, MB, G} = 1

const BPRegular{F, FV, M, MB} = BP{F, FV, M, MB, G} where {F, FV, M, MB, G<:RegularFactorGraph}
_free_energy_correction_factors(bp::BPRegular) = bp.g.kᵢ / bp.g.kₐ
_free_energy_correction_edges(bp::BPRegular) = bp.g.kᵢ

function bethe_free_energy_bp(bp::BP; fb = factor_beliefs(bp), b = beliefs(bp))
    (; g, ψ, ϕ) = bp
    fₐ = fᵢ = 0.0
    for a in factors(g)
        ∂a = neighbors(g, factor(a))
        for xₐ in Iterators.product((1:nstates(bp, i) for i in ∂a)...)
            fₐ += xlogx(fb[a][xₐ...]) - xlogy(fb[a][xₐ...], ψ[a](xₐ))
        end
    end
    fₐ *= _free_energy_correction_factors(bp)

    for i in variables(g)
        dᵢ = degree(g, variable(i))
        for xᵢ in eachindex(b[i])
            fᵢ += (1-dᵢ) * xlogx(b[i][xᵢ]) - xlogy(b[i][xᵢ], ϕ[i](xᵢ))
        end
    end
    return fₐ + fᵢ
end
bethe_free_energy(bp::BP) = bethe_free_energy(bethe_free_energy_bp, bp)

@doc raw"""
    energy(bp::BP, x)

Return the energy
```math
\sum_a \left[-\log\psi_a(\underline{x}_a)\right] + \sum_i \left[-\log\phi_i(x_i)\right]
```
of configuration `x`.
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

Return the unnormalized probability ``\\prod_a\\psi_a(\\underline{x}_a)\\prod_i\\phi_i(x_i)`` of configuration `x`.
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


struct BetheFreeEnergy{T<:AbstractVector{<:Real}, TF<:Real, TE<:Real}
    factors         :: T
    variables       :: T
    edges           :: T
    corr_factors    :: TF
    corr_edges      :: TE
end

"""
    init_free_energy(bp::BP)

Return a `BeliefPropagation.BetheFreeEnergy` which can be used to compute the Bethe Free Energy using message normalizations. In particular, this avoids explicit computation of factor beliefs, whose cost is exponential in the factor degree.

Example
======
```jldoctest init_free_energy
julia> using BeliefPropagation, BeliefPropagation.FactorGraphs, BeliefPropagation.Models

julia> using Random: MersenneTwister

julia> g = rand_factor_graph(MersenneTwister(0), 10, 15, 20);

julia> ψ = IsingCoupling.(randn(MersenneTwister(0), nfactors(g)));

julia> bp = BP(g, ψ, fill(2, nvariables(g)));

julia> f = init_free_energy(bp);

julia> iterate!(bp; maxiter=30, tol=1e-16, f);

julia> @assert sum(f) ≈ bethe_free_energy(bp)
```
"""
function init_free_energy(bp::BP)
    T = eltype(bp)
    a = zeros(T, nfactors(bp.g))
    i = zeros(T, nvariables(bp.g))
    ai = zeros(T, ne(bp.g))
    return BetheFreeEnergy(a, i, ai, 
        _free_energy_correction_factors(bp), _free_energy_correction_edges(bp))
end
# Base.length(::BetheFreeEnergy) = 3
# Base.iterate(f::BetheFreeEnergy, args...) = iterate((f.factors, f.variables, f.edges), args...)

function bethe_free_energy(f::BetheFreeEnergy)
    return f.corr_factors * sum(f.factors) + sum(f.variables) - f.corr_edges * sum(f.edges)
end
Base.sum(f::BetheFreeEnergy) = bethe_free_energy(f)

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
function iterate!(bp::BP;
        update_variable! = update_v_bp!,
        update_factor! = update_f_bp!,
        compute_fai! = compute_fai_bp!,
        maxiter=100, tol=1e-6, damp::Real=0.0, rein::Real=0.0,
        f::BetheFreeEnergy = init_free_energy(bp),
        callback = (bp, errv, errf, errb, it, f) -> nothing,
        check_convergence=message_convergence(tol),
        extra_kwargs...
        )
    (; g, u, h, b) = bp
    T = eltype(bp)
    unew = deepcopy(u); hnew = deepcopy(h); bnew = deepcopy(b)
    errv = zeros(T, nvariables(g)); errf = zeros(T, nfactors(g))
    errb = zeros(T, nvariables(g))
    for it in 1:maxiter
        compute_fai!(f, bp)
        @threads for a in factors(bp.g)
            errf[a] = update_factor!(bp, a, unew, damp, f; extra_kwargs...)
        end
        @threads for i in variables(bp.g)
            errv[i], errb[i] = update_variable!(bp, i, hnew, bnew, damp, rein*it, f; extra_kwargs...)
        end
        callback(bp, errv, errf, errb, it, f)
        check_convergence(bp, errv, errf, errb) && return it
    end
    return maxiter
end

function compute_fai_bp!(f::BetheFreeEnergy, bp::BPGeneric)
    fai = f.edges
    for (ai, uai, hia) in zip(eachindex(fai), bp.u, bp.h)
        fai[ai] = -log(sum(uaix * hiax for(uaix, hiax) in zip(uai, hia))) 
    end
    return nothing
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

function set_messages_variable!(bp, ei, i, hnew, bnew, damp, f)
    (; h, b) = bp
    zᵢ = sum(bnew[i])
    bnew[i] ./= zᵢ
    errb = maximum(abs, bnew[i] - b[i])
    f.variables[i] = -log(zᵢ)
    b[i] = bnew[i]
    errv = zero(eltype(bp))
    for ia in ei
        hnew[ia] ./= sum(hnew[ia])
        errv = max(errv, maximum(abs, hnew[ia] - h[ia]))
        h[ia] = damp!(h[ia], hnew[ia], damp)
    end
    return errv, errb
end

function update_v_bp!(bp::BPGeneric, i::Integer, hnew, bnew, damp::Real, rein::Real,
        f::BetheFreeEnergy; extra_kwargs...)
    (; g, ϕ, u, b) = bp
    ei = edge_indices(g, variable(i)) 
    ϕᵢ = [ϕ[i](x) * b[i][x]^rein for x in 1:nstates(bp, i)]
    msg_mult(m1, m2) = m1 .* m2
    bnew[i] = @views cavity!(hnew[ei], u[ei], msg_mult, ϕᵢ)
    errv, errb = set_messages_variable!(bp, ei, i, hnew, bnew, damp, f)
    return errv, errb
end

function compute_za(ψₐ, msg_in)
    isempty(msg_in) && return one(eltype(ψₐ))
    return sum(ψₐ(xₐ) * prod(m[xᵢ] for (m, xᵢ) in zip(msg_in, xₐ)) 
        for xₐ in Iterators.product(eachindex.(msg_in)...))
end

function set_messages_factor!(bp, ea, unew, damp)
    u = bp.u
    err = zero(eltype(bp))
    for ai in ea
        unew[ai] ./= sum(unew[ai])
        err = max(err, maximum(abs, unew[ai] - u[ai]))
        u[ai] = damp!(u[ai], unew[ai], damp)
    end
    return err
end

function update_f_bp!(bp::BPGeneric, a::Integer, unew, damp::Real,
        f::BetheFreeEnergy; extra_kwargs...)
    (; g, ψ, h) = bp
    ea = edge_indices(g, factor(a))
    ψₐ = ψ[a]
    zₐ = compute_za(ψₐ, h[ea])
    hflat = @views mortar(h[ea])
    uflat = @views mortar(unew[ea])
    ForwardDiff.gradient!(uflat, hflat -> compute_za(ψₐ, hflat.blocks), hflat)
    f.factors[a] = -log(zₐ)
    err = set_messages_factor!(bp, ea, unew, damp)
    return err
end