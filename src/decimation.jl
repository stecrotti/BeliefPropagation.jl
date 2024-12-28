"""
    Decimation <: Callback

A callback that implements the decimation procedure: whenever the desired convergence
tolerance has been reached, the variable with the most biased belief is fixed to that value
by modifying the corresponding ϕ factor. The procedure is repeated until all variables are fixed.
The recommended constructor is `Decimation(n::Integer, tol::Real)`.
"""
mutable struct Decimation{TP<:Progress, TB<:AbstractVector{<:Bool}, TF1<:Real, 
        TC<:ConvergenceChecker, TF2<:Real, TI<:Integer} <: Callback
    prog         :: TP
    decimated    :: TB
    tol          :: TF1
    conv_checker :: TC
    softinf      :: TF2
    iters        :: TI
    converged    :: Bool
end

"""
    Decimation(n, maxiter, tol)

Return an instance of the [`Decimation`](@ref) callback.

Arguments
========

- `n`: total number of variables
- `maxiter`: maximum number of iterations
- `tol`: tolerance for convergence check

Optional arguments
========

- `conv_checker`: a [`ConvergenceChecker`](@ref)
- `softinf`: the real value used to fix variables
"""
function Decimation(n::Integer, maxiter::Integer, tol::Real, conv_checker=MessageConvergence();
        softinf::Real=1e8)
    prog = Progress(maxiter; desc="Running BP + decimation", dt=2)
    decimated = falses(n)
    return Decimation(prog, decimated, tol, conv_checker, softinf, 0, false)
end
Decimation(bp::BP, args...; kw...) = Decimation(nvariables(bp.g), args...; kw...)

function _find_most_biased(bp::BP, decimated)
    b = beliefs(bp)
    i = xi = m = 0
    for j in eachindex(b)
        if !decimated[j]
            mx = maximum(b[j])
            if mx > m
                m = mx
                i = j
            end
        end
    end
    _, xi = findmax(b[i]) 
    return i, xi
end
function _fix_variable!(bp::BP, i, xi, callback::Decimation)
    for x in 1:nstates(bp, i)
        bp.ϕ[i].values[x] = x == xi ? callback.softinf : 0
    end
end

function (callback::Decimation)(bp::BP{<:BPFactor,<:TabulatedBPFactor}, errv, errf, errb, it)
    ε = callback.conv_checker(bp, errv, errf, errb, it)
    if ε < callback.tol
        i, xi = _find_most_biased(bp, callback.decimated)
        _fix_variable!(bp, i, xi, callback)
        callback.decimated[i] = true
        if sum(callback.decimated) == nvariables(bp.g)
            return true
        end
    end
    return false
end