struct Decimation{TB<:AbstractVector{<:Bool}, TF1<:Real, TC<:ConvergenceChecker, TF2<:Real} <: Callback
    decimated    :: TB
    tol          :: TF1
    conv_checker :: TC
    softinf      :: TF2
end
function Decimation(n::Integer, tol::TF1, conv_checker::TC=MessageConvergence();
        softinf::TF2 = 1e8) where {TF1<:Real, TC<:ConvergenceChecker, TF2<:Real}
    decimated = falses(n)
    return Decimation{typeof(decimated), TF1, TC, TF2}(decimated, tol, conv_checker, softinf)
end

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