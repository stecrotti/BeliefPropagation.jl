# function bethe_free_energy_slow(bp::BP; fb = factor_beliefs(bp), b = beliefs(bp))
#     (; g, ψ, ϕ) = bp
#     f = 0.0
#     for a in factors(g)
#         ∂a = inedges(g, factor(a))
#         for xₐ in Iterators.product((1:nstates(bp, src(e)) for e in ∂a)...)
#             f += log(fb[a][xₐ...] / ψ[a](xₐ)) * fb[a][xₐ...]
#         end
#     end
#     for i in variables(g)
#         dᵢ = degree(g, variable(i))
#         for xᵢ in eachindex(b[i])
#             f += log((b[i][xᵢ])^(1-dᵢ) / ϕ[i](xᵢ)) * b[i][xᵢ]
#         end
#     end
#     return f
# end

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

evaluate(bp::BP, x) = exp(-energy(bp, x))

function eachstate(bp::BP)
    return Iterators.product((1:nstates(bp, i) for i in variables(bp.g))...)
end

function exact_normalization(bp::BP)
    return sum(evaluate(bp, x) for x in eachstate(bp))
end

function exact_prob(bp::BP; Z = exact_normalization(bp))
    p = [evaluate(bp, x) / Z for x in eachstate(bp)]
    return p
end

function exact_marginals(bp::BP; p_exact = exact_prob(bp))
    dims = 1:ndims(p_exact)
    return map(variables(bp.g)) do i
        vec(sum(p_exact; dims=dims[Not(i)]))
    end
end

function exact_factor_marginals(bp::BP; p_exact = exact_prob(bp))
    dims = 1:ndims(p_exact)
    return map(factors(bp.g)) do a
        ∂a = neighbors(bp.g, factor(a))
        dropdims(sum(p_exact; dims=dims[Not(∂a)]); dims=tuple(dims[Not(∂a)]...))
    end
end

function exact_avg_energy(bp::BP; p_exact = exact_prob(bp))
    k = keys(p_exact)
    sum(energy(bp, Tuple(k[x])) * p_exact[x] for x in eachindex(p_exact))
end

function exact_minimum_energy(bp::BP)
    return minimum(energy(bp, x) for x in eachstate(bp))
end