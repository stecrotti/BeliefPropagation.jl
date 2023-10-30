function compute_za_bp(ψₐ, msg_in)
    isempty(msg_in) && return one(eltype(ψₐ))
    return sum(ψₐ(xₐ) * prod(m[xᵢ] for (m, xᵢ) in zip(msg_in, xₐ)) 
        for xₐ in Iterators.product(eachindex.(msg_in)...))
end

function compute_logza_maxsum(ψₐ, msg_in)
    isempty(msg_in) && return zero(eltype(ψₐ))
    return maximum(log(ψₐ(xₐ)) + sum(m[xᵢ] for (m, xᵢ) in zip(msg_in, xₐ)) 
        for xₐ in Iterators.product(eachindex.(msg_in)...))
end

function UpdateFactorBP(compute_za::Function=compute_za_bp)
        function update_f_bp!(bp::BP{F,FV,M,MB}, a::Integer, unew, damp::Real,
            f::BetheFreeEnergy{<:AtomicVector}; extra_kwargs...) where {
                F<:BPFactor, FV<:BPFactor, M<:AbstractVector{<:Real}, MB<:AbstractVector{<:Real}}
        (; g, ψ, u, h) = bp
        ea = edge_indices(g, factor(a))
        ψₐ = ψ[a]
        zₐ = compute_za(ψₐ, h[ea])
        hflat = @views mortar(h[ea])
        uflat = @views mortar(unew[ea])
        ForwardDiff.gradient!(uflat, hflat -> compute_za(ψₐ, hflat.blocks), hflat)

        f.factors[a] -= log(zₐ)
        err = typemin(eltype(bp))
        for ai in ea
            zₐ₂ᵢ = sum(unew[ai])
            unew[ai] ./= zₐ₂ᵢ
            err = max(err, maximum(abs, unew[ai] - u[ai]))
            u[ai] = damp!(u[ai], unew[ai], damp)
        end
        return err
    end
end

# function UpdateFactorMaxsum(compute_logza::Function=compute_logza_maxsum)
#     function update_f_ms!(bp::BP, a::Integer, unew, damp::Real,
#             f::BetheFreeEnergy{<:AtomicVector}; extra_kwargs...)
#         (; g, ψ, u, h) = bp
#         ea = edge_indices(g, factor(a))
#         ψₐ = ψ[a]
#         logzₐ = compute_logza(ψₐ, h[ea])
#         hflat = @views mortar(h[ea])
#         uflat = @views mortar(unew[ea])
#         ForwardDiff.gradient!(uflat, hflat -> compute_logza(ψₐ, hflat.blocks), hflat)
#         @show unew[ea]
#         f.factors[a] -= logzₐ
#         err = typemin(eltype(bp))
#         for ai in ea
#             # unew[ai] .-= h[ai]
#             logzₐ₂ᵢ = maximum(unew[ai])
#             unew[ai] .-= logzₐ₂ᵢ
#             err = max(err, maximum(abs, unew[ai] - u[ai]))
#             u[ai] = damp!(u[ai], unew[ai], damp)
#         end
#         return err
#     end
# end