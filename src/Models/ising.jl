potts2spin(x) = 3-2x

struct IsingCoupling{T<:Real}  <: BPFactor 
    βJ :: T 
end

function (f::IsingCoupling)(x)
    (; βJ) = f
    @assert length(x) == 2
    E = - βJ * prod(potts2spin(xᵢ) for xᵢ in x)
    return 1 / (1 + exp(2E))
end

struct IsingField{T<:Real}  <: VertexBPFactor 
    βh :: T 
end

function (f::IsingField)(x::Integer)
    (; βh) = f
    E = - βh * potts2spin(x)
    return 1 / (1 + exp(2E))
end