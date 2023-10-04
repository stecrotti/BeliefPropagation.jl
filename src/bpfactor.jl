abstract type AbstractBPFactor end
abstract type BPFactor <: AbstractBPFactor end
abstract type VertexBPFactor <: AbstractBPFactor end

struct UniformVertexFactor{T<:Integer} <: VertexBPFactor
    q :: T
end

(f::UniformVertexFactor)(x) = 1 / f.q