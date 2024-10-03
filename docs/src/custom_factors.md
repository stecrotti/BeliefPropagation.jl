This package is made with the goal of allowing users to define their own `BPFactor`s and the corresponding specialized updates.
In particular, the bottleneck of BP compuations typically is the update of factor-to-variable messages, whose cost grows exponentially with the degree of the factor node.

## Specialized factor updates

For many models one can devise more efficient implementations.
To integrate these with the BeliefPropagation.jl API, users can define a new `BPFactor` and override existing methods by dispatching on the factor type.
The minimum required API for a custom `MyFactor <: BPFactor` is:
- `(f::MyFactor)(x)`: a functor evaluating the factor for a given input `x`
- `BeliefPropagation.compute_za(ψₐ::MyFactor, msg_in)` which computes the factor normalization
```math
\begin{equation*}
z_a = \sum_{\underline{x}_a} \psi_a(\underline{x}_a) \prod_{i\in\partial a} h_{i\to a}(x_i)
\end{equation*}
``` 
where $h_{i\to a}$ are the incoming variable-to-factor messages.

And... that's it! From the computation of $z_a$, the one for outgoing messages is obtained under the hood using automatic differentiation.

For example, here is the efficient implementation for factors representing [K-SAT formulas](https://en.wikipedia.org/wiki/Boolean_satisfiability_problem):
```
struct KSATClause{T}  <: BPFactor where {T<:AbstractVector{<:Bool}}
    J :: T 
end

function (f::KSATClause)(x) 
    isempty(x) && return 1.0
    return any(xᵢ - 1 != Jₐᵢ for (xᵢ, Jₐᵢ) in zip(x, f.J)) |> float
end

function BeliefPropagation.compute_za(ψₐ::KSATClause, 
        msg_in::AbstractVector{<:AbstractVector{<:Real}})
    isempty(msg_in) && return one(eltype(ψₐ))
    z1 = prod(sum(hᵢₐ) for (hᵢₐ, Jₐᵢ) in zip(msg_in, ψₐ.J))
    z2 = prod(hᵢₐ[Jₐᵢ+1] for (hᵢₐ, Jₐᵢ) in zip(msg_in, ψₐ.J))
    return z1 - z2
end
```

## Further optimizations
BeliefPropagation.jl by default represents messages as `Vector`s of floats. However, for problems with binary variables, messages are binary distributions and only need one parameter to be specified.
In these case, one can override the functions that perform the update for a simplified type of messages, as well as a specific type of factor.
As an example, see the efficient implementation of BP for the Ising model provided [here](https://github.com/stecrotti/BeliefPropagation.jl/blob/main/src/Models/ising.jl).