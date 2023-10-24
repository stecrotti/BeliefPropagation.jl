# BeliefPropagation

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://stecrotti.github.io/BeliefPropagation.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://stecrotti.github.io/BeliefPropagation.jl/dev/)
[![Build Status](https://github.com/stecrotti/BeliefPropagation.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/stecrotti/BeliefPropagation.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/stecrotti/BeliefPropagation.jl/graph/badge.svg?token=KjSnA3UPCt)](https://codecov.io/gh/stecrotti/BeliefPropagation.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

This repository implements the [Belief Propagation](https://en.wikipedia.org/wiki/Belief_propagation) algorithm for the approximation of probability distributions factorized on a graph
```math
\begin{equation}
p(x_1,x_2,\ldots,x_n) \propto \prod_{a\in F} \psi_a(\underline{x}_a) \prod_{i\in V} \phi_i(x_i) 
\end{equation}
```
where $F$ is the set of factors, $V$ the set of variables, and $\underline{x}_a=\\{i\in V | \exists\\; \rm{edge}\\; (i,a)\\}$.

## Installation
```julia
import Pkg; Pkg.add("https://github.com/stecrotti/BeliefPropagation.jl.git")
```

## Quickstart
Check out the [examples](https://github.com/stecrotti/BeliefPropagation.jl/tree/main/examples) folder.
