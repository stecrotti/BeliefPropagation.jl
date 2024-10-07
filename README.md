# BeliefPropagation

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://stecrotti.github.io/BeliefPropagation.jl/dev/)
[![Build Status](https://github.com/stecrotti/BeliefPropagation.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/stecrotti/BeliefPropagation.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/stecrotti/BeliefPropagation.jl/graph/badge.svg?token=KjSnA3UPCt)](https://codecov.io/gh/stecrotti/BeliefPropagation.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

-------------------

⚠️ This package is heavily work in progress, some breaking changes should be expected.

-------------------

This package implements a generic version of the [Belief Propagation](https://en.wikipedia.org/wiki/Belief_propagation) (BP) algorithm for the approximation of probability distributions factorized on a graph
```math
\begin{equation}
p(x_1,x_2,\ldots,x_n) \propto \prod_{a\in F} \psi_a(\underline{x}_a) \prod_{i\in V} \phi_i(x_i) 
\end{equation}
```
where $F$ is the set of factors, $V$ the set of variables, and $\underline{x}_a$ is the set of variables involved in factor $a$.

## Installation
```julia
import Pkg; Pkg.add("https://github.com/stecrotti/BeliefPropagation.jl.git")
```

## Quickstart
Check out the [examples](https://github.com/stecrotti/BeliefPropagation.jl/tree/main/examples) folder.

## Overview
The goal of this package is to provide a simple, flexible, and ready-to-use interface to the BP algorithm. It is enough for the user to provide the factor graph (encoded in an adjacency matrix or as a [Graphs.jl](https://github.com/JuliaGraphs/Graphs.jl) graph) and the factors, everything else is taken care of.

At the same time, the idea is that refinements can be made to improve performance on a case-by-case basis. For example, messages are stored as `Vector`s by default, but when working with binary variables, one real number is enough, allowing for considerable speed-ups (see the [Ising](https://github.com/stecrotti/BeliefPropagation.jl/blob/9cbc01d6bbd0266531d6047482b8617bb6eb71ab/src/Models/ising.jl#L56) example).
Also, a version of BP for continuous variables such as Gaussian BP can be introduced in the framework, although it is not yet implemented.

## See also
- [BeliefPropagation.jl](https://github.com/ArtLabBocconi/BeliefPropagation.jl): implements BP for the Ising model and the matching problem.
- [FactorGraph.jl](https://github.com/mcosovic/FactorGraph.jl): implements Gaussian BP and other message-passing algorithms.
- [ITensorNetworks.jl](https://github.com/ITensor/ITensorNetworks.jl): implements BP as a technique for approximate tensor network contraction.
