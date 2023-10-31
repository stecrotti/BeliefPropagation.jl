module FactorGraphsPlotsExt

using BeliefPropagation.FactorGraphs
using Plots, GraphRecipes

"""
    plot(g::FactorGraph; kwargs...)

Plot factor graph `g` with boxes for factor nodes and circles for variable nodes.
It is based on [`GraphRecipes.graphplot`](https://docs.juliaplots.org/stable/GraphRecipes/introduction/#Usage).

Optional arguments
========
- `shownames`: if set to `true`, displays the index on every node
- optional arguments to `graphplot`

Examples
========

```@example plot
julia> using BeliefPropagation.FactorGraphs

julia> using Plots, GraphRecipes

julia> g = FactorGraph([0 1 1 0;
                        1 0 1 0;
                        0 0 1 1])
FactorGraph{Int64} with 3 factors, 4 variables and 6 edges

julia> plot(g)
```

"""
function Plots.plot(g::FactorGraph; shownames=false,
        nodecolor = fill(:white, nv(g)), curves=false, 
        nodesize = shownames ? 0.15 : 0.1, plotargs...)
    nodeshape = vcat(fill(:rect, nfactors(g)), fill(:circle, nvariables(g)))
    names = shownames ? vcat(1:nfactors(g), 1:nvariables(g)) : Int[]
    GraphRecipes.graphplot(g.g; nodeshape, curves, nodecolor, names, nodesize, plotargs...)
end

end # module