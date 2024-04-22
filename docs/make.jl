using BeliefPropagation
using Documenter

DocMeta.setdocmeta!(BeliefPropagation, :DocTestSetup, 
    :(using BeliefPropagation, IndexedFactorGraphs); recursive=true)

makedocs(;
    modules=[
        BeliefPropagation,
    ],
    authors="Stefano Crotti, Alfredo Braunstein, and contributors",
    repo="https://github.com/stecrotti/BeliefPropagation.jl/blob/{commit}{path}#{line}",
    sitename="BeliefPropagation.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://stecrotti.github.io/BeliefPropagation.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/stecrotti/BeliefPropagation.jl",
    devbranch="main",
    push_preview=true,
)
