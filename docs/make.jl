using BeliefPropagation
using Documenter

DocMeta.setdocmeta!(BeliefPropagation, :DocTestSetup, 
    :(using BeliefPropagation, IndexedFactorGraphs); recursive=true)

# Copy README to `index.md`
# copied from https://github.com/rafaqz/Interfaces.jl/blob/071d44f6ae9c5a1c0e53b4a06cc44598224fbcc7/docs/make.jl#L8-L25
base_url = "https://github.com/stecrotti/BeliefPropagation.jl/blob/main/"
index_path = joinpath(@__DIR__, "src", "index.md")
readme_path = joinpath(dirname(@__DIR__), "README.md")

# open(index_path, "w") do io
#     for line in eachline(readme_path)
#         println(io, line)
#     end
# end

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
        "Custom models" => "custom_factors.md",
        "API reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/stecrotti/BeliefPropagation.jl",
    devbranch="main",
    push_preview=true,
)
