using Documenter
using ExtendableSparse
using ExtendableGrids

push!(LOAD_PATH, "../src")
using JUFELIA

makedocs(
    modules=[JUFELIA],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    sitename="JUFELIA",
    authors="Christian Merdon",
    pages = [
        "Home" => "index.md",
        "Finite Element Spaces and Arrays" => "fespace.md",
        "Implemented Finite Elements" => "fems.md",
        "PDEDescription" => "pdedescription.md",
        "PDE Solvers" => "pdesolvers.md",
        "Abstract Actions" => "abstractactions.md",
        "Assembly Patterns" => "assemblypatterns.md",
        "Quadrature" => "quadrature.md",
    ]
)

deploydocs(
    repo = "github.com/chmerdon/juliaFE",
)