using Documenter
using ExtendableSparse
using ExtendableGrids

push!(LOAD_PATH, "../src")
using GradientRobustMultiPhysics

# utility function from https://github.com/JuliaOpt/Convex.jl/blob/master/docs/make.jl
fix_math_md(content) = replace(content, r"\$\$(.*?)\$\$"s => s"```math\1```")

makedocs(
    modules=[GradientRobustMultiPhysics],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    sitename="GradientRobustMultiPhysics.jl",
    authors="Christian Merdon",
    pages = [
        "Home" => "index.md",
        "Implemented Finite Elements" => "fems.md",
        "Finite Element Spaces and Arrays" => "fespace.md",
        "PDE Description" => "pdedescription.md",
        "PDE Prototypes" => "pdeprototypes.md",
        "PDE Solvers" => "pdesolvers.md",
        "AbstractActions" => "abstractactions.md",
        "AbstractAssemblyPatterns" => "assemblypatterns.md",
        "Quadrature" => "quadrature.md",
    ]
)

#deploydocs(
#    repo = "github.com/chmerdon/juliaFE",
#)