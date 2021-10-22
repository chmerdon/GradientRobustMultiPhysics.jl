#= 

# 201 : Poisson-Problem 2D
([source code](SOURCE_URL))

This example computes the solution ``u`` of the Poisson problem
```math
\begin{aligned}
-\Delta u & = f \quad \text{in } \Omega
\end{aligned}
```
with some right-hand side ``f`` on the unit cube domain ``\Omega`` on a given grid.

=#

module Example201_PoissonProblem2D

using GradientRobustMultiPhysics
using ExtendableGrids

## right-hand side function
const f = DataFunction([1]; name = "f")

## everything is wrapped in a main function
function main(; verbosity = 0, μ = 1, Plotter = nothing)

    ## set log level
    set_verbosity(verbosity)

    ## build/load any grid (here: a uniform-refined 2D unit square into triangles)
    xgrid = uniform_refine(grid_unitsquare(Triangle2D),5)

    ## create empty PDE description
    Problem = PDEDescription("Poisson problem")

    ## add unknown(s) (here: "u" that gets id 1 for later reference)
    add_unknown!(Problem; unknown_name = "u", equation_name = "Poisson equation")

    ## add left-hand side PDEoperator(s) (here: only Laplacian)
    add_operator!(Problem, [1,1], LaplaceOperator(μ))

    ## add right-hand side data (here: f = [1] in region(s) [1])
    add_rhsdata!(Problem, 1, RhsOperator(Identity, [1], f))

    ## add boundary data (here: zero data for boundary regions 1:4)
    add_boundarydata!(Problem, 1, [1,2,3,4], HomogeneousDirichletBoundary)

    ## discretise = choose FEVector with appropriate FESpaces
    FEType = H1P2{1,2}
    Solution = FEVector("u_h",FESpace{FEType}(xgrid))

    ## show problem and Solution structure
    @show Problem Solution

    ## solve for chosen Solution vector
    solve!(Solution, Problem)

    ## plot solution (for e.g. Plotter = PyPlot)
    GradientRobustMultiPhysics.plot(xgrid, [Solution[1], Solution[1]], [Identity, Gradient]; Plotter = Plotter)    
end

end