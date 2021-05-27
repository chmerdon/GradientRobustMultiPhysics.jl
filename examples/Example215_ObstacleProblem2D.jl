#= 

# 215 : Obstacle Problem 2D
([source code](SOURCE_URL))

This example computes the solution ``u`` of the nonlinear obstacle problem that seeks the minimiser of the energy functional
```math
\begin{aligned}
    E(u) = \int_\Omega \lvert \nabla u \rvert^2 dx - \int_\Omega f u dx
\end{aligned}
```
with some right-hand side ``f`` within the set of admissible functions that lie above an obstacle  ``\chi``
```math
\begin{aligned}
    \mathcal{K} := \lbrace u \in H^1_0(\Omega) : u \geq \chi \rbrace.
\end{aligned}
```

The obstacle constraint is realised via a penalty terms that is automatically differentiated for a Newton scheme.

=#

module Example215_ObstacleProblem2D

using GradientRobustMultiPhysics
using ExtendableSparse
using Printf

## problem data
function obstacle!(result,x)
    result[1] = -0.05 + 0.05*cos(4*x[1]*pi)*cos(4*x[2]*pi)
    return nothing
end
function obstacle_penalty_kernel!(result::Array{<:Real,1}, input::Array{<:Real,1},x)
    obstacle!(result,x)
    result[1] = min(0, input[1] - result[1])
    return nothing
end

function main(; Plotter = nothing, verbosity = 0, penalty = 1e4, nrefinements = 5, FEType = H1P1{1})

    ## set log level
    set_verbosity(verbosity)

    ## choose initial mesh
    xgrid = uniform_refine(grid_unitsquare(Triangle2D),nrefinements)

    ## generate problem description
    Problem = PDEDescription("obstacle problem")
    add_unknown!(Problem; unknown_name = "u", equation_name = "obstacle problem")
    add_operator!(Problem, [1,1], LaplaceOperator(1.0; store = true))
    add_operator!(Problem, [1,1], GenerateNonlinearForm("eps^{-1} ||(u-χ)_||", [Identity], [1], Identity, obstacle_penalty_kernel!, [1,1]; dependencies = "X", factor = penalty, quadorder = 4, ADnewton = true) )
    add_boundarydata!(Problem, 1, [1,2,3,4], HomogeneousDirichletBoundary)
    add_rhsdata!(Problem, 1,  RhsOperator(Identity, [0], DataFunction([-1]); store = true))
        
    ## create finite element space and solution vector
    FES = FESpace{FEType}(xgrid)
    Solution = FEVector{Float64}("u_h",FES)

    ## solve
    @show Problem Solution
    solve!(Solution, Problem; maxiterations = 20)

    ## plot
    GradientRobustMultiPhysics.plot(xgrid, [Solution[1], Solution[1]], [Identity, Gradient]; Plotter = Plotter)
end


end