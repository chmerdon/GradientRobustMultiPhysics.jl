#= 

# 301 : Poisson-Problem 3D
([source code](SOURCE_URL))

This example computes the solution ``u`` of the three dimensional Poisson problem
```math
\begin{aligned}
-\Delta u & = f \quad \text{in } \Omega
\end{aligned}
```
with some right-hand side ``f`` on the unit cube domain ``\Omega`` on a series of uniform refined meshes (tetrahedra or parallelepipeds).

=#

module Example301_Poisson3D

using GradientRobustMultiPhysics
using ExtendableGrids
using GridVisualize

## problem data
function exact_function!(result,x)
    result[1] = x[1]*(x[3] - x[2]) + x[2]*x[2]
    return nothing
end

## negotiate data functions to the package
const u = DataFunction(exact_function!, [1,3]; name = "u", dependencies = "X", bonus_quadorder = 2)
const f = DataFunction([-2]; name = "f") # = -Δu = -2

## everything is wrapped in a main function
function main(; Plotter = nothing, verbosity = 0, nlevels = 4)

    ## set log level
    set_verbosity(verbosity)

    ## choose initial mesh
    ## (replace Parallelepiped3D by Tetrahedron3D to change the cell geometries)
    xgrid = grid_unitcube(Tetrahedron3D)

    ## set finite element type used for discretisation
    FEType = H1P1{1}

    ## create Poisson problem via prototype and add data
    Problem = PoissonProblem(1.0)
    add_boundarydata!(Problem, 1, [1,2,3,4,5,6], BestapproxDirichletBoundary; data = u)
    add_rhsdata!(Problem, 1, LinearForm(Identity, f))

    ## prepare error calculation
    L2Error = L2ErrorIntegrator(u)
    H1Error = L2ErrorIntegrator(∇(u), Gradient)
    Results = zeros(Float64, nlevels, 2); NDofs = zeros(Int, nlevels)

    ## loop over levels
    Solution = nothing
    for level = 1 : nlevels
        ## uniform mesh refinement
        xgrid = uniform_refine(xgrid)
        
        ## create finite element space and solution vector
        FES = FESpace{FEType}(xgrid)
        Solution = FEVector("u_h",FES)

        ## solve the problem
        solve!(Solution, Problem)

        ## calculate L2 and H1 errors and save data
        NDofs[level] = length(Solution.entries)
        Results[level,1] = sqrt(evaluate(L2Error,Solution[1]))
        Results[level,2] = sqrt(evaluate(H1Error,Solution[1]))
    end

    ## plot (Plotter = GLMakie should work)
    p = GridVisualizer(; Plotter = Plotter, layout = (1,2), clear = true, resolution = (1000,500))
    scalarplot!(p[1,1], xgrid, view(nodevalues(Solution[1]),1,:), levels = 5, title = "u_h")
    convergencehistory!(p[1,2], NDofs, Results; add_h_powers = [1,2], X_to_h = X -> X.^(-1/3),  ylabels = ["|| u - u_h ||", "|| ∇(u - u_h) ||"])

    ## print/plot convergence history
    print_convergencehistory(NDofs, Results; X_to_h = X -> X.^(-1/3), ylabels = ["|| u - u_h ||", "|| ∇(u - u_h) ||"])
end

end