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
using Printf

## problem data
function exact_function!(result,x::Array{<:Real,1})
    result[1] = x[1]*(x[3] - x[2]) + x[2]*x[2]
    return nothing
end
function exact_gradient!(result,x::Array{<:Real,1})
    result[1] = x[3] - x[2]
    result[2] = - x[1] + 2*x[2]
    result[3] = x[1]
    return nothing
end

## everything is wrapped in a main function
function main(; Plotter = nothing, verbosity = 0, nlevels = 4)

    ## set log level
    set_verbosity(verbosity)

    ## choose initial mesh
    ## (replace Parallelepiped3D by Tetrahedron3D to change the cell geometries)
    xgrid = grid_unitcube(Tetrahedron3D)

    ## set finite element type used for discretisation
    FEType = H1P1{1}

    ## negotiate data functions to the package
    user_function = DataFunction(exact_function!, [1,3]; name = "u", dependencies = "X", quadorder = 2)
    user_function_gradient = DataFunction(exact_gradient!, [3,3]; name = "∇(u)", dependencies = "X", quadorder = 1)
    user_function_rhs = DataFunction([-2]; name = "f")

    ## create Poisson problem via prototype and add data
    Problem = PoissonProblem(1.0)
    add_boundarydata!(Problem, 1, [1,2,3,4,5,6], BestapproxDirichletBoundary; data = user_function)
    add_rhsdata!(Problem, 1,  RhsOperator(Identity, [0], user_function_rhs))

    ## prepare error calculation
    L2ErrorEvaluator = L2ErrorIntegrator(Float64, user_function, Identity)
    H1ErrorEvaluator = L2ErrorIntegrator(Float64, user_function_gradient, Gradient)
    Results = zeros(Float64, nlevels, 2); NDofs = zeros(Int, nlevels)

    ## loop over levels
    Solution = nothing
    for level = 1 : nlevels
        ## uniform mesh refinement
        xgrid = uniform_refine(xgrid)
        
        ## create finite element space and solution vector
        FES = FESpace{FEType}(xgrid)
        Solution = FEVector{Float64}("u_h",FES)

        ## solve the problem
        solve!(Solution, Problem)

        ## calculate L2 and H1 errors and save data
        NDofs[level] = length(Solution.entries)
        Results[level,1] = sqrt(evaluate(L2ErrorEvaluator,Solution[1]))
        Results[level,2] = sqrt(evaluate(H1ErrorEvaluator,Solution[1]))
    end

    ## plot (Plotter = GLMakie should work)
    GradientRobustMultiPhysics.plot(xgrid, [Solution[1]], [Identity]; Plotter = Plotter)

    ## print/plot convergence history
    print_convergencehistory(NDofs, Results; X_to_h = X -> X.^(-1/3), ylabels = ["|| u - u_h ||", "|| ∇(u - u_h) ||"])
    plot_convergencehistory(NDofs, Results; add_h_powers = [1,2], X_to_h = X -> X.^(-1/3), Plotter = Plotter, ylabels = ["|| u - u_h ||", "|| ∇(u - u_h) ||"])
end

end
