#= 

# 3D Poisson-Problem
([source code](SOURCE_URL))

This example computes the solution ``u`` of the three dimensional Poisson problem
```math
\begin{aligned}
-\Delta u & = f \quad \text{in } \Omega
\end{aligned}
```
with some right-hand side ``f`` on the unit cube domain ``\Omega`` on a series of uniform refined meshes (tetrahedra or parallelepipeds).

=#

module Example_3DPoisson

using GradientRobustMultiPhysics
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
    L2error = []; H1error = []; NDofs = []

    ## loop over levels
    Solution = nothing
    for level = 1 : nlevels
        ## uniform mesh refinement
        xgrid = uniform_refine(xgrid)
        
        ## create finite element space and solution vector
        FES = FESpace{FEType}(xgrid)
        Solution = FEVector{Float64}("u_h",FES)
        push!(NDofs,length(Solution.entries))

        ## solve the problem
        solve!(Solution, Problem)

        ## calculate L2 and H1 error
        append!(L2error,sqrt(evaluate(L2ErrorEvaluator,Solution[1])))
        append!(H1error,sqrt(evaluate(H1ErrorEvaluator,Solution[1])))
    end

    ## output errors in a nice table
    println("\n   NDOF  |   L2ERROR   |   H1ERROR")
    for j=1:nlevels
        @printf("  %6d |",NDofs[j]);
        @printf(" %.5e |",L2error[j])
        @printf(" %.5e\n",H1error[j])
    end

    ## plot (Plotter = Makie should work)
    GradientRobustMultiPhysics.plot(xgrid, [Solution[1]], [Identity]; Plotter = Plotter)
end

end

#=
### Output of default main() run
=#
Example_3DPoisson.main()