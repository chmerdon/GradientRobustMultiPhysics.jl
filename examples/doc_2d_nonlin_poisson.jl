#= 

# 2D Nonlinear Poisson-Problem
([source code](SOURCE_URL))

This example computes the solution ``u`` of the nonlinear Poisson problem
```math
\begin{aligned}
-\mathrm{div}((1+u^2) \nabla u) & = f \quad \text{in } \Omega
\end{aligned}
```
with some right-hand side ``f`` on a series of uniform refinements of the unit square ``\Omega``.

This example demonstrates the automatic differentation feature and explains how to setup a nonlinear expression
and how to assign it to the problem description. The setup is tested with some manufactured quadratic solution.

=#

module Example_2DNonlinearPoisson

using GradientRobustMultiPhysics
using Printf

## problem data
function exact_function!(result,x::Array{<:Real,1})
    result[1] = x[1]*x[2]
    return nothing
end
function exact_gradient!(result,x::Array{<:Real,1})
    result[1] = x[2]
    result[2] = x[1]
    return nothing
end
function rhs!(result,x::Array{<:Real,1})
    result[1] = -2*(x[1]^3*x[2] + x[2]^3*x[1]) # = -div((1+u^2)*grad(u))
    return nothing
end

## everything is wrapped in a main function
## default argument trigger P1-FEM calculation, you might also want to try H1P2{1,2}
function main(; Plotter = nothing, verbosity = 0, nlevels = 6, FEType = H1P1{1}, testmode = false)

    ## set log level
    set_verbosity(verbosity)

    ## choose initial mesh
    xgrid = grid_unitsquare(Triangle2D)

    ## negotiate data functions to the package
    user_function = DataFunction(exact_function!, [1,2]; name = "u_exact", dependencies = "X", quadorder = 2)
    user_function_gradient = DataFunction(exact_gradient!, [2,2]; name = "grad(u_exact)", dependencies = "X", quadorder = 1)
    user_function_rhs = DataFunction(rhs!, [1,2]; dependencies = "X", name = "f", quadorder = 4)

    ## prepare nonlinear expression (1+u^2)*grad(u)
    function nonlin_kernel(result::Array{<:Real,1}, input::Array{<:Real,1})
        ## input = [u, grad(u)]
        result[1] = (1+input[1]^2)*input[2]
        result[2] = (1+input[1]^2)*input[3]
        return nothing
    end 
    action_kernel = ActionKernel(nonlin_kernel, [2,3]; dependencies = "", quadorder = 2)
    nonlin_diffusion = GenerateNonlinearForm("((1+u^2)*grad(u))*grad(v)", [Identity, Gradient], [1,1], Gradient, action_kernel; ADnewton = true)   

    ## generate problem description and assign nonlinear operator and data
    Problem = PDEDescription("nonlinear Poisson problem")
    add_unknown!(Problem; unknown_name = "unknown", equation_name = "nonlinear Poisson equation")
    add_operator!(Problem, [1,1], nonlin_diffusion)
    add_boundarydata!(Problem, 1, [1,2,3,4], BestapproxDirichletBoundary; data = user_function)
    add_rhsdata!(Problem, 1,  RhsOperator(Identity, [0], user_function_rhs; store = true))
    @show Problem

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
        Solution = FEVector{Float64}("Solution",FES)
        push!(NDofs,length(Solution.entries))

        ## solve
        solve!(Solution, Problem)

        ## calculate L2 and H1 error
        append!(L2error,sqrt(evaluate(L2ErrorEvaluator,Solution[1])))
        append!(H1error,sqrt(evaluate(H1ErrorEvaluator,Solution[1])))
    end

    if testmode == true
        return H1error[end]
    else
        ## output errors in a nice table
        println("\n   NDOF  |   L2ERROR   |   H1ERROR")
        for j=1:nlevels
            @printf("  %6d | %.5e | %.5e\n",NDofs[j],L2error[j],H1error[j]);
        end
    
        ## plot
        GradientRobustMultiPhysics.plot(xgrid, [Solution[1], Solution[1]], [Identity, Gradient]; Plotter = Plotter)
    end
end


## test function that is called by test unit
## tests if the above problem is solved exactly by P2-FEM
function test()
    return main(; FEType = H1P2{1,2}, nlevels = 1, testmode = true)
end

end

#=
### Output of default main() run
=#
Example_2DNonlinearPoisson.main()