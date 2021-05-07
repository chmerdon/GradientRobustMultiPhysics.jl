#= 

# 211 : p-Laplacian 2D
([source code](SOURCE_URL))

This example computes the solution ``u`` of the nonlinear p-Laplace problem
```math
\begin{aligned}
-\mathrm{div}((\kappa + \lvert \nabla u \rvert)^{p-2} \nabla u) & = f \quad \text{in } \Omega
\end{aligned}
```
with some ``p \in (1,\infty)``, ``\kappa \geq 0`` and some right-hand side ``f`` on a series of uniform refinements of the unit square ``\Omega``.

This example demonstrates the automatic differentation feature and explains how to setup a nonlinear expression
and how to assign it to the problem description. The setup is tested with some manufactured quadratic solution.

Also the factorization in the linear solver can be changed to anything <:ExtendableSparse.AbstractFactorization
(but not every one will work in this example).

=#

module Example211_pLaplacian2D

using GradientRobustMultiPhysics
using ExtendableSparse
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
function rhs!(p,κ)
    function closure(result,x::Array{<:Real,1})
        result[1] = -2*(p-2) * (κ + x[1]^2+x[2]^2)^((p-2)/2-1) * x[1] * x[2] # = -div((\kappa + |grad(u)|)^p-2*grad(u))
    end
    return closure
end
function diffusion_kernel!(p,κ)
    function closure(result::Array{<:Real,1}, input::Array{<:Real,1})
        ## input[1:2] = [grad(u)]
        ## we use result[1] as temporary storage to compute (κ + |∇u|)^(p-2)
        result[1] = (κ + input[1]^2 + input[2]^2)^((p-2)/2)
        result[2] = result[1] * input[2]
        result[1] = result[1] * input[1]
        return nothing
    end
    return closure
end 

## everything is wrapped in a main function
## default argument trigger P1-FEM calculation, you might also want to try H1P2{1,2}
function main(; Plotter = nothing, p = 2.7, κ = 0.0001, verbosity = 0, nlevels = 6, FEType = H1P1{1}, factorization = ExtendableSparse.LUFactorization)

    ## set log level
    set_verbosity(verbosity)

    ## choose initial mesh
    xgrid = grid_unitsquare(Triangle2D)

    ## negotiate data functions to the package
    user_function = DataFunction(exact_function!, [1,2]; name = "u_exact", dependencies = "X", quadorder = 2)
    user_function_gradient = DataFunction(exact_gradient!, [2,2]; name = "grad(u_exact)", dependencies = "X", quadorder = 1)
    user_function_rhs = DataFunction(rhs!(p,κ), [1,2]; dependencies = "X", name = "f", quadorder = 5)

    ## prepare nonlinear expression (1+u^2)*grad(u)
    nonlin_diffusion = GenerateNonlinearForm("(κ+|∇u|^2) ∇u ⋅ ∇v", [Gradient], [1], Gradient, diffusion_kernel!(p,κ), [2,2]; quadorder = 5, ADnewton = true)   

    ## generate problem description and assign nonlinear operator and data
    Problem = PDEDescription("p-Laplace problem (p = $p, κ = $κ)")
    add_unknown!(Problem; unknown_name = "u", equation_name = "p-Laplace equation")
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
        Solution = FEVector{Float64}("u_h",FES)
        push!(NDofs,length(Solution.entries))

        ## solve
        solve!(Solution, Problem; linsolver = factorization, maxiterations = 10)

        ## calculate L2 and H1 error
        append!(L2error,sqrt(evaluate(L2ErrorEvaluator,Solution[1])))
        append!(H1error,sqrt(evaluate(H1ErrorEvaluator,Solution[1])))
    end

    ## output errors in a nice table
    println("\n   NDOF  |   L2ERROR   |   H1ERROR")
    for j=1:nlevels
        @printf("  %6d | %.5e | %.5e\n",NDofs[j],L2error[j],H1error[j]);
    end

    ## plot
    GradientRobustMultiPhysics.plot(xgrid, [Solution[1], Solution[1]], [Identity, Gradient]; Plotter = Plotter)
end

end
