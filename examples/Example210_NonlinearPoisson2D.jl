#= 

# 210 : Nonlinear Poisson Problem 2D
([source code](SOURCE_URL))

This example computes the solution ``u`` of the nonlinear Poisson problem
```math
\begin{aligned}
-\mathrm{div}(q(u) \nabla u) & = f \quad \text{in } \Omega
\end{aligned}
```
with some right-hand side ``f`` on a series of uniform refinements of the unit square ``\Omega``.
The quantity q(u) makes the problem nonlinear and we consider the two possibilites
```math
\begin{aligned}
    q_1(u) &:= 1 + u^2\\
    q_2(u) &:= (\kappa + \lvert \nabla u \rvert)^{p-2} 
\end{aligned}
```
where the second one is known is the p-Laplacian (plus some small regularisation $\kappa \geq 0$ to make it solvable with the Newton solver).

This example demonstrates the automatic differentation feature and explains how to setup a nonlinear expression
and how to assign it to the problem description. The setup is tested with some manufactured quadratic solution.

Also the factorization in the linear solver can be changed to anything <:ExtendableSparse.AbstractFactorization
(but not every one will work in this example).

=#

module Example210_NonlinearPoisson2D

using GradientRobustMultiPhysics
using ExtendableSparse
using ExtendableGrids
using GridVisualize

## problem data
function exact_function!(result,x)
    result[1] = x[1]*x[2]
    return nothing
end
function exact_gradient!(result,x)
    result[1] = x[2]
    result[2] = x[1]
    return nothing
end
function rhs!(q,p,κ)
    function closure(result,x)
        if q == 1
            result[1] = -2*(x[1]^3*x[2] + x[2]^3*x[1]) # = -div((1+u^2)*grad(u))
        elseif q == 2
            result[1] = -2*(p-2) * (κ + x[1]^2+x[2]^2)^((p-2)/2-1) * x[1] * x[2] # = -div((κ + |grad(u)|)^p-2*grad(u))
        end
        return nothing
    end
    return closure
end
function diffusion_kernel1!(result, input)
    ## input[1,2:3] = [u, grad(u)]
    result[1] = (1+input[1]^2)*input[2]
    result[2] = (1+input[1]^2)*input[3]
    return nothing
end 
function diffusion_kernel2!(p,κ)
    function closure(result, input)
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
function main(; q = 1, p = 2.7, κ = 0.0001, Plotter = nothing, verbosity = 0, nlevels = 6, FEType = H1P1{1}, testmode = false, factorization = ExtendableSparse.LUFactorization)

    ## set log level
    set_verbosity(verbosity)

    ## choose initial mesh
    xgrid = grid_unitsquare(Triangle2D)

    ## negotiate data functions to the package
    u = DataFunction(exact_function!, [1,2]; name = "u_exact", dependencies = "X", quadorder = 2)
    ∇u = DataFunction(exact_gradient!, [2,2]; name = "grad(u_exact)", dependencies = "X", quadorder = 1) 
    f = DataFunction(rhs!(q,p,κ), [1,2]; dependencies = "X", name = "f", quadorder = 4)

    ## prepare nonlinear expression (1+u^2)*grad(u)
    if q == 1
        nonlin_diffusion = NonlinearForm([Identity, Gradient], [1,1], Gradient, diffusion_kernel1!, [2,3]; name = "(1+u^2) ∇u ⋅ ∇v", quadorder = 4, ADnewton = true) 
    elseif q == 2
        nonlin_diffusion = NonlinearForm([Gradient], [1], Gradient, diffusion_kernel2!(p,κ), [2,2]; name = "(κ+|∇u|^2) ∇u ⋅ ∇v", quadorder = 5, ADnewton = true)   
    else 
        @error "only q ∈ [1,2] !"
    end

    ## generate problem description and assign nonlinear operator and data
    Problem = PDEDescription("nonlinear Poisson problem")
    add_unknown!(Problem; unknown_name = "u", equation_name = "nonlinear Poisson equation")
    add_operator!(Problem, [1,1], nonlin_diffusion)
    add_boundarydata!(Problem, 1, [1,2,3,4], BestapproxDirichletBoundary; data = u)
    add_rhsdata!(Problem, 1,  RhsOperator(Identity, [0], f; store = true))
    @show Problem

    ## prepare error calculation
    L2Error = L2ErrorIntegrator(Float64, u, Identity)
    H1Error = L2ErrorIntegrator(Float64, ∇u, Gradient)
    NDofs = zeros(Int,nlevels)
    Results = zeros(Float64,nlevels,2)

    ## loop over levels
    Solution = nothing
    for level = 1 : nlevels
        ## uniform mesh refinement
        xgrid = uniform_refine(xgrid)
        
        ## create finite element space and solution vector
        FES = FESpace{FEType}(xgrid)
        Solution = FEVector{Float64}("u_h",FES)

        ## solve
        @show Solution
        solve!(Solution, Problem; linsolver = factorization, show_statistics = true)

        ## calculate L2 and H1 error and save data
        NDofs[level] = length(Solution.entries)
        Results[level,1] = sqrt(evaluate(L2Error,Solution[1]))
        Results[level,2] = sqrt(evaluate(H1Error,Solution[1]))
    end

    if testmode == true
        return Results[end,2]
    else
        ## plot
        p = GridVisualizer(; Plotter = Plotter, layout = (1,2), clear = true, resolution = (1000,500))
        scalarplot!(p[1,1], xgrid, view(nodevalues(Solution[1]),1,:), levels = 11, title = "u_h")
        scalarplot!(p[1,2], xgrid, view(nodevalues(Solution[1], Gradient; abs = true),1,:), levels=7)
        vectorplot!(p[1,2], xgrid, evaluate(PointEvaluator(Solution[1], Gradient)), spacing = 0.1, clear = false, title = "∇u_h (abs + quiver)")

        ## print/plot convergence history
        print_convergencehistory(NDofs, Results; X_to_h = X -> X.^(-1/2), ylabels = ["|| u - u_h ||", "|| ∇(u - u_h) ||"])
        plot_convergencehistory(NDofs, Results; add_h_powers = [1,2], X_to_h = X -> X.^(-1/2), Plotter = Plotter, ylabels = ["|| u - u_h ||", "|| ∇(u - u_h) ||"])
    end
end

## test function that is called by test unit
## tests if the above problem is solved exactly by P2-FEM
function test()
    return main(; FEType = H1P2{1,2}, q = 1, nlevels = 1, testmode = true)
end

end
