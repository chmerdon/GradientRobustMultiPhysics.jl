#= 

# 205 : Nonlinear Poisson Problem 2D
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
(either via a function (q = 1) or a callable struct (q = 2))
and how to assign it to the problem description. The setup is tested with some manufactured quadratic solution.

Also the factorization in the linear solver can be changed to anything <:ExtendableSparse.AbstractFactorization
(but not every one will work in this example).

=#

module Example205_NonlinearPoisson2D

using GradientRobustMultiPhysics
using ExtendableSparse
using ExtendableGrids
using GridVisualize

## problem data
function exact_function!(result,x)
    result[1] = x[1]*x[2]
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

## for kernels withouts parameters, closures should work fine
function diffusion_kernel1!(result, input)
    ## input[1,2:3] = [u, grad(u)]
    result[1] = (1+input[1]^2)*input[2]
    result[2] = (1+input[1]^2)*input[3]
    return nothing
end 
function jac_diffusion_kernel1!(jacobian, input)
    jacobian[1,1] = 2*input[1]*input[2]
    jacobian[1,2] = (1+input[1]^2)
    jacobian[2,1] = 2*input[1]*input[3]
    jacobian[2,3] = (1+input[1]^2)
    return nothing
end 

## alternatively callable structs are possible
## (and currently suggested if kernel should depend on parameters)
mutable struct diffusion_kernel2{T}
    p::T
    κ::T
end

(DK::diffusion_kernel2)(result,input) = (
    result[1] = (DK.κ + input[1]^2 + input[2]^2)^((DK.p-2)/2);
    result[2] = result[1] * input[2];
    result[1] = result[1] * input[1];
    return nothing
)

## needs to be initialized as const currently to avoid some internal allocations
## (but parameters can be changed in main)
const DK = diffusion_kernel2(0.0,1.0)

## everything is wrapped in a main function
## default argument trigger P1-FEM calculation, you might also want to try H1P2{1,2}
function main(; 
    q::Int = 1,           # which nonlinear operator should be used?
    p::Float64 = 2.7,     # coefficient for diffusion kernel 2
    κ::Float64 = 0.0001,  # coefficient for diffusion kernel 2
    nlevels = 6,          # number of levels in refinement loop  
    FEType = H1P1{1},     # FEType to be used (H1P2{1,2} should give exact solution)
    autodiff = false,     # only for q = 1: use jacobians from automatic differentiation or the ones provided above? (q = 2 always uses autodiff)
    Plotter = nothing, 
    verbosity = 0,
    testmode = false,
    factorization = ExtendableSparse.LUFactorization)

    ## set log level
    set_verbosity(verbosity)

    ## choose initial mesh
    xgrid = grid_unitsquare(Triangle2D)

    ## negotiate data functions to the package
    u = DataFunction(exact_function!, [1,2]; name = "u_exact", dependencies = "X", quadorder = 2)
    ∇u = ∇(u)
    f = DataFunction(rhs!(q,p,κ), [1,2]; dependencies = "X", name = "f", quadorder = 4)

    ## prepare nonlinear expression (1+u^2)*grad(u)
    if q == 1
        nonlin_diffusion = NonlinearForm(Gradient, [Identity, Gradient], [1,1], diffusion_kernel1!, [2,3]; name = "(1+u^2) ∇u ⋅ ∇v", bonus_quadorder = 2, jacobian = autodiff ? "auto" : jac_diffusion_kernel1!, sparse_jacobian = true) 
    elseif q == 2
        DK.κ = κ
        DK.p = p
        nonlin_diffusion = NonlinearForm(Gradient, [Gradient], [1], DK, [2,2]; name = "(κ+|∇u|^2) ∇u ⋅ ∇v", bonus_quadorder = 4, jacobian = "auto", sparse_jacobian = true)   
    else 
        @error "only q ∈ [1,2] !"
    end

    ## generate problem description and assign nonlinear operator and data
    Problem = PDEDescription("nonlinear Poisson problem")
    add_unknown!(Problem; unknown_name = "u", equation_name = "nonlinear Poisson equation")
    add_operator!(Problem, [1,1], nonlin_diffusion)
    add_boundarydata!(Problem, 1, [1,2,3,4], BestapproxDirichletBoundary; data = u)
    add_rhsdata!(Problem, 1, LinearForm(Identity, f; store = true))
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
        GradientRobustMultiPhysics.solve!(Solution, Problem; linsolver = factorization, maxiterations = 10, show_statistics = true)

        ## calculate L2 and H1 error and save data
        NDofs[level] = length(Solution.entries)
        Results[level,1] = sqrt(evaluate(L2Error,Solution[1]))
        Results[level,2] = sqrt(evaluate(H1Error,Solution[1]))
    end

    if testmode == true
        return Results[end,2]
    else
        ## plot
        p = GridVisualizer(; Plotter = Plotter, layout = (1,3), clear = true, resolution = (1500,500))
        scalarplot!(p[1,1], xgrid, nodevalues_view(Solution[1])[1], levels = 7, title = "u_h")
        scalarplot!(p[1,2], xgrid, view(nodevalues(Solution[1], Gradient; abs = true),1,:), levels = 7, title = "∇u_h (abs + quiver)")
        vectorplot!(p[1,2], xgrid, evaluate(PointEvaluator(Solution[1], Gradient)), spacing = 0.1, clear = false)
        convergencehistory!(p[1,3], NDofs, Results; add_h_powers = [1,2], X_to_h = X -> X.^(-1/2), ylabels = ["|| u - u_h ||", "|| ∇(u - u_h) ||"])

        ## print/plot convergence history
        print_convergencehistory(NDofs, Results; X_to_h = X -> X.^(-1/2), ylabels = ["|| u - u_h ||", "|| ∇(u - u_h) ||"])
    end
end

## test function that is called by test unit
## tests if the above problem is solved exactly by P2-FEM
function test()
    return main(; FEType = H1P2{1,2}, q = 1, nlevels = 1, testmode = true, autodiff = true) + main(; FEType = H1P2{1,2}, q = 1, nlevels = 1, testmode = true, autodiff = false)
end

end
