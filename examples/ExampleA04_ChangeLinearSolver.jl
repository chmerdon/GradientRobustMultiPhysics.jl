#= 

# A04 : Change Linear Solvers
([source code](SOURCE_URL))

This example revisits the nonlinear Poisson example from the introductory examples
and shows how to change the linear solver (in each Newton iteration)
via the parameter linsolver in the solve function. In principle any
Linearsolve.AbstractFactorization subtype can be used. One can use, e.g.

- UMFPACKFactorization (default, via LinearSolve.jl)
- KLUFactorization (via LinearSolve.jl)
- KrylovJL_BICGSTAB (via LinearSolve.jl)
- MKLPardisoFactorize (MKLPardiso via LinearSolvePardiso.jl)

For more choices see LinearSolve.jl documentation.

=#

module ExampleA04_ChangeLinearSolver

using GradientRobustMultiPhysics
using LinearSolve
using ExtendableGrids
using ExtendableSparse
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
function rhs!(result,x)
    result[1] = -2*(x[1]^3*x[2] + x[2]^3*x[1]) # = -div((1+u^2)*grad(u))
    return nothing
end

## everything is wrapped in a main function
function main(; Plotter = nothing, verbosity = 0, nrefinements = 6, FEType = H1P1{1}, linsolver = KLUFactorization)

    ## set log level
    set_verbosity(verbosity)

    ## choose initial mesh
    xgrid = uniform_refine(grid_unitsquare(Triangle2D),nrefinements)

    ## negotiate data functions to the package
    u = DataFunction(exact_function!, [1,2]; name = "u_exact", dependencies = "X", bonus_quadorder = 2)
    u_gradient = DataFunction(exact_gradient!, [2,2]; name = "grad(u_exact)", dependencies = "X", bonus_quadorder = 1)
    u_rhs = DataFunction(rhs!, [1,2]; dependencies = "X", name = "f", bonus_quadorder = 4)

    ## prepare nonlinear expression (1+u^2)*grad(u)
    function diffusion_kernel!(result, input)
        ## input = [u, grad(u)]
        result[1] = (1+input[1]^2)*input[2]
        result[2] = (1+input[1]^2)*input[3]
        return nothing
    end 
    nonlin_diffusion = NonlinearForm(Gradient, [Identity, Gradient], [1,1], diffusion_kernel!, [2,3]; name = "((1+u^2)*grad(u))*grad(v)", bonus_quadorder = 2, newton = true)   

    ## generate problem description and assign nonlinear operator and data
    Problem = PDEDescription("nonlinear Poisson problem")
    add_unknown!(Problem; unknown_name = "u", equation_name = "nonlinear Poisson equation")
    add_operator!(Problem, [1,1], nonlin_diffusion)
    add_boundarydata!(Problem, 1, [1,2,3,4], BestapproxDirichletBoundary; data = u)
    add_rhsdata!(Problem, 1, LinearForm(Identity, u_rhs; store = true))

    ## create finite element space and solution vector
    FES = FESpace{FEType}(xgrid)
    Solution = FEVector(FES)

    ## solve the problem (here the newly defined linear solver type is used)
    solve!(Solution, Problem; linsolver = linsolver, show_solver_config = true, show_statistics = true)

    ## calculate error
    L2Error = L2ErrorIntegrator(u, Identity)
    H1Error = L2ErrorIntegrator(u_gradient, Gradient)
    println("\tL2error = $(sqrt(evaluate(L2Error,Solution[1])))")
    println("\tH1error = $(sqrt(evaluate(H1Error,Solution[1])))")

    ## plot
    p = GridVisualizer(; Plotter = Plotter, layout = (1,2), clear = true, resolution = (1000,500))
    scalarplot!(p[1,1], xgrid, view(nodevalues(Solution[1]),1,:), levels = 11, title = "u_h")
    scalarplot!(p[1,2], xgrid, view(nodevalues(Solution[1], Gradient; abs = true),1,:), levels=7)
    vectorplot!(p[1,2], xgrid, evaluate(PointEvaluator(Solution[1], Gradient)), spacing = 0.1, clear = false, title = "∇u_h (abs + quiver)")
end


end
