#= 

# A05 : Comparison with DifferentialEquations.jl
([source code](SOURCE_URL))

This example computes a transient velocity ``\mathbf{u}`` solution of the nonlinear Poisson problem
```math
\begin{aligned}
\mathbf{u}_t - \mathrm{div}((1+\beta\mathbf{u}^2) \nabla \mathbf{u}) & = \mathbf{f}\\
\end{aligned}
```
with (some time-dependent) exterior force ``\mathbf{f}``. The parameter ``\beta`` steers the strength of the nonlinearity. 

The time integration will be performed by a solver from DifferentialEquations.jl or by the iternal backward Euler method of GradientRobustMultiPhysics.

Note: To run this example the DifferentialEquations.jl package has to be installed.
=#


module ExampleA05_DiffEQ

using GradientRobustMultiPhysics
using ExtendableGrids
using DifferentialEquations

## problem data
function exact_solution!(result,x, t)
    result[1] = x[1]*x[2]*(1-t)
    return nothing
end
function rhs!(beta)
    function closure(result,x,t)
        result[1] = -2*beta*(x[1]^3*x[2] + x[2]^3*x[1]) # = -div(beta*u^2*grad(u))
        result .*= (1-t)^3
        result[1] += -x[1]*x[2] ## = u_t
        return nothing
    end
end

## everything is wrapped in a main function
## the last four parametes steer the solver from DifferentialEquations.jl
## for beta = 0, abstol and reltol can be choosen much larger
function main(; verbosity = 0, nlevels = 3, timestep = 1e-1, T = 0.5, FEType = H1P1{1}, beta = 1,
    use_diffeq::Bool = true,
    solver = ImplicitEuler(autodiff = false),# Rosenbrock23(autodiff = false),
    adaptive_timestep = true,
    abstol = 1e-3,
    reltol = 1e-3,
    testmode = false)

    ## set log level
    set_verbosity(verbosity)

    ## initial grid and final time
    xgrid = uniform_refine(grid_unitsquare(Triangle2D),1);

    ## negotiate data functions to the package
    u = DataFunction(exact_solution!, [1,2]; name = "u", dependencies = "XT", bonus_quadorder = 5)
    ∇u = ∇(u)
    u_rhs = DataFunction(rhs!(beta), [1,1]; name = "f", dependencies = "XT", bonus_quadorder = 5)

    ## prepare nonlinear expression (1+u^2)*grad(u)
    function diffusion_kernel!(result, input)
        ## input = [u, grad(u)]
        result[1] = (1+beta*input[1]^2)*input[2]
        result[2] = (1+beta*input[1]^2)*input[3]
        return nothing
    end 
    nonlin_diffusion = NonlinearForm(Gradient, [Identity, Gradient], [1,1], diffusion_kernel!, [2,3]; name = "(1+ βu^2) ∇u ⋅ ∇v", bonus_quadorder = 2, newton = true)  

    ## generate problem description and assign nonlinear operator and data
    Problem = PDEDescription(beta == 0 ? "linear Poisson problem" : "nonlinear Poisson problem")
    add_unknown!(Problem; unknown_name = "u", equation_name = beta == 0 ? "linear Poisson problem" : "nonlinear Poisson equation")
    add_operator!(Problem, [1,1], beta == 0 ? LaplaceOperator() : nonlin_diffusion)
    add_boundarydata!(Problem, 1, [1,2,3,4], BestapproxDirichletBoundary; data = u)
    add_rhsdata!(Problem, 1,  LinearForm(Identity, u_rhs))

    ## define error evaluators
    L2Error = L2ErrorIntegrator(u, Identity; time = T)
    H1Error = L2ErrorIntegrator(∇u, Gradient; time = T)
    Results = zeros(Float64, nlevels, 2); NDofs = zeros(Int, nlevels)
    
    ## loop over levels
    for level = 1 : nlevels

        ## refine grid
        xgrid = uniform_refine(xgrid)

        ## generate FESpace and solution vector
        FES = FESpace{FEType}(xgrid)
        Solution = FEVector("u_h",FES)

        ## set initial solution
        interpolate!(Solution[1], u) 

        ## generate time-dependent solver
        sys = TimeControlSolver(Problem, Solution, BackwardEuler; timedependent_equations = [1], skip_update = [beta == 0 ? -1 : 1], nonlinear_iterations = beta == 0 ? 1 : 5)

        if use_diffeq == true
            ## use time integration by DifferentialEquations
            advance_until_time!(DifferentialEquations, sys, timestep, T; solver = solver, abstol = abstol, reltol = reltol, adaptive = adaptive_timestep)
        else
            ## use time control solver by GradientRobustMultiPhysics
            advance_until_time!(sys, timestep, T)
        end

        ## compute L2 and H1 errors and save data
        NDofs[level] = length(Solution.entries)
        Results[level,1] = sqrt(evaluate(L2Error,Solution[1]))
        Results[level,2] = sqrt(evaluate(H1Error,Solution[1]))
    end    

    ## print/plot convergence history
    print_convergencehistory(NDofs, Results; X_to_h = X -> X.^(-1/2), ylabels = ["|| u - u_h ||", "|| ∇(u - u_h) ||"])
   
end

function test()
    return main(; use_diffeq = false, nlevels = 1, testmode = true)
end


end
