#= 

# 2D Nonlinear Poisson (time-dependent)
([source code](SOURCE_URL))

This example computes a transient velocity ``\mathbf{u}`` solution of the nonlinear Poisson problem
```math
\begin{aligned}
\mathbf{u}_t - \mathrm{div}((1+\beta\mathbf{u}^2) \nabla \mathbf{u}) & = \mathbf{f}\\
\end{aligned}
```
with (some time-dependent) exterior force ``\mathbf{f}``. The parameter ``\beta`` steers the strength of the nonlinearity. 

The time integration will be performed by the internal backward Euler method of GradientRobustMultiPhysics.
A manufactureed quadratic-in-space and linear-in-time solution is prescribed to test if the solver computes the exact solution.

=#


module Example_2DTransientNonlinPoisson

using GradientRobustMultiPhysics
using Printf


## problem data
function exact_solution!(result,x::Array{<:Real,1}, t)
    result[1] = x[1]*x[2]*(1-t)
    return nothing
end
function exact_gradient!(result,x::Array{<:Real,1}, t)
    result[1] = x[2]
    result[2] = x[1]
    result .*= 1-t
    return nothing
end
function rhs!(result,x::Array{<:Real,1},t)
    result[1] = -2*(x[1]^3*x[2] + x[2]^3*x[1]) # = -div(beta*u^2*grad(u))
    result .*= (1-t)^3
    result[1] += -x[1]*x[2] ## = u_t
    return nothing
end
function diffusion_kernel!(result::Array{<:Real,1}, input::Array{<:Real,1})
    ## input = [u, grad(u)]
    result[1] = (1+input[1]^2)*input[2]
    result[2] = (1+input[1]^2)*input[3]
    return nothing
end 

## everything is wrapped in a main function
function main(; verbosity = 0, Plotter = nothing, nlevels = 3, timestep = 1e-1, T = 0.5, FEType = H1P2{1,2}, testmode = false)

    ## set log level
    set_verbosity(verbosity)

    ## initial grid and final time
    xgrid = grid_unitsquare(Triangle2D);

    ## negotiate data functions to the package
    user_function = DataFunction(exact_solution!, [1,1]; name = "u", dependencies = "XT", quadorder = 5)
    user_function_gradient = DataFunction(exact_gradient!, [2,1]; name = "∇(u)", dependencies = "XT", quadorder = 4)
    user_function_rhs = DataFunction(rhs!, [1,1]; name = "f", dependencies = "XT", quadorder = 5)

    ## prepare nonlinear expression (1+u^2)*grad(u)
    nonlin_diffusion = GenerateNonlinearForm("(1+u^2) ∇u ⋅ ∇v", [Identity, Gradient], [1,1], Gradient, diffusion_kernel!, [2,3]; quadorder = 2, ADnewton = true)  

    ## generate problem description and assign nonlinear operator and data
    Problem = PDEDescription("nonlinear Poisson problem")
    add_unknown!(Problem; unknown_name = "u", equation_name = "nonlinear Poisson equation")
    add_operator!(Problem, [1,1], nonlin_diffusion)
    add_boundarydata!(Problem, 1, [1,2,3,4], BestapproxDirichletBoundary; data = user_function)
    add_rhsdata!(Problem, 1,  RhsOperator(Identity, [0], user_function_rhs))

    ## define error evaluators
    L2ErrorEvaluator = L2ErrorIntegrator(Float64, user_function, Identity; time = T)
    H1ErrorEvaluator = L2ErrorIntegrator(Float64, user_function_gradient, Gradient; time = T)
    L2error = []; NDofs = []; H1error = []; 
    
    ## loop over levels
    for level = 1 : nlevels

        ## refine grid
        xgrid = uniform_refine(xgrid)

        ## generate FESpace and solution vector
        FES = FESpace{FEType}(xgrid)
        Solution = FEVector{Float64}("u_h",FES)
        push!(NDofs,length(Solution.entries))

        ## set initial solution
        interpolate!(Solution[1], user_function) 

        ## generate time-dependent solver
        sys = TimeControlSolver(Problem, Solution, BackwardEuler; timedependent_equations = [1], nonlinear_iterations = 5)

        ## use time control solver by GradientRobustMultiPhysics
        advance_until_time!(sys, timestep, T)

        ## plot solution at final time
        GradientRobustMultiPhysics.plot(xgrid, [Solution[1]], [Identity]; Plotter = Plotter)

        ## compute L2 and H1 error of all solutions
        append!(L2error,sqrt(evaluate(L2ErrorEvaluator,Solution[1])))
        append!(H1error,sqrt(evaluate(H1ErrorEvaluator,Solution[1])))
    end    

    ## ouput errors
    if testmode
        return L2error[1]
    else
        println("\n   NDOF  |   L2ERROR      order   ")
        order = 0
        for j=1:nlevels
            if j > 1
                order = log(L2error[j-1]/L2error[j]) / (log(NDofs[j]/NDofs[j-1])/2)
            end
            @printf("  %6d |",NDofs[j]);
            @printf(" %.5e ",L2error[j])
            @printf("   %.3f   \n",order)
        end
        println("\n   NDOF  |   H1ERROR      order   ")
        order = 0
        for j=1:nlevels
            if j > 1
                order = log(H1error[j-1]/H1error[j]) / (log(NDofs[j]/NDofs[j-1])/2)
            end
            @printf("  %6d |",NDofs[j]);
            @printf(" %.5e ",H1error[j])
            @printf("   %.3f   \n",order)
        end
    end
end

function test()
    return main(; nlevels = 1, FEType = H1P2{1,2}, testmode = true)
end

end


#=
### Output of default main() run
=#
Example_2DTransientNonlinPoisson.main()
