#= 

# 2D User-defined Linear Solvers
([source code](SOURCE_URL))

This example revisits the nonlinear Poisson example from the introductory examples and showcases how to define a user-specified linear solver.

=#

module Example_2DCustomLinSolver

using GradientRobustMultiPhysics
using ExtendableSparse
using Printf


## first define a subtype of AbstractLinearSystem, which is later given as an optional parameter to the problem solve! call
mutable struct MySolver{T} <: GradientRobustMultiPhysics.AbstractLinearSystem{T}
    x::AbstractVector{T}
    A::ExtendableSparseMatrix{T,Int64}
    b::AbstractVector{T}
    ## add stuff here that you need e.g. for preconditioners
    MySolver{T}(x,A,b) where {T} = new{T}(x,A,b)
end

## you need to define update! and solve! functions for your new subtype
function GradientRobustMultiPhysics.update!(LS::MySolver{T}) where {T}
    ## this function is called before the solve (if other solver configuration not cause to skip it)
    ## do anything here (e.g. updating the preconditioner)
    print("\n                                            Hi! update! is called at start and every skip_update time...")
end
function GradientRobustMultiPhysics.solve!(LS::MySolver{T}) where {T}
    ## this function is called to solve the linear system
    print("\n                                            Hi! solve! under way...")
    LS.x .= LS.A \ LS.b
end


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
function main(; Plotter = nothing, verbosity = 0, nrefinements = 5, FEType = H1P1{1}, testmode = false, skip_update = 2)

    ## set log level
    set_verbosity(verbosity)

    ## choose initial mesh
    xgrid = uniform_refine(grid_unitsquare(Triangle2D),nrefinements)

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
    add_unknown!(Problem; unknown_name = "u", equation_name = "nonlinear Poisson equation")
    add_operator!(Problem, [1,1], nonlin_diffusion)
    add_boundarydata!(Problem, 1, [1,2,3,4], BestapproxDirichletBoundary; data = user_function)
    add_rhsdata!(Problem, 1,  RhsOperator(Identity, [0], user_function_rhs; store = true))

    ## create finite element space and solution vector
    FES = FESpace{FEType}(xgrid)
    Solution = FEVector{Float64}("u_h",FES)

    ## solve the problem (here the newly defined linear solver type is used)
    solve!(Solution, Problem; linsolver = MySolver{Float64}, skip_update = [skip_update])

    ## calculate error
    L2ErrorEvaluator = L2ErrorIntegrator(Float64, user_function, Identity)
    H1ErrorEvaluator = L2ErrorIntegrator(Float64, user_function_gradient, Gradient)
    println("\tL2error = $(sqrt(evaluate(L2ErrorEvaluator,Solution[1])))")
    println("\tH1error = $(sqrt(evaluate(H1ErrorEvaluator,Solution[1])))")

    ## plot
    GradientRobustMultiPhysics.plot(xgrid, [Solution[1]], [Identity]; Plotter = Plotter)
end


end
