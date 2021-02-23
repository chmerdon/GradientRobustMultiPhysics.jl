#= 

# 2D Equilibration Error Estimation (Global)
([source code](SOURCE_URL))

This example computes a global equilibration error estimator for the $H^1$ error of some $H^1$-conforming
approximation ``u_h`` to the solution ``u`` of some Poisson problem ``-\Delta u = f`` on an L-shaped domain, i.e.
```math
\eta^2(\sigma_h) := \| \sigma_h - \nabla u_h \|^2_{L^2(T)}
```
where ``\sigma_h`` is an Hdiv-conforming approximation of the exact ``\sigma`` in the dual mixed problem
```math
\sigma - \nabla u = 0
\quad \text{and} \quad
\mathrm{div}(\sigma) + f = 0
```
by solving the problem globally.


!!! note

    Equilibration error estimators yield guaranteed upper bounds (efficiency index above 1) for the H1 error
    possibly with some additional term that weighs in the oscillations of $f$, which are zero in this example,
    and some additional terms that quantifies the Dirichlet boundary error, which is neglected here.

    See the local equilibrated version for a less costly alternative.

=#

module Example_EQLshape

using GradientRobustMultiPhysics
using ExtendableGrids
using Printf

## exact solution u for the Poisson problem
function exact_function!(result,x::Array{<:Real,1})
    result[1] = atan(x[2],x[1])
    if result[1] < 0
        result[1] += 2*pi
    end
    result[1] = sin(2*result[1]/3)
    result[1] *= (x[1]^2 + x[2]^2)^(1/3)
end
## ... and its gradient
function exact_function_gradient!(result,x::Array{<:Real,1})
    result[1] = atan(x[2],x[1])
    if result[1] < 0
        result[1] += 2*pi
    end
    ## du/dy = du/dr * sin(phi) + (1/r) * du/dphi * cos(phi)
    result[2] = sin(2*result[1]/3) * sin(result[1]) + cos(2*result[1]/3) * cos(result[1])
    result[2] *= (x[1]^2 + x[2]^2)^(-1/6) * 2/3 
    ## du/dx = du/dr * cos(phi) - (1/r) * du/dphi * sin(phi)
    result[1] = sin(2*result[1]/3) * cos(result[1]) - cos(2*result[1]/3) * sin(result[1])
    result[1] *= (x[1]^2 + x[2]^2)^(-1/6) * 2/3 
end

## everything is wrapped in a main function
function main(; verbosity = 1, nlevels = 12, theta = 1//2, Plotter = nothing)

    ## initial grid
    xgrid = grid_lshape(Triangle2D)

    ## choose some finite elements for primal and dual problem
    FEType = H1P1{1}
    FETypeDual = [HDIVBDM1{2},H1P0{1}]
    
    ## negotiate data functions to the package
    user_function = DataFunction(exact_function!, [1,2]; name = "u_exact", dependencies = "X", quadorder = 5)
    user_function_gradient = DataFunction(exact_function_gradient!, [2,2]; name = "grad(u_exact)", dependencies = "X", quadorder = 4)

    ## setup Poisson problem
    Problem = PoissonProblem(2; ncomponents = 1, diffusion = 1.0)
    add_boundarydata!(Problem, 1, [2,3,4,5,6,7], BestapproxDirichletBoundary; data = user_function)
    add_boundarydata!(Problem, 1, [1,8], HomogeneousDirichletBoundary)

    ## setup dual mixed Poisson problem
    DualProblem = PDEDescription("dual mixed formulation")
    add_unknown!(DualProblem; unknown_name = "Stress", equation_name = "stress equation")
    add_operator!(DualProblem, [1,1], ReactionOperator(DoNotChangeAction(2)))
    add_rhsdata!(DualProblem, 1, RhsOperator(NormalFlux, [2,3,4,5,6,7], user_function; on_boundary = true))
    add_unknown!(DualProblem; unknown_name = "Lagrange multiplier for divergence", equation_name = "divergence constraint")
    add_operator!(DualProblem, [1,2], LagrangeMultiplier(Divergence))

    ## setup exact error evaluations
    L2ErrorEvaluator = L2ErrorIntegrator(Float64, user_function, Identity)
    H1ErrorEvaluator = L2ErrorIntegrator(Float64, user_function_gradient, Gradient)
    L2ErrorEvaluatorDual = L2ErrorIntegrator(Float64, user_function_gradient, Identity)

    ## define error estimator : || sigma_h - nabla u_h ||^2_{L^2(T)}
    ## this can be realised via a kernel function
    function eqestimator_kernel(result, input)
        ## input = [Identity(sigma_h), Gradient(u_h)]
        result[1] = (input[1] - input[3])^2 + (input[2] - input[4])^2
        return nothing
    end
    estimator_action_kernel = ActionKernel(eqestimator_kernel, [1,4]; name = "estimator kernel", dependencies = "", quadorder = 2)
    ## ... which generates an action...
    estimator_action = Action(Float64,estimator_action_kernel)
    ## ... which is used inside an ItemIntegrator
    EQIntegrator = ItemIntegrator(Float64,ON_CELLS,[Identity, Gradient],estimator_action)
          
    ## refinement loop (only uniform for now)
    NDofs = zeros(Int, nlevels)
    NDofsDual = zeros(Int, nlevels)
    Results = zeros(Float64, nlevels, 4)
    Solution = nothing
    DualSolution = nothing
    for level = 1 : nlevels

        ## create a solution vector and solve the problem
        FES = FESpace{FEType}(xgrid)
        Solution = FEVector{Float64}("Discrete Solution",FES)
        solve!(Solution, Problem; verbosity = verbosity - 1)
        NDofs[level] = length(Solution[1])

        ## solve the dual problem
        FESDual = [FESpace{FETypeDual[1]}(xgrid),FESpace{FETypeDual[2]}(xgrid)]
        DualSolution = FEVector{Float64}("Discrete Dual Solution",FESDual)
        NDofsDual[level] = length(DualSolution.entries)
        solve!(DualSolution, DualProblem; verbosity = verbosity - 1)

        if verbosity > 0
            println("\n  SOLVE LEVEL $level")
            println("    ndofs = $(NDofs[level])")
            println("    ndofsDual = $(NDofsDual[level])")
        end

        ## evaluate eqilibration error estimator
        error4cell = zeros(Float64,1,num_sources(xgrid[CellNodes]))
        evaluate!(error4cell, EQIntegrator, [DualSolution[1], Solution[1]])

        ## calculate L2 error, H1 error, estimator, dual L2 error and write to results
        Results[level,1] = sqrt(evaluate(L2ErrorEvaluator,[Solution[1]]))
        Results[level,2] = sqrt(evaluate(H1ErrorEvaluator,[Solution[1]]))
        Results[level,3] = sqrt(sum(error4cell))
        Results[level,4] = sqrt(evaluate(L2ErrorEvaluatorDual,[DualSolution[1]]))
        if verbosity > 0
            println("  ESTIMATE")
            println("    estim H1 error = $(Results[level,3])")
            println("    exact H1 error = $(Results[level,2])")
            println("     dual L2 error = $(Results[level,4])")
        end

        ## mesh refinement
        if theta >= 1
            ## uniform mesh refinement
            xgrid = uniform_refine(xgrid)
        else
            ## adaptive mesh refinement
            ## refine by red-green-blue refinement (incl. closuring)
            facemarker = bulk_mark(xgrid, view(error4cell,1,:), theta; verbosity = verbosity)
            xgrid = RGB_refine(xgrid, facemarker; verbosity = verbosity)
        end
    end
    
    ## plot
    GradientRobustMultiPhysics.plot(Solution, [0,1], [Identity,Identity]; Plotter = Plotter, verbosity = verbosity, use_subplots = false)
    
    ## print results
    @printf("\n  NDOFS  |   L2ERROR      order   |   H1ERROR      order   | H1-ESTIMATOR   order      efficiency   ")
    @printf("\n=========|========================|========================|========================================\n")
    order = 0
    for j=1:nlevels
        @printf("  %6d |",NDofs[j]);
        for k = 1 : 3
            if j > 1
                order = log(Results[j-1,k]/Results[j,k]) / (log(NDofs[j]/NDofs[j-1])/2)
            end
            @printf(" %.5e ",Results[j,k])
            if k == 3
                @printf("   %.3f       %.3f",order,Results[j,k]/Results[j,k-1])
            else
                @printf("   %.3f   |",order)
            end
        end
        @printf("\n")
    end
    
end

end