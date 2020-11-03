#= 

# 2D Poisson Problem + Error Estimator + H2 Error
([source code](SOURCE_URL))

This example computes the standard-residual error estimator for the $H^1$ error of some (vector-valued) $H^1$-conforming
approximation ``\mathbf{u}_h`` to the solution ``\mathbf{u}`` of some vector Poisson problem ``-\Delta \mathbf{u} = \mathbf{f}``, i.e.
```math
\eta^2(\mathbf{u}_h) := \sum_{T \in \mathcal{T}} \lvert T \rvert \| \mathbf{f} + \Delta \mathbf{u}_h \|^2_{L^2(T)}
+ \sum_{F \in \mathcal{F}} \lvert F \rvert \| [[D\mathbf{u}_h \mathbf{n}]] \|^2_{L^2(F)}
```
This example script showcases the evaluation of 2nd order derivatives, i.e. the Laplacian in the volume term of the error estimator and
the full Hessian for the computation of the error in the H2 semi-norm.

(In future updates adaptive mesh refinement may be possible based on these error indicators.)

=#

module Example_ErrorEstimatorPoisson

using GradientRobustMultiPhysics
using Printf

## exact solution u for the Poisson problem
function exact_function!(result,x)
    result[1] = x[1]^3
    result[2] = x[1]*x[2]
end
## ... and its derivatives
function exact_function_gradient!(result,x)
    result[1] = 3*x[1]^2 # = du1 / dx
    result[2] = 0        # = du1 / dy
    result[3] = x[2]     # = du2 / dx
    result[4] = x[1]     # = du2 / dy
end
## ... and its 2nd order derivatives
function exact_function_hessian!(result,x)
    result[1] = 6*x[1]  # = d^2 u1 / dx^2
    result[2] = 0       # = d^2 u1 / dxdy
    result[3] = 0       # = d^2 u1 / dydx
    result[4] = 0       # = d^2 u1 / dy^2
    result[5] = 0       # = d^2 u2 / dx^2
    result[6] = 1       # = d^2 u2 / dxdy
    result[7] = 1       # = d^2 u2 / dydx
    result[8] = 0       # = d^2 u2 / dy^2
end
## right-hand side for the Poisson problem, i.e. f = - \Delta u
function rhs!(result,x)
    result[1] = -6*x[1]
    result[2] = 0
end

## everything is wrapped in a main function
function main(; verbosity = 1, nlevels = 6)

    ## initial grid
    xgrid = simplexgrid([0.0,1.0],[0.0,1.0])

    ## choose some finite element
    FEType = H1P2{2,2}
    
    ## setup a bestapproximation problem via a predefined prototype
    Problem = PoissonProblem(2; ncomponents = 2, diffusion = 1.0)
    add_boundarydata!(Problem, 1, [1,2,3,4], BestapproxDirichletBoundary; data = exact_function!, bonus_quadorder = 3)
    add_rhsdata!(Problem, 1,  RhsOperator(Identity, [0], rhs!, 2, 2; bonus_quadorder = 1))

    ## setup exact error evaluations
    L2ErrorEvaluator = L2ErrorIntegrator(exact_function!, Identity, 2, 2; bonus_quadorder = 4)
    H1ErrorEvaluator = L2ErrorIntegrator(exact_function_gradient!, Gradient, 2, 4; bonus_quadorder = 3)
    H2ErrorEvaluator = L2ErrorIntegrator(exact_function_hessian!, Hessian, 2, 8; bonus_quadorder = 2)

    ## refinement loop (only uniform for now)
    NDofs = zeros(Int, nlevels)
    Results = zeros(Float64, nlevels, 4)
    for level = 1 : nlevels

        ## uniform mesh refinement
        xgrid = uniform_refine(xgrid)

        ## create a solution vector and solve the problem
        FES = FESpace{FEType}(xgrid)
        Solution = FEVector{Float64}("Discrete Solution",FES)
        solve!(Solution, Problem; verbosity = verbosity)
        NDofs[level] = length(Solution[1])

        ## error estimator jump term 
        xFaceVolumes = xgrid[FaceVolumes]
        function L2jump_integrand(result, input, item)
            for j = 1 : length(input)
                result[j] = input[j]^2 * xFaceVolumes[j]
            end
            return nothing
        end
        jumpIntegrator = ItemIntegrator{Float64,ON_IFACES}(GradientDisc{Jump},ItemWiseFunctionAction(L2jump_integrand, [4,4]; bonus_quadorder = 2), [0])
      
        ## error estimator volume term : h * ||f + Laplace(u_h)||
        xCellVolumes = xgrid[CellVolumes]
        function L2vol_integrand(result, input, x, item)
            rhs!(result,x)
            for j = 1 : length(input)
                input[j] += result[j]
                result[j] = input[j]^2 * xCellVolumes[j]
            end
            return nothing
        end
        jumpIntegrator = ItemIntegrator{Float64,ON_CELLS}(Laplacian,ItemWiseXFunctionAction(L2vol_integrand, [2,2], 2; bonus_quadorder = 2), [0])
          
        ## complete error estimator
        vol_error = sqrt(sum(evaluate(jumpIntegrator,[Solution[1]])))
        jump_error = sqrt(sum(evaluate(jumpIntegrator,[Solution[1]])))

        ## calculate L2 error, H1 error, estimator and H2 error Results and write to results
        Results[level,1] = sqrt(evaluate(L2ErrorEvaluator,[Solution[1]]))
        Results[level,2] = sqrt(evaluate(H1ErrorEvaluator,[Solution[1]]))
        Results[level,3] = sqrt(jump_error^2 + vol_error^2)
        Results[level,4] = sqrt(evaluate(H2ErrorEvaluator,[Solution[1]]))

    end
    
    ## print results
    @printf("\n  NDOFS  |   L2ERROR      order   |   H1ERROR      order   | H1-ESTIMATOR   order   |   H2ERROR      order   ")
    @printf("\n=========|========================|========================|========================|=========================\n")
    
    order = 0
    for j=1:nlevels
        @printf("  %6d |",NDofs[j]);
        for k = 1 : 4
            if j > 1
                order = log(Results[j-1,k]/Results[j,k]) / (log(NDofs[j]/NDofs[j-1])/2)
            end
            @printf(" %.5e ",Results[j,k])
            @printf("   %.3f   |",order)
        end
        @printf("\n")
    end
    
end

end