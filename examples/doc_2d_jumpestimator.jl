#= 

# 2D Jump-Estimator
([source code](SOURCE_URL))

This example calculates an error estimator for an Hdiv approximation by evaluating the jump over all faces, i.e.
```math
\eta^2(\sigma) := \sum_{F \in \mathcal{F}} \mathrm{diam}(F) \| [[\sigma]] \|^2_{L^2(F)}
```
where ``\sigma`` is some finite element function. Here, we check the error of some Hdiv-conforming L2 bestapproximation.
(In future updates adaptive mesh refinement may be possible based on these indicators)

=#

module Example_2DJumpEstimator

using GradientRobustMultiPhysics

## define some vector field that should be approximated
function exact_function!(result,x)
    result[1] = x[1]^3+x[2]^2
    result[2] = -x[1]^2 + x[2] + 1
end

## everything is wrapped in a main function
function main(; verbosity = 1)

    ## generate a unit square mesh and refine
    xgrid = simplexgrid([0.0,1.0],[0.0,1.0])
    xgrid = uniform_refine(xgrid, 5)
    
    ## setup a bestapproximation problem via a predefined prototype
    Problem = L2BestapproximationProblem(exact_function!, 2, 2; bestapprox_boundary_regions = [1,2,3,4], bonus_quadorder = 3)

    ## choose some (inf-sup stable) finite element types
    FEType = HDIVBDM1{2}
    FES = FESpace{FEType}(xgrid)

    ## create a solution vector and solve the problem
    Solution = FEVector{Float64}("L2-Bestapproximation",FES)
    solve!(Solution, Problem; verbosity = verbosity)

    ## calculate estimator by evaluating the jumps on faces
    xFaceVolumes = xgrid[FaceVolumes]
    xFaceCells = xgrid[FaceCells]
    function L2jump_integrand(result, input, item)
        ## input = [IdentityDisc{Jump}]
        for j = 1 : length(input)
            result[j] = input[j]^2 * xFaceVolumes[item]
        end
        return nothing
    end
    jumpIntegrator = ItemIntegrator{Float64,ON_IFACES}(IdentityDisc{Jump},ItemWiseFunctionAction(L2jump_integrand, 2; bonus_quadorder = 2), [0])
    println("\nEstimator = $(sqrt(sum(evaluate(jumpIntegrator,[Solution[1]]))))")

    ## calculate L2 error and print results
    L2ErrorEvaluator = L2ErrorIntegrator(exact_function!, Identity, 2, 2)
    println("L2error(Id) = $(sqrt(evaluate(L2ErrorEvaluator,[Solution[1]])))")
    
end

end