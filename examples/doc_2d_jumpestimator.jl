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

using GradientRobustMultiPhysics
using ExtendableGrids
ENV["MPLBACKEND"]="qt5agg"
using PyPlot


## define some vector field that should be approximated
function exact_function!(result,x)
    result[1] = x[1]^3+x[2]^2
    result[2] = -x[1]^2 + x[2] + 1
end

## everything is wrapped in a main function
function main()

    ## generate a unit square mesh and refine
    xgrid = simplexgrid([0.0,1.0],[0.0,1.0])
    xgrid = uniform_refine(xgrid,3)
    
    ## setup a bestapproximation problem via a predefined prototype
    Problem = L2BestapproximationProblem(exact_function!, 2, 2; bestapprox_boundary_regions = [1,2,3,4], bonus_quadorder = 3)

    ## choose some (inf-sup stable) finite element types
    FEType = HDIVBDM1{2}
    FES = FESpace{FEType}(xgrid)

    ## create a solution vector and solve the problem
    Solution = FEVector{Float64}("L2-Bestapproximation",FES)
    solve!(Solution, Problem; verbosity = 1)

    ## calculate estimator by evaluating the jumps on faces
    xFaceVolumes = xgrid[FaceVolumes]
    xFaceCells = xgrid[FaceCells]
    function L2jump_integrand(result, input, item)
        for j = 1 : length(input)
            result[j] = input[j]^2 * xFaceVolumes[j]
        end
        return nothing
    end
    jumpIntegrator = ItemIntegrator{Float64,ON_FACES}(FaceJumpIdentity,ItemWiseFunctionAction(L2jump_integrand, 2; bonus_quadorder = 2), [0])
    nfaces = num_sources(xgrid[FaceNodes])
    jump4face = zeros(Float64,2,nfaces)
    evaluate!(jump4face,jumpIntegrator,Solution[1]; verbosity = 1)

    ## set jumps on boundary faces to zero
    jump4face[:,xgrid[BFaces]] .= 0

    ## calculate L2 error and L2 divergence error
    L2ErrorEvaluator = L2ErrorIntegrator(exact_function!, Identity, 2, 2)
    println("\nL2error(Id) = $(sqrt(evaluate(L2ErrorEvaluator,Solution[1])))")
    println("\nEstimator = $(sqrt(sum(jump4face[:])))")
        
    ## plot the vector field
    PyPlot.figure("|u|")
    nodevals = zeros(Float64,2,size(xgrid[Coordinates],2))
    nodevalues!(nodevals,Solution[1],FES)
    ExtendableGrids.plot(xgrid, sqrt.(nodevals[1,:].^2 + nodevals[2,:].^2); Plotter = PyPlot, isolines = 5)
end

main()