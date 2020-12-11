#= 

# 1D L2-Bestapproximation
([source code](SOURCE_URL))

This example computes the L2-bestapproximation of some given scalar-valued function into the piecewise quadratic continuous polynomials.
Afterwards the L2 error is computed and the solution is plotted.

=#

module Example_1DBestapprox

using GradientRobustMultiPhysics
using ExtendableGrids

## define some (vector-valued) function (to be L2-bestapproximated in this example)
function exact_function!(result,x::Array{<:Real,1})
    result[1] = (x[1]-1//2)*(x[1]-9//10)*(x[1]-1//3)*(x[1]-1//10)
end

## everything is wrapped in a main function
function main(; Plotter = nothing, verbosity = 1, nrefs = 2, broken::Bool = false)

    ## generate mesh and uniform refine nrefs times
    xgrid = simplexgrid([0.0,1//3,2//3,1.0])
    xgrid = uniform_refine(xgrid,ref)

    ## negotiate exact_function! to the package
    user_function = DataFunction(exact_function!, [1,1]; name = "u_exact", dependencies = "X", quadorder = 4)
    
    ## setup a bestapproximation problem via a predefined prototype
    ## and an L2ErrorEvaluator that can be used later to compute the L2 error
    Problem = L2BestapproximationProblem(user_function; bestapprox_boundary_regions = [1,2])
    L2ErrorEvaluator = L2ErrorIntegrator(Float64, user_function, Identity)

    ## choose some finite element type and generate a FESpace for the grid
    ## (here it is a one-dimensional H1-conforming P2 element H1P2{1,1})
    ## the broken switch toggles a broken dofmap
    FEType = H1P2{1,1}
    FES = FESpace{FEType}(xgrid; broken = broken)

    ## generate a solution vector and solve the problem
    ## (the verbosity argument that many functions have steers the talkativity,
    ##  the larger the number, the more details)
    Solution = FEVector{Float64}("L2-Bestapproximation",FES)
    solve!(Solution, Problem; verbosity = verbosity)
    
    ## calculate the L2 error
    L2error = sqrt(evaluate(L2ErrorEvaluator,Solution[1]))
    println("\nL2error(BestApprox) = $L2error")

    ## to compare our discrete solution with a finer one, we interpolate the exact function
    ## again on some more refined mesh and also compute the L2 error on this one
    xgrid_fine = uniform_refine(xgrid,2)
    FES_fine = FESpace{FEType}(xgrid_fine)
    Interpolation = FEVector{Float64}("fine-grid interpolation",FES_fine)
    interpolate!(Interpolation[1], ON_CELLS, user_function)
    println("L2error(FineInterpol) = $(sqrt(evaluate(L2ErrorEvaluator,Interpolation[1])))")
        
    ## evaluate/interpolate function at nodes and plot
    if Plotter != nothing
        nodevals = zeros(Float64,1,size(xgrid[Coordinates],2))
        nodevalues!(nodevals,Solution[1],FES)
        p = ExtendableGrids.plot(xgrid, nodevals[1,:]; Plotter = Plotter, label = "coarse approximation")

        nodevals_fine = zeros(Float64,1,size(xgrid_fine[Coordinates],2))
        nodevalues!(nodevals_fine,Interpolation[1],FES_fine)
        ExtendableGrids.plot(xgrid_fine, nodevals_fine[1,:]; Plotter = Plotter, clear = false, p = p, color = (1,0,0), label = "fine interpolation")
    end
end

end
