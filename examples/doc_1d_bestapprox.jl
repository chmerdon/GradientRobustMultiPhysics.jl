#= 

# 1D L2-Bestapproximation
([source code](SOURCE_URL))

This example computes the L2-bestapproximation of some given scalar-valued function into the piecewise quadratic continuous polynomials.
Afterwards the L2 error is computed and the solution is plotted (using PyPlot)

=#

push!(LOAD_PATH, "../src")
using ExtendableGrids
ENV["MPLBACKEND"]="qt5agg"
using PyPlot
using GradientRobustMultiPhysics


## define some (vector-valued) function (to be L2-bestapproximated in this example)
## a user-defined (time-independent) function should always have this interface
function exact_function!(result,x)
    result[1] = (x[1]-1//2)*(x[1]-9//10)*(x[1]-1//3)*(x[1]-1//10)
end

## everything is wrapped in a main function
function main()

    ## generate mesh and uniform refine twice
    xgrid = simplexgrid([0.0,1//3,2//3,1.0])
    xgrid = uniform_refine(xgrid,2)
    
    ## setup a bestapproximation problem via a predefined prototype
    ## and an L2ErrorEvaluator that can be used later to compute the L2 error
    Problem = L2BestapproximationProblem(exact_function!,1, 1; bestapprox_boundary_regions = [1,2], bonus_quadorder = 4)
    L2ErrorEvaluator = L2ErrorIntegrator(exact_function!, Identity, 1, 1; bonus_quadorder = 4)

    ## choose some finite element type and generate a FESpace for the grid
    ## (here it is a one-dimensional H1-conforming P2 element H1P2{1,1})
    FEType = H1P2{1,1}
    FES = FESpace{FEType}(xgrid)

    ## generate a solution vector and solve the problem
    ## (the verbosity argument that many functions have steers the talkativity,
    ##  the larger the number, the more details)
    Solution = FEVector{Float64}("L2-Bestapproximation",FES)
    solve!(Solution, Problem; verbosity = 1)
    
    ## calculate the L2 error
    L2error = sqrt(evaluate(L2ErrorEvaluator,Solution[1]))
    println("\nL2error(BestApprox) = $L2error")
        
    ## evaluate/interpolate function at nodes and plot
    PyPlot.figure("plot of solution") 
    nodevals = zeros(Float64,1,size(xgrid[Coordinates],2))
    nodevalues!(nodevals,Solution[1],FES)
    ExtendableGrids.plot(xgrid, nodevals[1,:]; Plotter = PyPlot, label = "coarse approximation")

    ## to compare our discrete solution with a finer one, we interpolate the exact function
    ## again on some more refined mesh and also compute the L2 error on this one
    xgrid = uniform_refine(xgrid,3)
    FES = FESpace{FEType}(xgrid)
    Interpolation = FEVector{Float64}("fine-grid interpolation",FES)
    interpolate!(Interpolation[1], exact_function!)
    println("L2error(FineInterpol) = $(sqrt(evaluate(L2ErrorEvaluator,Interpolation[1])))")

    nodevals_fine = zeros(Float64,1,size(xgrid[Coordinates],2))
    nodevalues!(nodevals_fine,Interpolation[1],FES)
    ExtendableGrids.plot(xgrid, nodevals_fine[1,:]; Plotter = PyPlot, clear = false, color = (1,0,0), label = "fine interpolation")
    PyPlot.legend()
end

main()
