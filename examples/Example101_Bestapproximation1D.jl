#= 

# 101 : L2-Bestapproximation 1D
([source code](SOURCE_URL))

This example computes the L2-bestapproximation of some given scalar-valued function into the piecewise quadratic continuous polynomials.
Afterwards the L2 error is computed and the solution is plotted.

=#

module Example101_Bestapproximation1D

using GradientRobustMultiPhysics
using ExtendableGrids
using GridVisualize

## define some (vector-valued) function (to be L2-bestapproximated in this example)
function exact_function!(result,x)
    result[1] = (x[1]-1//2)*(x[1]-9//10)*(x[1]-1//3)*(x[1]-1//10)
end
const u = DataFunction(exact_function!, [1,1]; name = "u", dependencies = "X", quadorder = 4)

## everything is wrapped in a main function
function main(; Plotter = nothing, verbosity = 0, h = 1e-1)

    ## set log level
    set_verbosity(verbosity)

    ## generate coarse and fine mesh
    xgrid = simplexgrid(0:h:1)
    xgrid_fine = simplexgrid(0:h/10:1)
    
    ## setup a bestapproximation problem via a predefined prototype
    ## and an L2ErrorEvaluator that can be used later to compute the L2 error
    Problem = L2BestapproximationProblem(u; bestapprox_boundary_regions = [1,2])
    L2ErrorEvaluator = L2ErrorIntegrator(Float64, u, Identity)

    ## choose some finite element type and generate a FESpace for the grid
    ## (here it is a one-dimensional H1-conforming P2 element H1P2{1,1})
    FEType = H1P2{1,1}
    FES = FESpace{FEType}(xgrid)

    ## generate a solution vector and solve the problem on the coarse grid
    Solution = FEVector{Float64}("u_h",FES)
    solve!(Solution, Problem)
    
    ## we want to compare our discrete solution with a finer interpolation of u
    FES_fine = FESpace{FEType}(xgrid_fine)
    Interpolation = FEVector{Float64}("u_h (fine)",FES_fine)
    interpolate!(Interpolation[1], ON_CELLS, u)

    ## calculate the L2 errors
    L2error = sqrt(evaluate(L2ErrorEvaluator,Solution[1]))
    L2error_fine = (sqrt(evaluate(L2ErrorEvaluator,Interpolation[1])))
    println("\t|| u - u_h || = $L2error")
    println("\t|| u - u_h (fine) ||= $L2error_fine")
        
    ## evaluate/interpolate function at nodes and plot
    if Plotter !== nothing
        nodevals = zeros(Float64,1,size(xgrid[Coordinates],2))
        nodevalues!(nodevals,Solution[1],FES)
        p=GridVisualizer(Plotter=Plotter,layout=(1,1))
        scalarplot!(p[1,1],xgrid, view(nodevals,1,:), color=(0,1,0), label = "u_h (coarse bestapprox)")
        nodevals_fine = zeros(Float64,1,size(xgrid_fine[Coordinates],2))
        nodevalues!(nodevals_fine,Interpolation[1],FES_fine)
        scalarplot!(p[1,1],xgrid_fine, view(nodevals_fine,1,:), clear = false, color = (1,0,0), label = "u_h (fine interpolation)",show=true)
    end
end

end