using FEXGrid
using ExtendableGrids
using FiniteElements
using FEAssembly
using PDETools
ENV["MPLBACKEND"]="qt5agg"
using PyPlot



# define some (vector-valued) function (to be L2-bestapproximated in this example)
function exact_function!(result,x)
    result[1] = x[1]^2+x[2]^2
    result[2] = -x[1]^2 + 1.0
end
# define its divergence (used in error calculation)
function exact_divergence!(result,x)
    result[1] = 2*x[1]
end


function main()

    verbosity = 2 # <-- increase/decrease this number to get more/less printouts on what is happening

    # load mesh and refine
    xgrid = simplexgrid([0.0,1.0],[0.0,1.0])
    for j=1:6
        xgrid = uniform_refine(xgrid)
    end
    
    # Define Bestapproximation problem via PDETooles_PDEProtoTypes
    Problem = L2BestapproximationProblem(exact_function!, 2, 2; bestapprox_boundary_regions = [1,2,3,4], bonus_quadorder = 2)

    # choose some finite element space
    FEType = FiniteElements.HDIVRT0{2}
    #FEType = FiniteElements.H1P2{2}
    FESpace = FiniteElements.FESpace{FEType}(xgrid)
    Base.show(FESpace)

    # solve the problem
    Solution = FEVector{Float64}("L2-Bestapproximation",FESpace)
    solve!(Solution, Problem; verbosity = verbosity)
    
    # calculate L2 error and L2 divergence error
    L2ErrorEvaluator = L2ErrorIntegrator(exact_function!, Identity, 2, 2; bonus_quadorder = 2)
    L2DivergenceErrorEvaluator = L2ErrorIntegrator(exact_divergence!, Divergence, 2, 1; bonus_quadorder = 1)
    println("\nL2error(Id) = $(sqrt(evaluate(L2ErrorEvaluator,Solution[1])))")
    println("L2error(div) = $(sqrt(evaluate(L2DivergenceErrorEvaluator,Solution[1])))")
        
    # evaluate/interpolate function at nodes and plot
    PyPlot.figure(1)
    nodevals = zeros(Float64,2,size(xgrid[Coordinates],2))
    nodevalues!(nodevals,Solution[1],FESpace)
    ExtendableGrids.plot(xgrid, nodevals[1,:]; Plotter = PyPlot)
end

main()