
using ExtendableGrids
ENV["MPLBACKEND"]="qt5agg"
using PyPlot

# load finite element module
push!(LOAD_PATH, "../src")
using GradientRobustMultiPhysics

# define some (vector-valued) function (to be L2-bestapproximated in this example)
function exact_function!(result,x)
    result[1] = (x[1]-1//2)*(x[1]-9//10)*(x[1]-1//3)*(x[1]-1//10)
end

function main()

    verbosity = 1 # <-- increase/decrease this number to get more/less printouts on what is happening

    # load mesh and refine
    xgrid = simplexgrid([0.0,1//3,2//3,1.0])
    for j=1:2
        xgrid = uniform_refine(xgrid)
    end
    
    # Define Bestapproximation problem via PDETooles_PDEProtoTypes
    # (actually in 1D interpolation and L2-bestapproximation coincide, but nevertheless...)
    Problem = L2BestapproximationProblem(exact_function!,1, 1; bestapprox_boundary_regions = [1,2], bonus_quadorder = 2)
    show(Problem)

    # choose some finite element space
    FEType = H1P2{1,1}
    FES = FESpace{FEType}(xgrid)
    show(FES)

    # solve the problem
    Solution = FEVector{Float64}("L2-Bestapproximation",FES)
    solve!(Solution, Problem; verbosity = verbosity)
    
    # calculate L2 error and L2 divergence error
    L2ErrorEvaluator = L2ErrorIntegrator(exact_function!, Identity, 1, 1; bonus_quadorder = 2)
    println("\nL2error(BestApprox) = $(sqrt(evaluate(L2ErrorEvaluator,Solution[1])))")
        
    # evaluate/interpolate function at nodes and plot
    PyPlot.figure(1) 
    nodevals = zeros(Float64,1,size(xgrid[Coordinates],2))
    nodevalues!(nodevals,Solution[1],FES)
    ExtendableGrids.plot(xgrid, nodevals[1,:]; Plotter = PyPlot, label = "approximation")

    # refine some more and also plot interpolated exact function 
    for j=1:3
        xgrid = uniform_refine(xgrid)
    end
    FES = FESpace{FEType}(xgrid)
    Interpolation = FEVector{Float64}("fine-grid interpolation",FES)
    interpolate!(Interpolation[1], exact_function!)

    nodevals_fine = zeros(Float64,1,size(xgrid[Coordinates],2))
    nodevalues!(nodevals_fine,Interpolation[1],FES)
    ExtendableGrids.plot(xgrid, nodevals_fine[1,:]; Plotter = PyPlot, clear = false, color = (1,0,0), label = "exact")
    PyPlot.legend()
    println("L2error(FineInterpol) = $(sqrt(evaluate(L2ErrorEvaluator,Interpolation[1])))")
end

main()