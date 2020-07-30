
using ExtendableGrids
ENV["MPLBACKEND"]="qt5agg"
using PyPlot

# load finite element module
push!(LOAD_PATH, "../src")
using GradientRobustMultiPhysics

# define some (vector-valued) function (to be L2-bestapproximated in this example)
function exact_function!(result,x)
    result[1] = x[1]^2+x[2]^2
    result[2] = -x[1]^2 + 1.0
end

function main()

    verbosity = 1 # <-- increase/decrease this number to get more/less printouts on what is happening

    # load mesh and refine
    xgrid = simplexgrid([0.0,1.0],[0.0,1.0])
    for j=1:6
        xgrid = uniform_refine(xgrid)
    end
    
    # Define Bestapproximation problem via PDETooles_PDEProtoTypes
    Problem = L2BestapproximationProblem(exact_function!, 2, 2; bestapprox_boundary_regions = [1,2,3,4], bonus_quadorder = 2)
    show(Problem)

    # choose some finite element space
    FEType = HDIVRT0{2}
    #FEType = L2P1{2}
    #FEType = H1P2{2,2}
    FES = FESpace{FEType}(xgrid)
    show(FES)

    # solve the problem
    Solution = FEVector{Float64}("L2-Bestapproximation",FES)
    solve!(Solution, Problem; verbosity = verbosity)
    
    # calculate L2 error
    L2ErrorEvaluator = L2ErrorIntegrator(exact_function!, Identity, 2, 2; bonus_quadorder = 2)
    println("\nL2error(Id) = $(sqrt(evaluate(L2ErrorEvaluator,Solution[1])))")
        
    # evaluate/interpolate function at nodes and plot
    nodevals = zeros(Float64,2,size(xgrid[Coordinates],2))
    nodevalues!(nodevals,Solution[1],FES)
    PyPlot.figure("solution[1]")
    ExtendableGrids.plot(xgrid, nodevals[1,:]; Plotter = PyPlot)
    PyPlot.figure("solution[2]")
    ExtendableGrids.plot(xgrid, nodevals[1,:]; Plotter = PyPlot)
end

main()