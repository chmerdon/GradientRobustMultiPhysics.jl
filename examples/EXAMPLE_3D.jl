
using ExtendableGrids
ENV["MPLBACKEND"]="qt5agg"
using PyPlot

# load finite element module
push!(LOAD_PATH, "../src")
using GradientRobustMultiPhysics


include("../src/testgrids.jl")

# define some (vector-valued) function (to be L2-bestapproximated in this example)
function exact_function!(result,x)
    result[1] = x[1] + x[2] + x[3]
end

function main()

    verbosity = 1 # <-- increase/decrease this number to get more/less printouts on what is happening

    # load mesh and refine
    xgrid = testgrid_cube_uniform()

    for j = 1:3
        xgrid = uniform_refine(xgrid)
    end

    # Define Bestapproximation problem via PDETooles_PDEProtoTypes
    # (actually in 1D interpolation and L2-bestapproximation coincide, but nevertheless...)
    Problem = L2BestapproximationProblem(exact_function!,3, 1; bestapprox_boundary_regions = [], bonus_quadorder = 2)
    add_boundarydata!(Problem, 1, [1,2,3,4,5,6], InterpolateDirichletBoundary; data = exact_function!, bonus_quadorder = 2)

    show(Problem)

    # choose some finite element space
    FEType = H1P1{1}
    FES = FESpace{FEType}(xgrid)

    # solve the problem
    Solution = FEVector{Float64}("L2-Bestapproximation",FES)
    solve!(Solution, Problem; verbosity = verbosity)
    
    # calculate L2 error and L2 divergence error
    L2ErrorEvaluator = L2ErrorIntegrator(exact_function!, Identity, 3, 1; bonus_quadorder = 1)
    println("\nL2error(BestApprox) = $(sqrt(evaluate(L2ErrorEvaluator,Solution[1])))")
        
    # evaluate/interpolate function at nodes and plot
    #PyPlot.figure(1) 
    #nodevals = zeros(Float64,1,size(xgrid[Coordinates],2))
    #nodevalues!(nodevals,Solution[1],FES)
    #ExtendableGrids.plot(xgrid, nodevals[1,:]; Plotter = PyPlot, label = "approximation")

end

main()