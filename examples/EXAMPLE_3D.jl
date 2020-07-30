
using ExtendableGrids
ENV["MPLBACKEND"]="qt5agg"
using PyPlot

# load finite element module
push!(LOAD_PATH, "../src")
using GradientRobustMultiPhysics


include("../src/testgrids.jl")

# define some (vector-valued) function (to be L2-bestapproximated in this example)
function exact_function!(result,x)
    result[1] = x[1] + x[2]*x[2] + x[3]
    result[2] = x[3]
    result[3] = x[2]*x[2]
end

function main()

    verbosity = 1 # <-- increase/decrease this number to get more/less printouts on what is happening

    # load mesh and refine
    #xgrid = testgrid_cube_uniform(Hexahedron3D)
    xgrid = testgrid_cube_uniform(Tetrahedron3D)

    for j = 1:3
        xgrid = uniform_refine(xgrid)
    end

    # Define Bestapproximation problem via PDETooles_PDEProtoTypes
    Problem = L2BestapproximationProblem(exact_function!,3, 3; bestapprox_boundary_regions = [1,2,3,4,5,6], bonus_quadorder = 2)
    show(Problem)

    # choose some finite element space
    #FEType = H1P1{3}
    FEType = HDIVRT0{3}
    FES = FESpace{FEType}(xgrid)

    # solve the problem
    Solution = FEVector{Float64}("L2-Bestapproximation",FES)
    solve!(Solution, Problem; verbosity = verbosity)
    
    # calculate L2 error and L2 divergence error
    L2ErrorEvaluator = L2ErrorIntegrator(exact_function!, Identity, 3, 3; bonus_quadorder = 1)
    println("\nL2error(BestApprox) = $(sqrt(evaluate(L2ErrorEvaluator,Solution[1])))")

end

main()