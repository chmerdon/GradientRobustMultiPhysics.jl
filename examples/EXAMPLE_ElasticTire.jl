
using ExtendableGrids
ENV["MPLBACKEND"]="qt5agg"
using PyPlot
using Printf


push!(LOAD_PATH, "../src")
using JUFELIA

include("../src/testgrids.jl")

# problem data
function neumann_force_center!(result,x)
    result[1] = 0.0
    result[2] = -10.0
end    

function main()

    #####################################################################################
    #####################################################################################

    # meshing parameters
    xgrid, ConnectionPoints = testgrid_tire(3,4) # initial simplex grid

    # problem parameters
    elasticity_modulus = 100000 # elasticity modulus
    elasticity_modulus_spoke = 1000 # elasticity modulus for spokes
    poisson_number = 0.3 # Poisson number
    shear_modulus = (1/(1+poisson_number))*elasticity_modulus
    lambda = (poisson_number/(1-2*poisson_number))*shear_modulus

    # choose finite element type
    FEType = H1P1{2}

    # other parameters[]
    verbosity = 2 # deepness of messaging (the larger, the more)
    factor_plotdisplacement = 50

    #####################################################################################    
    #####################################################################################

    # PDE description
    LinElastProblem = LinearElasticityProblem(2; shearmodulus = shear_modulus, lambda = lambda)
    LinElastProblem.LHSOperators[1,1] = [HookStiffnessOperator2D(shear_modulus,lambda; regions = [1]),
                                         HookStiffnessOperator1D(elasticity_modulus_spoke; regions = [2])]
    append!(LinElastProblem.BoundaryOperators[1], [1], HomogeneousDirichletBoundary)
    push!(LinElastProblem.RHSOperators[1], RhsOperator(Identity, neumann_force_center!, 2, 2; regions = [3], on_boundary = true, bonus_quadorder = 0))
    show(LinElastProblem)

    # generate FESpace
    FES = FESpace{FEType}(xgrid)

    # solve PDE
    Solution = FEVector{Float64}("Displacement",FES)
    solve!(Solution, LinElastProblem; verbosity = verbosity, dirichlet_penalty=1e10)
    
    # plot triangulation
    PyPlot.figure(1)
    xgrid = split_grid_into(xgrid,Triangle2D)
    ExtendableGrids.plot(xgrid, Plotter = PyPlot)

    # plot solution
    PyPlot.figure(2)
    nnodes = size(xgrid[Coordinates],2)
    nodevals = zeros(Float64,3,nnodes)
    nodevalues!(nodevals,Solution[1],FES)
    nodevals[3,:] = sqrt.(nodevals[1,:].^2 + nodevals[2,:].^2)
    ExtendableGrids.plot(xgrid, nodevals[3,:]; Plotter = PyPlot)

    # plot displaced triangulation
    xgrid[Coordinates] = xgrid[Coordinates] + factor_plotdisplacement*nodevals[[1,2],:]
    PyPlot.figure(3)
    ExtendableGrids.plot(xgrid, Plotter = PyPlot)

end


main()
