
using ExtendableGrids
ENV["MPLBACKEND"]="qt5agg"
using PyPlot
using Printf

push!(LOAD_PATH, "../src")
using GradientRobustMultiPhysics

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
    nnodes = num_sources(xgrid[Coordinates])

    # problem parameters
    elasticity_modulus = 100000 # elasticity modulus
    elasticity_modulus_spoke = 1000 # elasticity modulus for spokes
    poisson_number = 0.3 # Poisson number
    shear_modulus = (1/(1+poisson_number))*elasticity_modulus
    lambda = (poisson_number/(1-2*poisson_number))*shear_modulus

    # choose finite element type for wheel and mid area
    #FEType_rest = H1P1{2}; ConnectionDofs1 = [ConnectionPoints;nnodes .+ ConnectionPoints]
    FEType_rest = H1P2{2,2}; ConnectionDofs1 = [ConnectionPoints;(nnodes + num_sources(xgrid[FaceNodes])) .+ ConnectionPoints]

    # finite element type for spokes
    FEType_spokes = H1P1{2}; ConnectionDofs2 = [ConnectionPoints;nnodes .+ ConnectionPoints]
    #FEType_spokes = H1P2{2,1}; ConnectionDofs2 = [ConnectionPoints;(nnodes + num_sources(xgrid[CellNodes])) .+ ConnectionPoints]

    # other parameters
    verbosity = 2 # deepness of messaging (the larger, the more)
    factor_plotdisplacement = 100

    #####################################################################################    
    #####################################################################################

    # start from an empty PDEDescription for two globally defined unknowns
    # unknonw 1 : displacement for rest
    # unknown 2 : displacement for spokes
    LinElastProblem = PDEDescription("linear elasticity problem", 2, [2,2], 2)

    # add PDEoperator for region 1 (triangles) and 2 (spokes = 1DEdges)
    add_operator!(LinElastProblem,[1,1],HookStiffnessOperator2D(shear_modulus,lambda; regions = [1])) # 2D deformation
    add_operator!(LinElastProblem,[1,1],DiagonalOperator(1e60;regions = [2])) # penalization to zero on spokes
    add_operator!(LinElastProblem,[2,2],HookStiffnessOperator1D(elasticity_modulus_spoke; regions = [2])) # 1D deformation along tangent on spokes
    add_operator!(LinElastProblem,[2,2],DiagonalOperator(1e60;regions = [1])) # penalization to zero on rest

    # add constraint that dofs at boundary of region 1 and 2 are the same
    add_constraint!(LinElastProblem,CombineDofs(1,2,ConnectionDofs1,ConnectionDofs2))
    
    # add boundary data
    add_rhsdata!(LinElastProblem, 1,  RhsOperator(Identity, neumann_force_center!, 2, 2; regions = [3], on_boundary = true, bonus_quadorder = 0))
    add_boundarydata!(LinElastProblem, 1, [1], HomogeneousDirichletBoundary)

    # show PDEDescription
    show(LinElastProblem)

    # generate FESpace
    FESpace_rest = FESpace{FEType_rest}(xgrid)
    FESpace_spokes = FESpace{FEType_spokes}(xgrid)

    # solve PDE
    Solution = FEVector{Float64}("Displacement",FESpace_rest)
    append!(Solution,"Displacement Spokes",FESpace_spokes)
    solve!(Solution, LinElastProblem; verbosity = verbosity, dirichlet_penalty=1e60)
    
    # plot triangulation
    PyPlot.figure(1)
    xgrid = split_grid_into(xgrid,Triangle2D)
    ExtendableGrids.plot(xgrid, Plotter = PyPlot)

    # plot solution
    PyPlot.figure(2)
    nnodes = size(xgrid[Coordinates],2)
    nodevals = zeros(Float64,3,nnodes)
    nodevalues!(nodevals,Solution[1],FESpace_rest)
    nodevals[3,:] = sqrt.(nodevals[1,:].^2 + nodevals[2,:].^2)
    ExtendableGrids.plot(xgrid, nodevals[3,:]; Plotter = PyPlot)

    # plot displaced triangulation
    xgrid[Coordinates] = xgrid[Coordinates] + factor_plotdisplacement*nodevals[[1,2],:]
    PyPlot.figure(3)
    ExtendableGrids.plot(xgrid, Plotter = PyPlot)

end


main()