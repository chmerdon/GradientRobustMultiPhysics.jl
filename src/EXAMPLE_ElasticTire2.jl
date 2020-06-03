
using FEXGrid
using ExtendableGrids
using ExtendableSparse
using FiniteElements
using FEAssembly
using PDETools
using QuadratureRules
#using VTKView
ENV["MPLBACKEND"]="qt5agg"
using PyPlot
using BenchmarkTools
using Printf


include("testgrids.jl")

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
    #FEType_rest = FiniteElements.H1P1{2}; ConnectionDofs1 = [ConnectionPoints;nnodes .+ ConnectionPoints]
    FEType_rest = FiniteElements.H1P2{2,2}; ConnectionDofs1 = [ConnectionPoints;(nnodes + num_sources(xgrid[FaceNodes])) .+ ConnectionPoints]

    # finite element type for spokes
    FEType_spokes = FiniteElements.H1P1{2}; ConnectionDofs2 = [ConnectionPoints;nnodes .+ ConnectionPoints]
    #FEType_spokes = FiniteElements.H1P2{2,1}; ConnectionDofs2 = [ConnectionPoints;(nnodes + num_sources(xgrid[CellNodes])) .+ ConnectionPoints]

    # other parameters
    verbosity = 2 # deepness of messaging (the larger, the more)
    factor_plotdisplacement = 100

    #####################################################################################    
    #####################################################################################

    # PDE description
    # LEFT-HAND-SIDE
    MyLHS = Array{Array{AbstractPDEOperator,1},2}(undef,2,2)
    MyLHS[1,1] = [HookStiffnessOperator2D(shear_modulus,lambda; regions = [1]), DiagonalOperator(1e60;regions = [2])]
    MyLHS[1,2] = []
    MyLHS[2,1] = []
    MyLHS[2,2] = [HookStiffnessOperator1D(elasticity_modulus_spoke; regions = [2]), DiagonalOperator(1e60;regions = [1])]

    # RIGHT-HAND SIDE
    MyRHS = Array{Array{AbstractPDEOperator,1},1}(undef,2)
    MyRHS[1] = [RhsOperator(Identity, neumann_force_center!, 2, 2; regions = [3], on_boundary = true, bonus_quadorder = 0)]
    MyRHS[2] = []

    # BOUNDARY OPERATOR
    MyBoundaryRest = BoundaryOperator(2,2)
    MyBoundaryEmpty = BoundaryOperator(2,2)
    append!(MyBoundaryRest, [1], HomogeneousDirichletBoundary)

    # GLOBAL CONSTRAINTS
    MyGlobalConstraints = Array{AbstractGlobalConstraint,1}(undef,1)
    MyGlobalConstraints[1] = CombineDofs(1,2,ConnectionDofs1,ConnectionDofs2)
    name = "linear elasticity problem"
    LinElastProblem = PDEDescription(name,MyLHS,MyRHS,[MyBoundaryRest,MyBoundaryEmpty,MyBoundaryEmpty],MyGlobalConstraints)
    show(LinElastProblem)

    # generate FESpace
    FESpace_rest = FiniteElements.FESpace{FEType_rest}(xgrid)
    FESpace_spokes = FiniteElements.FESpace{FEType_spokes}(xgrid)

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