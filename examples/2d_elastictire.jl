
using ExtendableGrids
ENV["MPLBACKEND"]="qt5agg"
using PyPlot
using Printf

push!(LOAD_PATH, "../src")
using GradientRobustMultiPhysics


# Tire domain
#
# region 1 = wheel (2D triangles)
# region 2 = spokes (1D intervals)
#
# bfaceregion 1 = wheel exterior
# bfaceregion 2 = wheel interior
# bfaceregion 3 = inner wheel boundary
# bfaceregion 4 = spokes (yes they are marked as both cells and bfaces, do whatever you like with them)
#
# refinement can be steered with k
# number of spokes can be steered with S
# 
# note that this is certainly not an optimal mesh especially in the wheel interior region
function grid_tire(k::Int = 1, S::Int = 2)
    
    N = 4^k # number of points on circle

    r1 = 1.0 # outer tire radius
    r2 = 0.9 # innter tire radius
    r3 = 0.1 # innter circle radius

    xgrid=ExtendableGrid{Float64,Int32}()

    npoints = 3*N + 1# = three circles
    xCoordinates = zeros(Float64,2,npoints)
    for j = 1 : N
        xCoordinates[1,j] = r1*cos(2*pi*j/N)
        xCoordinates[2,j] = r1*sin(2*pi*j/N)
        xCoordinates[1,N+j] = r2*cos(2*pi*j/N)
        xCoordinates[2,N+j] = r2*sin(2*pi*j/N)
        xCoordinates[1,2*N+j] = r3*cos(2*pi*j/N)
        xCoordinates[2,2*N+j] = r3*sin(2*pi*j/N)
    end
    xgrid[Coordinates]=xCoordinates

    ntriangles = 3*N
    nspokes = Int(ceil(N/S))

    ConnectionPoints = zeros(Int,nspokes*2)
    xCellNodes=VariableTargetAdjacency(Int32)
    xCellRegions=ones(Int32,ntriangles+nspokes)
    xCellGeometries=Array{DataType,1}(undef,ntriangles+nspokes)
    for j = 1 : ntriangles
        xCellGeometries[j] = Triangle2D
    end
    for j = 1 : N
        if j==N
            append!(xCellNodes,[j 1 N+1])
            append!(xCellNodes,[j N+1 N+j])
            append!(xCellNodes,[2*N+j 2*N+1 3*N+1])
        else
            append!(xCellNodes,[j j+1 N+j+1])
            append!(xCellNodes,[j N+j+1 N+j])
            append!(xCellNodes,[2*N+j 2*N+j+1 3*N+1])
        end
    end
    for j = 1 : nspokes
        xCellGeometries[ntriangles+j] = Edge1D
        append!(xCellNodes,[N+j*S,2*N+j*S])
        xCellRegions[ntriangles+j] = 2
        ConnectionPoints[2*j-1] = N+j*S
        ConnectionPoints[2*j] = 2*N+j*S
    end
    xgrid[CellNodes] = xCellNodes
    xgrid[CellGeometries] = xCellGeometries
    ncells = num_sources(xCellNodes)
    xgrid[CellRegions]=xCellRegions

    nbfaces = 3*N+nspokes
    xBFaceRegions=zeros(Int32,nbfaces)
    xBFaceNodes=zeros(Int32,2,nbfaces)
    for j = 1 : N
        xBFaceRegions[j] = 1
        xBFaceRegions[N+j] = 2
        xBFaceRegions[2*N+j] = 3
        if j == N
            xBFaceNodes[:,j] = [j, 1]
            xBFaceNodes[:,N+j] = [N+1,N+j]
            xBFaceNodes[:,2*N+j] = [2*N+j,2*N+1]
        else
            xBFaceNodes[:,j] = [j, j+1]
            xBFaceNodes[:,N+j] = [N+j+1,N+j]
            xBFaceNodes[:,2*N+j] = [2*N+j,2*N+j+1]
        end
    end
    for j = 1 : nspokes
        xBFaceNodes[:,3*N+j] = [N+j*S,2*N+j*S]
        xBFaceRegions[3*N+j] = 4
    end
    xgrid[BFaceNodes]=xBFaceNodes
    xgrid[BFaceRegions]=xBFaceRegions
    xgrid[BFaceGeometries]=VectorOfConstants(Edge1D,nbfaces)
    xgrid[CoordinateSystem]=Cartesian2D

    return xgrid, ConnectionPoints
end

# problem data
function neumann_force_center!(result,x)
    result[1] = 0.0
    result[2] = -10.0
end    

function main()

    #####################################################################################
    #####################################################################################

    # meshing parameters
    xgrid, ConnectionPoints = grid_tire(4,8) # initial simplex grid
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

    # solve parameter
    verbosity = 1 # deepness of messaging (the larger, the more)

    # postprocess parameters
    plot_grid = true
    plot_stress = true
    factor_plotdisplacement = 200

    #####################################################################################    
    #####################################################################################

    # start from an empty PDEDescription for two globally defined unknowns
    # unknown 1 : displacement for rest
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
    add_rhsdata!(LinElastProblem, 1,  RhsOperator(Identity, [3], neumann_force_center!, 2, 2; on_boundary = true, bonus_quadorder = 0))
    add_boundarydata!(LinElastProblem, 1, [1], HomogeneousDirichletBoundary)

    # show PDEDescription
    show(LinElastProblem)

    # generate FESpace
    FES_rest = FESpace{FEType_rest}(xgrid)
    FES_spokes = FESpace{FEType_spokes}(xgrid)

    # solve PDE
    Solution = FEVector{Float64}("Displacement",FES_rest)
    append!(Solution,"Displacement Spokes",FES_spokes)
    solve!(Solution, LinElastProblem; verbosity = verbosity, dirichlet_penalty=1e60)
    
    xgrid = split_grid_into(xgrid,Triangle2D)
    
    # plot triangulation
    if plot_grid
        nnodes = size(xgrid[Coordinates],2)
        nodevals = zeros(Float64,2,nnodes)
        xgrid[Coordinates] = xgrid[Coordinates] + factor_plotdisplacement*nodevals[[1,2],:]
        xCoordinates = xgrid[Coordinates]
        PyPlot.figure("displaced grid")
        ExtendableGrids.plot(xgrid, Plotter = PyPlot)
    end

    if plot_stress
        # plot solution
        PyPlot.figure("|eps(u)|")
        nnodes = size(xgrid[Coordinates],2)
        nodevals = zeros(Float64,4,nnodes)
        @time nodevalues!(nodevals,Solution[1],FES_rest,SymmetricGradient; regions = [1])
        ExtendableGrids.plot(xgrid, sqrt.(nodevals[1,:].^2 + nodevals[2,:].^2 + nodevals[3,:].^2 + nodevals[4,:].^2); Plotter = PyPlot, isolines = 3)
    end

end


main()
