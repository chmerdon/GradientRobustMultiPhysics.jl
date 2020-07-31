
using ExtendableGrids
ENV["MPLBACKEND"]="qt5agg"
using PyPlot
using Printf
using Triangulate

push!(LOAD_PATH, "../src")
using GradientRobustMultiPhysics

# Cook membrane grid as triangles with four boundary regions (left, bottom, right, top)
function grid_cookmembrane(maxarea::Float64)
    triin=Triangulate.TriangulateIO()
    triin.pointlist=Matrix{Cdouble}([0 0; 0 -44; 48 0; 48 16]');
    triin.segmentlist=Matrix{Cint}([1 2 ; 2 3 ; 3 4 ; 4 1 ]')
    triin.segmentmarkerlist=Vector{Int32}([1, 2, 3, 4])
    xgrid = simplexgrid("pALVa$(@sprintf("%.15f",maxarea))", triin)
    xgrid[CellRegions] = VectorOfConstants(Int32(1),num_sources(xgrid[CellNodes]))
    xgrid[CellGeometries] = VectorOfConstants(Triangle2D,num_sources(xgrid[CellNodes]))
    return xgrid
end

# problem data
function neumann_force_right!(result,x)
    result[1] = 0.0
    result[2] = 10.0
end    

function main()

    #####################################################################################
    #####################################################################################

    # meshing parameters
    maxarea = 0.2
    xgrid = grid_cookmembrane(maxarea) 

    # problem parameters
    elasticity_modulus = 1000 # elasticity modulus
    poisson_number = 1//3 # Poisson number
    shear_modulus = (1/(1+poisson_number))*elasticity_modulus
    lambda = (poisson_number/(1-2*poisson_number))*shear_modulus

    # choose finite element type
    FEType = H1P1{2} # P1-Courant
    #FEType = H1P2{2,2} # P2

    # solver parameters
    verbosity = 1 # deepness of messaging (the larger, the more)

    # postprocess parameters
    plot_grid = false
    plot_stress = true
    factor_plotdisplacement = 4
    plot_displacement = true

    #####################################################################################    
    #####################################################################################

    # PDE description
    LinElastProblem = LinearElasticityProblem(2; shearmodulus = shear_modulus, lambda = lambda)
    # add Neumann boundary data
    add_rhsdata!(LinElastProblem, 1,  RhsOperator(Identity, neumann_force_right!, 2, 2; regions = [3], on_boundary = true, bonus_quadorder = 0))
    # add Dirichlet boundary data
    add_boundarydata!(LinElastProblem, 1, [1], HomogeneousDirichletBoundary)
    show(LinElastProblem)

    # generate FESpace
    FES = FESpace{FEType}(xgrid)

    # solve PDE
    Solution = FEVector{Float64}("displacement",FES)
    solve!(Solution, LinElastProblem; verbosity = verbosity)
    
    xgrid = split_grid_into(xgrid,Triangle2D)

    # plot triangulation
    if plot_grid
        PyPlot.figure("grid")
        ExtendableGrids.plot(xgrid, Plotter = PyPlot)
    end

    if plot_stress
        # plot solution
        PyPlot.figure("|eps(u)|")
        nnodes = size(xgrid[Coordinates],2)
        nodevals = zeros(Float64,4,nnodes)
        @time nodevalues!(nodevals,Solution[1],FES,SymmetricGradient)
        ExtendableGrids.plot(xgrid, sqrt.(nodevals[1,:].^2 + nodevals[2,:].^2 + nodevals[3,:].^2 + nodevals[4,:].^2); Plotter = PyPlot)
    end

    # plot displacement
    if plot_displacement
        PyPlot.figure("|u| on displaced grid")
        nnodes = size(xgrid[Coordinates],2)
        nodevals = zeros(Float64,2,nnodes)
        nodevalues!(nodevals,Solution[1],FES)
        xgrid[Coordinates] = xgrid[Coordinates] + factor_plotdisplacement*nodevals[[1,2],:]
        xCoordinates = xgrid[Coordinates]
        ExtendableGrids.plot(xgrid, sqrt.(nodevals[1,:].^2 + nodevals[2,:].^2); Plotter = PyPlot, isolines = 3)
        # quiver(xCoordinates[1,:],xCoordinates[2,:],nodevals[1,:],nodevals[2,:])
    end

end


main()
