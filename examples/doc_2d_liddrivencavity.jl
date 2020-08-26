#= 

# 2D Lid-driven cavity (Navier--Stokes)
([source code](SOURCE_URL))

This example solves the lid-driven cavity problem where one seeks
a velocity ``\mathbf{u}`` and pressure ``\mathbf{p}`` of the incompressible Navier--Stokes problem
```math
\begin{aligned}
- \mu \Delta \mathbf{u} + (\mathbf{u} \cdot \nabla) \mathbf{u} + \nabla p & = 0\\
\mathrm{div}(u) & = 0
\end{aligned}
```
where ``\mathbf{u} = (1,0)`` along the top boundary of a square domain.



=#

push!(LOAD_PATH, "../src")
using GradientRobustMultiPhysics
using Triangulate
ENV["MPLBACKEND"]="qt5agg"
using PyPlot
using Printf


## data
function boundary_data_top!(result,x)
    result[1] = 1.0;
    result[2] = 0.0;
end

## grid generator that generates  unstructured simplex mesh
function grid_square(maxarea::Float64)
    triin=Triangulate.TriangulateIO()
    triin.pointlist=Matrix{Cdouble}([-1 -1; 1 -1; 1 1; -1 1]');
    triin.segmentlist=Matrix{Cint}([1 2 ; 2 3 ; 3 4 ; 4 1 ]')
    triin.segmentmarkerlist=Vector{Int32}([1, 2, 3, 4])
    xgrid = simplexgrid("pALVa$(@sprintf("%.15f",maxarea))", triin)
    xgrid[CellRegions] = VectorOfConstants(Int32(1),num_sources(xgrid[CellNodes]))
    xgrid[CellGeometries] = VectorOfConstants(Triangle2D,num_sources(xgrid[CellNodes]))
    return xgrid
end

## everything is wrapped in a main function
function main()
    #####################################################################################
    #####################################################################################

    ## grid
    xgrid = grid_square(1e-3)

    ## problem parameters
    viscosity = 1e-3
    nonlinear = true
    barycentric_refinement = false # do not change
    maxIterations = 50  # termination criterion 1 for nonlinear mode
    maxResidual = 1e-10 # termination criterion 2 for nonlinear mode
    anderson_iterations = 5 # number of Anderson iterations

    ## choose one of these (inf-sup stable) finite element type pairs
    ReconstructionOperator = Identity
    #FETypes = [H1P2{2,2}, H1P1{1}] # Taylor--Hood
    #FETypes = [H1CR{2}, L2P0{1}] # Crouzeix--Raviart
    #FETypes = [H1MINI{2,2}, H1P1{1}] # MINI element on triangles only
    #FETypes = [H1MINI{2,2}, H1CR{1}] # MINI element on triangles/quads
    FETypes = [H1BR{2}, L2P0{1}] # Bernardi--Raugel
    #FETypes = [H1P2{2,2}, L2P1{1}]; barycentric_refinement = true # Scott-Vogelius 

    ## postprocess parameters
    plot_grid = false
    plot_pressure = false
    plot_velocity = true

    #####################################################################################    
    #####################################################################################

    ## load Stokes problem prototype and assign data
    StokesProblem = IncompressibleNavierStokesProblem(2; viscosity = viscosity, nonlinear = nonlinear)
    if nonlinear
        ## store matrix of Laplace operator for nonlinear solver
        StokesProblem.LHSOperators[1,1][1].store_operator = true   
    end
    add_boundarydata!(StokesProblem, 1, [1,2,4], HomogeneousDirichletBoundary)
    add_boundarydata!(StokesProblem, 1, [3], BestapproxDirichletBoundary; data = boundary_data_top!, bonus_quadorder = 0)
    Base.show(StokesProblem)

    ## uniform mesh refinement
    ## in case of Scott-Vogelius we use barycentric refinement
    if barycentric_refinement == true
        xgrid = barycentric_refine(xgrid)
    end

    ## generate FESpaces
    FESpaceVelocity = FESpace{FETypes[1]}(xgrid)
    FESpacePressure = FESpace{FETypes[2]}(xgrid)

    ## solve Stokes problem
    Solution = FEVector{Float64}("Stokes velocity",FESpaceVelocity)
    append!(Solution,"Stokes pressure",FESpacePressure)
    solve!(Solution, StokesProblem; verbosity = 1, AndersonIterations = anderson_iterations, maxIterations = maxIterations, maxResidual = maxResidual)

    
    ## split grid into triangles for plotter
    xgrid = split_grid_into(xgrid,Triangle2D)
    xCoordinates = xgrid[Coordinates]
    nnodes = size(xgrid[Coordinates],2)

    ## plot triangulation
    if plot_grid
        PyPlot.figure("grid")
        ExtendableGrids.plot(xgrid, Plotter = PyPlot)
    end

    ## plot pressure
    if plot_pressure
        nodevals = zeros(Float64,1,nnodes)
        PyPlot.figure("pressure")
        nodevalues!(nodevals,Solution[2],FESpacePressure)
        ExtendableGrids.plot(xgrid, nodevals[1,:]; Plotter = PyPlot)
    end

    ## plot velocity (speed + quiver)
    if plot_velocity
        nodevals = zeros(Float64,2,nnodes)
        nodevalues!(nodevals,Solution[1],FESpaceVelocity)
        PyPlot.figure("velocity")
        ExtendableGrids.plot(xgrid, sqrt.(nodevals[1,:].^2+nodevals[2,:].^2); Plotter = PyPlot, isolines = 3)
        quiver(xCoordinates[1,:],xCoordinates[2,:],nodevals[1,:],nodevals[2,:])
    end
end


main()
