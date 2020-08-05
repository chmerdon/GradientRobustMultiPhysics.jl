
using ExtendableGrids
#using VTKView
ENV["MPLBACKEND"]="qt5agg"
using PyPlot

push!(LOAD_PATH, "../src")
using GradientRobustMultiPhysics


# inlet data for Karman vortex street example
function bnd_inlet!(result,x)
    result[1] = 6*x[2]*(0.41-x[2])/(0.41*0.41);
    result[2] = 0.0;
end

function main()
    #####################################################################################
    #####################################################################################

    # load grid from sg file
    xgrid = simplexgrid(IOStream;file = "2d_grid_karmanvortexstreet.sg")

    # problem parameters
    viscosity = 2e-3
    nonlinear = true    # add nonlinear convection term?
    barycentric_refinement = false; reconstruct = false # do not change this line

    # choose finite element type
    #FETypes = [H1P2{2,2}, H1P1{1}] # Taylor--Hood
    #FETypes = [H1MINI{2,2}, H1P1{1}] # MINI element
    #FETypes = [H1BR{2}, L2P0{1}] # Bernardi--Raugel
    FETypes = [H1BR{2}, L2P0{1}]; reconstruct = true # Bernardi--Raugel gradient-robust
    #FETypes = [H1P2{2,2}, L2P1{1}]; barycentric_refinement = true # Scott-Vogelius (on barycentric refined mesh, much more runtime!)
 
    # solver parameters
    maxIterations = 50  # termination criterion 1 for nonlinear mode
    maxResidual = 1e-12 # termination criterion 2 for nonlinear mode
    verbosity = 1 # deepness of messaging (the larger, the more)

    # postprocess parameters
    plot_grid = false
    plot_pressure = true
    plot_velocity = true

    #####################################################################################    
    #####################################################################################

    if barycentric_refinement
        xgrid = barycentric_refine(xgrid)
    end

    # load Stokes problem prototype and assign data
    StokesProblem = IncompressibleNavierStokesProblem(2; viscosity = viscosity, nonlinear = nonlinear, no_pressure_constraint = true)
    add_boundarydata!(StokesProblem, 1, [1,3,5], HomogeneousDirichletBoundary)
    add_boundarydata!(StokesProblem, 1, [4], BestapproxDirichletBoundary; data = bnd_inlet!, bonus_quadorder = 2)

    if reconstruct
        # apply reconstruction operator
        StokesProblem.LHSOperators[1,1][2] = ConvectionOperator(1, ReconstructionIdentity{HDIVRT0{2}}, 2, 2; testfunction_operator = ReconstructionIdentity{HDIVRT0{2}})
    end
    # store Laplacian to avoid reassembly in each iteration
    StokesProblem.LHSOperators[1,1][1].store_operator = true
    Base.show(StokesProblem)

    # generate FESpaces
    FESpaceVelocity = FESpace{FETypes[1]}(xgrid)
    FESpacePressure = FESpace{FETypes[2]}(xgrid)

    # solve Stokes problem
    Solution = FEVector{Float64}("velocity",FESpaceVelocity)
    append!(Solution,"pressure",FESpacePressure)
    solve!(Solution, StokesProblem; verbosity = verbosity, maxIterations = maxIterations, maxResidual = maxResidual)

    # plot triangulation
    if plot_grid
        PyPlot.figure("grid")
        ExtendableGrids.plot(xgrid, Plotter = PyPlot)
    end

    # plot pressure
    if plot_pressure
        nnodes = size(xgrid[Coordinates],2)
        nodevals = zeros(Float64,1,nnodes)
        PyPlot.figure("pressure")
        nodevalues!(nodevals,Solution[2],FESpacePressure)
        ExtendableGrids.plot(xgrid, nodevals[1,:]; Plotter = PyPlot)
    end

    # plot velocity (speed + quiver)
    if plot_velocity
        xCoordinates = xgrid[Coordinates]
        nodevals = zeros(Float64,2,nnodes)
        nodevalues!(nodevals,Solution[1],FESpaceVelocity)
        PyPlot.figure("velocity")
        ExtendableGrids.plot(xgrid, sqrt.(nodevals[1,:].^2+nodevals[2,:].^2); Plotter = PyPlot, isolines = 3)
        quiver(xCoordinates[1,:],xCoordinates[2,:],nodevals[1,:],nodevals[2,:])
    end

end


main()
