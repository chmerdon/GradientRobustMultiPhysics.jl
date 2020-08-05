
using ExtendableGrids
#using VTKView
ENV["MPLBACKEND"]="qt5agg"
using PyPlot
using Printf

push!(LOAD_PATH, "../src")
using GradientRobustMultiPhysics


# inlet data and viscosity for Karman vortex street example
viscosity = 1e-2

function bnd_inlet!(result,x,t)
    result[1] = 6*x[2]*(0.41-x[2])/(0.41*0.41)*max(0,sin(t*pi/8));
    result[2] = 0.0;
end

function main()
    #####################################################################################
    #####################################################################################

    # load grid from sg file
    xgrid = simplexgrid(IOStream;file = "2d_grid_karmanvortexstreet.sg")

    barycentric_refinement = false;
    reconstruct = false
    testfunction_operator = Identity

    # problem parameters
    nonlinear = true    # add nonlinear convection term
    IMEX = true         # beware: may need smaller timestep !

    # choose finite element type
    #FETypes = [H1P2{2,2}, H1P1{1}] # Taylor--Hood
    #FETypes = [H1MINI{2,2}, H1P1{1}] # MINI element
    #FETypes = [H1BR{2}, L2P0{1}] # Bernardi--Raugel
    FETypes = [H1BR{2}, L2P0{1}]; testfunction_operator = ReconstructionIdentity{HDIVRT0{2}} # Bernardi--Raugel gradient-robust
    #FETypes = [H1P2{2,2}, L2P1{1}]; barycentric_refinement = true # Scott-Vogelius 
 
    # solver parameters
    timestep = 1 // 50
    finaltime = 10
    plot_every_nth_step = 10 #
    verbosity = 1 # deepness of messaging (the larger, the more)

    # postprocess parameters
    plot_grid = false
    plot_pressure = true
    plot_velocity = true

    #####################################################################################    
    #####################################################################################

    # load Stokes problem prototype and assign data
    StokesProblem = IncompressibleNavierStokesProblem(2; viscosity = viscosity, nonlinear = false, no_pressure_constraint = true)
    add_boundarydata!(StokesProblem, 1, [1,3,5], HomogeneousDirichletBoundary)
    add_boundarydata!(StokesProblem, 1, [4], BestapproxDirichletBoundary; data = bnd_inlet!, bonus_quadorder = 2, timedependent = true)

    # generate FESpaces
    FESpaceVelocity = FESpace{FETypes[1]}(xgrid)
    FESpacePressure = FESpace{FETypes[2]}(xgrid)
    Solution = FEVector{Float64}("velocity",FESpaceVelocity)
    append!(Solution,"pressure",FESpacePressure)

    # add IMEX version of nonlinear term
    if nonlinear
        if IMEX
            add_rhsdata!(StokesProblem, 1, TLFeval(ConvectionOperator(1, ReconstructionIdentity{HDIVRT0{2}}, 2, 2; testfunction_operator = testfunction_operator), Solution[1], Solution[1],-1))
        else
            # store Laplacian to avoid reassembly in each iteration
            StokesProblem.LHSOperators[1,1][1].store_operator = true
            add_operator!(StokesProblem, [1,1], ConvectionOperator(1, testfunction_operator, 2, 2; testfunction_operator = testfunction_operator))
        end
    end


    # show problem and solution structure
    Base.show(StokesProblem)
    Base.show(Solution)

    # plot triangulation
    if plot_grid
        PyPlot.figure("grid")
        ExtendableGrids.plot(xgrid, Plotter = PyPlot)
    end
        

    # generate time-dependent solver
    TCS = TimeControlSolver(StokesProblem, Solution, BackwardEuler; timedependent_equations = [1], dt_testfunction_operator = [testfunction_operator], verbosity = verbosity)
    
    # time loop
    change = 0.0
    maxIterations = ceil(finaltime / timestep)
    xCoordinates = xgrid[Coordinates]
    nnodes = size(xCoordinates,2)
    nodevals = zeros(Float64,2,nnodes)
    for iteration = 1 : maxIterations
        change = advance!(TCS, timestep; reuse_matrix = true)
        @printf("  iteration %4d",iteration)
        @printf("  time = %.4e",TCS.ctime)
        @printf("  change = %.4e \n",change)

        # update solution plot
        if (iteration % plot_every_nth_step == 1) || plot_every_nth_step == 1
            println("  updating plots...")
            # plot pressure
            if plot_pressure
                PyPlot.figure("pressure")
                nodevalues!(nodevals,Solution[2],FESpacePressure)
                ExtendableGrids.plot(xgrid, nodevals[1,:]; Plotter = PyPlot)
            end
        
            # plot velocity (speed + quiver)
            if plot_velocity
                nodevalues!(nodevals,Solution[1],FESpaceVelocity)
                PyPlot.figure("velocity")
                ExtendableGrids.plot(xgrid, sqrt.(nodevals[1,:].^2+nodevals[2,:].^2); Plotter = PyPlot, isolines = 3)
                quiver(xCoordinates[1,:],xCoordinates[2,:],nodevals[1,:],nodevals[2,:])
            end
        end
    
    end

    # show final solution
    # plot pressure
    if plot_pressure
        PyPlot.figure("pressure")
        nodevalues!(nodevals,Solution[2],FESpacePressure)
        ExtendableGrids.plot(xgrid, nodevals[1,:]; Plotter = PyPlot)
    end

    # plot velocity (speed + quiver)
    if plot_velocity
        PyPlot.figure("velocity")
        ExtendableGrids.plot(xgrid, sqrt.(nodevals[1,:].^2+nodevals[2,:].^2); Plotter = PyPlot, isolines = 3)
        quiver(xCoordinates[1,:],xCoordinates[2,:],nodevals[1,:],nodevals[2,:])
    end

end


main()
