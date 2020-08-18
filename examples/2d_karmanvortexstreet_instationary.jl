push!(LOAD_PATH, "../src")
using GradientRobustMultiPhysics
using ExtendableGrids
#using VTKView
ENV["MPLBACKEND"]="qt5agg"
using PyPlot
using Printf


# inlet data
function bnd_inlet!(result,x,t)
    result[1] = 6*x[2]*(0.41-x[2])/(0.41*0.41)*max(0,sin(t*pi/8));
    result[2] = 0.0;
end

## everything is wrapped in a main function
function main()
    #####################################################################################
    #####################################################################################

    # load grid from sg file
    xgrid = simplexgrid(IOStream;file = "2d_grid_karmanvortexstreet.sg")

    barycentric_refinement = false;
    reconstruct = false
    testfunction_operator = Identity

    # problem parameters
    viscosity = 1e-4
    nonlinear = true    # add nonlinear convection term
    IMEX = true         # beware: may need smaller timestep !

    # choose finite element type
    #FETypes = [H1P2{2,2}, H1P1{1}] # Taylor--Hood
    #FETypes = [H1MINI{2,2}, H1P1{1}] # MINI element
    #FETypes = [H1BR{2}, L2P0{1}] # Bernardi--Raugel
    FETypes = [H1BR{2}, L2P0{1}]; testfunction_operator = ReconstructionIdentity{HDIVRT0{2}} # Bernardi--Raugel gradient-robust
    #FETypes = [H1P2{2,2}, L2P1{1}]; barycentric_refinement = true # Scott-Vogelius 
 
    # solver parameters
    timestep = 1e-4
    finaltime = 10
    plot_every_nth_step = 1e-2/timestep #
    verbosity = 1 # deepness of messaging (the larger, the more)

    # postprocess parameters
    plot_grid = false
    plot_pressure = false
    plot_velocity = true

    #####################################################################################    
    #####################################################################################

    # load Stokes problem prototype and assign data
    StokesProblem = IncompressibleNavierStokesProblem(2; viscosity = viscosity, nonlinear = false, no_pressure_constraint = true)
    add_boundarydata!(StokesProblem, 1, [1,3,5], HomogeneousDirichletBoundary)
    add_boundarydata!(StokesProblem, 1, [4], BestapproxDirichletBoundary; data = bnd_inlet!, bonus_quadorder = 2)

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
                ExtendableGrids.plot(xgrid, view(nodevals,1,:); Plotter = PyPlot)
            end
        
            # plot velocity
            if plot_velocity
                nodevalues!(nodevals,Solution[1],FESpaceVelocity)
                PyPlot.figure("velocity")
                for j = 1 : nnodes
                    nodevals[1,j] = sqrt(nodevals[1,j]^2 + nodevals[2,j]^2)
                end
                ExtendableGrids.plot(xgrid, view(nodevals,1,:); Plotter = PyPlot, isolines = 11)
            end
        end
    
    end

end


main()
