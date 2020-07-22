
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
    xgrid = simplexgrid(IOStream;file = "EXAMPLE_KarmanStreet.sg")

    barycentric_refinement = false;
    reconstruct = false
    testfunction_operator = Identity

    # problem parameters
    nonlinear = true 

    # choose finite element type
    #FETypes = [H1P2{2,2}, H1P1{1}] # Taylor--Hood
    #FETypes = [H1MINI{2,2}, H1P1{1}] # MINI element
    #FETypes = [H1BR{2}, L2P0{1}] # Bernardi--Raugel
    FETypes = [H1BR{2}, L2P0{1}]; testfunction_operator = ReconstructionIdentity{HDIVRT0{2}} # Bernardi--Raugel gradient-robust
    #FETypes = [H1P2{2,2}, L2P1{1}]; barycentric_refinement = true # Scott-Vogelius 
 
    # solver parameters
    timestep = 1 // 10
    finaltime = 10
    plot_every_nth_step = 10 #
    verbosity = 1 # deepness of messaging (the larger, the more)

    #####################################################################################    
    #####################################################################################

    # load Stokes problem prototype and assign data
    StokesProblem = IncompressibleNavierStokesProblem(2; viscosity = viscosity, nonlinear = nonlinear, no_pressure_constraint = true)
    add_boundarydata!(StokesProblem, 1, [1,3,5], HomogeneousDirichletBoundary)
    add_boundarydata!(StokesProblem, 1, [4], BestapproxDirichletBoundary; data = bnd_inlet!, bonus_quadorder = 2, timedependent = true)

    if reconstruct && nonlinear
        # apply reconstruction operator
        StokesProblem.LHSOperators[1,1][2] = ConvectionOperator(1, 2, 2; testfunction_operator = testfunction_operator)
    end
    # store Laplacian to avoid reassembly in each iteration
    StokesProblem.LHSOperators[1,1][1].store_operator = true
    Base.show(StokesProblem)

    # generate FESpaces
    FESpaceVelocity = FESpace{FETypes[1]}(xgrid)
    FESpacePressure = FESpace{FETypes[2]}(xgrid)
    Solution = FEVector{Float64}("velocity",FESpaceVelocity)
    append!(Solution,"pressure",FESpacePressure)
    show(Solution)

    # plot triangulation
    PyPlot.figure(1)
    ExtendableGrids.plot(xgrid, Plotter = PyPlot)
    nodevals = zeros(Float64,2,size(xgrid[Coordinates],2))

    # generate time-dependent solver
    TCS = TimeControlSolver(StokesProblem, Solution, BackwardEuler; timedependent_equations = [1], dt_testfunction_operator = [testfunction_operator], verbosity = verbosity)
    
    # time loop
    change = 0.0
    maxIterations = ceil(finaltime / timestep)
    for iteration = 1 : maxIterations
        change = advance!(TCS, timestep)
        @printf("  iteration %4d |",iteration)
        @printf("  time = %.4e",TCS.ctime)
        @printf("  change = %.4e \n",change)

        # update solution plot
        if (iteration % plot_every_nth_step == 1) || plot_every_nth_step == 1
            println("  updating plots...")
            nodevalues!(nodevals,Solution[1],FESpaceVelocity)
            PyPlot.figure(2)
            ExtendableGrids.plot(xgrid, view(nodevals,1,:); Plotter = PyPlot)
            PyPlot.figure(3)
            nodevalues!(nodevals,Solution[2],FESpacePressure)
            ExtendableGrids.plot(xgrid, view(nodevals,1,:); Plotter = PyPlot)
        end
    
    end

    # show final solution
    nodevalues!(nodevals,Solution[1],FESpaceVelocity)
    PyPlot.figure(2)
    ExtendableGrids.plot(xgrid, view(nodevals,1,:); Plotter = PyPlot)
    PyPlot.figure(3)
    nodevalues!(nodevals,Solution[2],FESpacePressure)
    ExtendableGrids.plot(xgrid, view(nodevals,1,:); Plotter = PyPlot)

end


main()
