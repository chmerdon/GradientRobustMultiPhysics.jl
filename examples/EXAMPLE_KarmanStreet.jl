
using ExtendableGrids
#using VTKView
ENV["MPLBACKEND"]="qt5agg"
using PyPlot

push!(LOAD_PATH, "../src")
using GradientRobustMultiPhysics


# inlet data and viscosity for Karman vortex street example
viscosity = 1e-2

function bnd_inlet!(result,x)
    result[1] = 6*x[2]*(0.41-x[2])/(0.41*0.41);
    result[2] = 0.0;
end

function main()
    #####################################################################################
    #####################################################################################

    # load grid from sg file
    xgrid = simplexgrid(IOStream;file = "EXAMPLE_KarmanStreet.sg")

    barycentric_refinement = false;
    reconstruct = false

    # problem parameters
    nonlinear = true

    # choose finite element type
    #FETypes = [H1P2{2,2}, H1P1{1}] # Taylor--Hood
    #FETypes = [H1MINI{2,2}, H1P1{1}] # MINI element
    #FETypes = [H1BR{2}, L2P0{1}] # Bernardi--Raugel
    FETypes = [H1BR{2}, L2P0{1}]; reconstruct = true # Bernardi--Raugel gradient-robust
    #FETypes = [H1P2{2,2}, L2P1{1}]; barycentric_refinement = true # Scott-Vogelius 
 
    # solver parameters
    maxIterations = 20  # termination criterion 1 for nonlinear mode
    maxResidual = 1e-12 # termination criterion 2 for nonlinear mode
    verbosity = 1 # deepness of messaging (the larger, the more)

    #####################################################################################    
    #####################################################################################

    # load Stokes problem prototype and assign data
    StokesProblem = IncompressibleNavierStokesProblem(2; viscosity = viscosity, nonlinear = nonlinear, no_pressure_constraint = true)
    add_boundarydata!(StokesProblem, 1, [1,3,5], HomogeneousDirichletBoundary)
    add_boundarydata!(StokesProblem, 1, [4], BestapproxDirichletBoundary; data = bnd_inlet!, bonus_quadorder = 2)

    if reconstruct
        # apply reconstruction operator
        StokesProblem.LHSOperators[1,1][2] = ConvectionOperator(1, 2, 2; testfunction_operator = ReconstructionIdentity{HDIVRT0{2}})
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
    PyPlot.figure(1)
    ExtendableGrids.plot(xgrid, Plotter = PyPlot)

    # plot solution
    nnodes = size(xgrid[Coordinates],2)
    nodevals = zeros(Float64,2,nnodes)
    nodevalues!(nodevals,Solution[1],FESpaceVelocity)
    PyPlot.figure(2)
    ExtendableGrids.plot(xgrid, nodevals[1,:][1:nnodes]; Plotter = PyPlot)
    PyPlot.figure(3)
    nodevalues!(nodevals,Solution[2],FESpacePressure)
    ExtendableGrids.plot(xgrid, nodevals[1,:]; Plotter = PyPlot)

end


main()
