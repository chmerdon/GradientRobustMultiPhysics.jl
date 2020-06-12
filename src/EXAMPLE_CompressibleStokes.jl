
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
using Printf


include("testgrids.jl")


# problem data
viscosity = 1
lambda = - 1//3 * viscosity
c = 10
gamma = 1.4
M = 1 

function equation_of_state!(pressure,density)
    for j = 1 : length(density)
        pressure[j] = c*density[j]^gamma
    end
end

d = log(M/(c*(exp(1)^(1/c)-1.0)))
function exact_density!(result,x) # only exact for gamma = 1
    result[1] = 1.0 - (x[2] - 0.5)/c
end 

function rhs_gravity!(result,x)
    exact_density!(result,x)
    result[2] = -result[1]^(gamma-2) * gamma
    result[1] = 0.0
end   


function main()

    #####################################################################################
    #####################################################################################

    # meshing parameters
    xgrid = testgrid_mixedEG(); # initial grid
    #xgrid = split_grid_into(xgrid,Triangle2D) # if you want just triangles

    # uniform mesh refinement
    for j = 1:4
        xgrid = uniform_refine(xgrid)
    end

    # choose finite element type [velocity, density,  pressure]
    FETypes = [FiniteElements.H1BR{2}, FiniteElements.L2P0{1}, FiniteElements.L2P0{1}] # Bernardi--Raugel
    #FETypes = [FiniteElements.H1CR{2}, FiniteElements.L2P0{1}, FiniteElements.L2P0{1}] # Crouzeix--Raviart (needs smaller timesteps)

    #TestFunctionOperator = Identity # classical scheme
    TestFunctionOperator = ReconstructionIdentity{FiniteElements.HDIVRT0{2}} # gradient-robust scheme

    # solver parameters
    timestep = viscosity // (10*c)
    start_with_constant_density = false
    maxIterations = 300  # termination criterion 1 for nonlinear mode
    maxResidual = 1e-8 # termination criterion 2 for nonlinear mode
    verbosity = 0 # deepness of messaging (the larger, the more)

    #####################################################################################    
    #####################################################################################

    # load Stokes problem prototype and assign data
    StokesProblem = CompressibleNavierStokesProblem(equation_of_state!, rhs_gravity!, 2; timestep = timestep, viscosity = viscosity, lambda = lambda, nonlinear = false)
    StokesProblem.LHSOperators[1,2][1].operator1 = TestFunctionOperator
    StokesProblem.LHSOperators[1,2][1].store_operator = true
    StokesProblem.LHSOperators[1,3][1].store_operator = true
    append!(StokesProblem.BoundaryOperators[1], [1,2,3,4], HomogeneousDirichletBoundary)
    Base.show(StokesProblem)

    # define best approximation problem
    L2DensityBestapproximationProblem = L2BestapproximationProblem(exact_density!, 2, 1; bestapprox_boundary_regions = [], bonus_quadorder = 2)

    # generate FESpaces
    FESpaceVelocity = FiniteElements.FESpace{FETypes[1]}(xgrid)
    FESpacePressureDensity = FiniteElements.FESpace{FETypes[2]}(xgrid)

    # set initial values
    Solution = FEVector{Float64}("velocity",FESpaceVelocity)
    append!(Solution,"density",FESpacePressureDensity)
    append!(Solution,"pressure",FESpacePressureDensity)

    if start_with_constant_density
        for j = 1 : FESpacePressureDensity.ndofs
            Solution[2][j] = M
        end
    else
        InitialDensity = FEVector{Float64}("L2-Bestapproximation density",FESpacePressureDensity)
        solve!(InitialDensity, L2DensityBestapproximationProblem; verbosity = verbosity)

        for j = 1 : FESpacePressureDensity.ndofs
            Solution[2][j] = InitialDensity[1][j]
        end
    end
    equation_of_state!(Solution[3],Solution[2])


    # solve by manual solver config
    TCS = TimeControlSolver(StokesProblem, Solution, BackwardEuler; subiterations = [[1],[2],[3]], timedependent_equations = [1,2], verbosity = verbosity)

    change = 0.0
    for iteration = 1 : maxIterations
        change = advance(TCS, timestep)
        M = sum(Solution[2][:] .* xgrid[CellVolumes])
        println("  iteration $iteration | time = $(TCS.ctime) | change = $change | M = $M")
        if change < maxResidual
            println("  terminated (below tolerance)")
            break;
        end
    end
    
    # plot
    # split grid into triangles for plotter
    xgrid = split_grid_into(xgrid,Triangle2D)

    # plot triangulation
    PyPlot.figure(1)
    ExtendableGrids.plot(xgrid, Plotter = PyPlot)

    # plot solution
    nnodes = size(xgrid[Coordinates],2)
    nodevals = zeros(Float64,2,nnodes)

    # velocity
    nodevalues!(nodevals,Solution[1],FESpaceVelocity)
    PyPlot.figure(2)
    ExtendableGrids.plot(xgrid, nodevals[1,:][1:nnodes]; Plotter = PyPlot)

    # density
    nodevalues!(nodevals,Solution[2],FESpacePressureDensity)
    PyPlot.figure(3)
    ExtendableGrids.plot(xgrid, nodevals[1,:]; Plotter = PyPlot)

    # pressure
    nodevalues!(nodevals,Solution[3],FESpacePressureDensity)
    PyPlot.figure(4)
    ExtendableGrids.plot(xgrid, nodevals[1,:]; Plotter = PyPlot)


end


main()
