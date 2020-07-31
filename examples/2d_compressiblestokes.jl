###############################
# EXAMPLE COMPRESSIBLE STOKES #
###############################
#
# Julia implementation of scheme from
#
# https://doi.org/10.1016/j.cma.2020.113069
# https://arxiv.org/abs/1911.01295
#
# here even for triangles/quads



using ExtendableGrids
ENV["MPLBACKEND"]="qt5agg"
using PyPlot
using Printf


push!(LOAD_PATH, "../src")
using GradientRobustMultiPhysics

include("../src/testgrids.jl")


# problem data
viscosity = 1
lambda = - 1//3 * viscosity
c = 1
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

    # uniform mesh refinement
    for j = 1:3
        xgrid = uniform_refine(xgrid)
    end

    # choose finite element type [velocity, density,  pressure]
    FETypes = [H1BR{2}, L2P0{1}, L2P0{1}] # Bernardi--Raugel
    #FETypes = [H1CR{2}, L2P0{1}, L2P0{1}] # Crouzeix--Raviart (needs smaller timesteps)

    #TestFunctionOperatorIdentity = Identity; TestFunctionOperatorDivergence = Divergence # classical scheme
    TestFunctionOperatorIdentity = ReconstructionIdentity{HDIVRT0{2}} # identity operator for gradient-robust scheme
    TestFunctionOperatorDivergence = ReconstructionDivergence{HDIVRT0{2}} # divergence operator for gradient-robust scheme

    # solver parameters
    timestep = viscosity // (2*c)
    start_with_constant_density = false
    maxIterations = 300  # termination criterion 1
    maxResidual = 1e-10 # termination criterion 2
    verbosity = 0 # deepness of messaging (the larger, the more)

    # postprocess parameters
    plot_grid = false
    plot_pressure = true
    plot_density = true
    plot_velocity = true

    #####################################################################################    
    #####################################################################################

    # load Stokes problem prototype and assign data
    StokesProblem = CompressibleNavierStokesProblem(equation_of_state!, rhs_gravity!, 2; viscosity = viscosity, lambda = lambda, nonlinear = false)

    # modify testfunction in operators
    StokesProblem.LHSOperators[1,1][2].operator1 = TestFunctionOperatorDivergence
    StokesProblem.LHSOperators[1,1][2].operator2 = TestFunctionOperatorDivergence
    StokesProblem.LHSOperators[1,2][1].operator1 = TestFunctionOperatorIdentity

    # store matrix of velo-pressure and velo-gravity operator
    # so that only a matrix-vector multiplication is needed in every iteration
    StokesProblem.LHSOperators[1,2][1].store_operator = true
    StokesProblem.LHSOperators[1,3][1].store_operator = true

    # assign boundary data
    add_boundarydata!(StokesProblem, 1,  [1,2,3,4], HomogeneousDirichletBoundary)
    Base.show(StokesProblem)

    # define best approximation problem
    L2DensityBestapproximationProblem = L2BestapproximationProblem(exact_density!, 2, 1; bestapprox_boundary_regions = [], bonus_quadorder = 2)

    # generate FESpaces
    FESpaceVelocity = FESpace{FETypes[1]}(xgrid)
    FESpacePressureDensity = FESpace{FETypes[2]}(xgrid)

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

    # generate time-dependent solver
    TCS = TimeControlSolver(StokesProblem, Solution, BackwardEuler; subiterations = [[1],[2],[3]], timedependent_equations = [2], verbosity = verbosity)

    # time loop
    change = 0.0
    for iteration = 1 : maxIterations
        change = advance!(TCS, timestep)
        M = sum(Solution[2][:] .* xgrid[CellVolumes])
        @printf("  iteration %4d",iteration)
        @printf("  time = %.4e",TCS.ctime)
        @printf("  change = %.4e",change)
        @printf("  M = %.4e \n",M)
        if change < maxResidual
            println("  terminated (below tolerance)")
            break;
        end
    end
    
    # split grid into triangles for plotter
    xgrid = split_grid_into(xgrid,Triangle2D)

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
        nodevalues!(nodevals,Solution[2],FESpacePressureDensity)
        ExtendableGrids.plot(xgrid, nodevals[1,:]; Plotter = PyPlot)
    end

    # plot density
    if plot_pressure
        nnodes = size(xgrid[Coordinates],2)
        nodevals = zeros(Float64,1,nnodes)
        PyPlot.figure("density")
        nodevalues!(nodevals,Solution[3],FESpacePressureDensity)
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
