#= 

# 2D Gresho vortex (transient Navier-Stokes-Problem)
([source code](SOURCE_URL))

This example runs the famous Gresho vortex example for two different discretisations, one classical and one pressure-robustly modified one,
simultanously.
=#

push!(LOAD_PATH, "../src")
using GradientRobustMultiPhysics
using ExtendableGrids
using Triangulate
#using VTKView
ENV["MPLBACKEND"]="qt5agg"
using PyPlot
using Printf


## initial data
function initial_velocity!(result,x)
    r = sqrt(x[1]^2+x[2]^2)
    if (r <= 0.2)
        result[1] = -5*x[2]
        result[2] = 5*x[1]
    elseif (r <= 0.4)
        result[1] = -2*x[2]/r + 5*x[2]
        result[2] = 2*x[1]/r - 5*x[1]
    else
        result[1] = 0.0;
        result[2] = 0.0;
    end
end

## grid generator that generates  unstructured simplex mesh
function grid_square(maxarea::Float64)
    triin=Triangulate.TriangulateIO()
    triin.pointlist=Matrix{Cdouble}([-1 -1; 1 -1; 1 1; -1 1]');
    triin.segmentlist=Matrix{Cint}([1 2 ; 2 3 ; 3 4 ; 4 1 ]')
    triin.segmentmarkerlist=Vector{Int32}([4, 1, 2, 3])
    xgrid = simplexgrid("pALVa$(@sprintf("%.15f",maxarea))", triin)
    xgrid[CellRegions] = VectorOfConstants(Int32(1),num_sources(xgrid[CellNodes]))
    xgrid[CellGeometries] = VectorOfConstants(Triangle2D,num_sources(xgrid[CellNodes]))
    return xgrid
end

## everything is wrapped in a main function
function main()
    #####################################################################################
    #####################################################################################

    ## choose grid
    xgrid = grid_square(2e-3)

    ## problem parameters
    viscosity = 1e-6
    timestep = 5e-4
    T = 1 # final time
    plot_every_nth_step = 5e-2/timestep # number of steps before redraw
    

    ## choose one of these (inf-sup stable) finite element type pairs
    #FETypes = [H1CR{2}, L2P0{1}] # Crouzeix--Raviart
    FETypes = [H1BR{2}, L2P0{1}] # Bernardi--Raugel

    reconstruction_operator = ReconstructionIdentity{HDIVRT0{2}}

    ## postprocess parameters
    plot_grid = false
    plot_pressure = false
    plot_velocity = true

    #####################################################################################    
    #####################################################################################

    ## load Stokes problem prototype and assign data
    StokesProblem = IncompressibleNavierStokesProblem(2; viscosity = viscosity, nonlinear = false)
    add_boundarydata!(StokesProblem, 1, [1,2,3,4], HomogeneousDirichletBoundary)

    ## generate a copy for pressure-robust variant
    StokesProblem2 = deepcopy(StokesProblem)

    ## define bestapproximation problems
    L2VelocityBestapproximationProblem = L2BestapproximationProblem(initial_velocity!, 2, 2; bestapprox_boundary_regions = [1,2,3,4], bonus_quadorder = 5, time = 0)

    ## generate FESpaces
    FESpaceVelocity = FESpace{FETypes[1]}(xgrid)
    FESpacePressure = FESpace{FETypes[2]}(xgrid)

    ## generate solution vectors
    Solution = FEVector{Float64}("Stokes velocity",FESpaceVelocity)
    append!(Solution,"Stokes pressure",FESpacePressure)
    Solution2 = deepcopy(Solution)

    # add IMEX version of nonlinear convection term
    add_rhsdata!(StokesProblem, 1, TLFeval(ConvectionOperator(1, Identity, 2, 2; testfunction_operator = Identity), Solution[1], Solution[1],-1))
    add_rhsdata!(StokesProblem2, 1, TLFeval(ConvectionOperator(1,  reconstruction_operator, 2, 2; testfunction_operator =  reconstruction_operator), Solution2[1], Solution2[1],-1))
   

    ## set initial solution ( = bestapproximation at time 0)
    VelocityBA = FEVector{Float64}("L2-Bestapproximation velocity",FESpaceVelocity)
    solve!(VelocityBA, L2VelocityBestapproximationProblem)
    Solution[1][:] = VelocityBA[1][:]
    Solution2[1][:] = VelocityBA[1][:]


    ## plot triangulation   
    xCoordinates = xgrid[Coordinates]
    nnodes = size(xCoordinates,2)
    if plot_grid
        PyPlot.figure("grid")
        ExtendableGrids.plot(xgrid, Plotter = PyPlot)
    end

    ## plot velocity
    if plot_velocity
        nodevals = zeros(Float64,2,nnodes)
        nodevalues!(nodevals,Solution[1],FESpaceVelocity)
        PyPlot.figure("velocity (classical)")
        ExtendableGrids.plot(xgrid, sqrt.(nodevals[1,:].^2+nodevals[2,:].^2); Plotter = PyPlot, isolines = 5)
        fill!(nodevals,0)
        nodevalues!(nodevals,Solution2[1],FESpaceVelocity)
        PyPlot.figure("velocity (pressure-robust)")
        ExtendableGrids.plot(xgrid, sqrt.(nodevals[1,:].^2+nodevals[2,:].^2); Plotter = PyPlot, isolines = 5)
    end

    ## generate time-dependent solver and chance rhs data
    TCS = TimeControlSolver(StokesProblem, Solution, BackwardEuler; timedependent_equations = [1], dt_testfunction_operator = [Identity], verbosity = 1)
    TCS2 = TimeControlSolver(StokesProblem2, Solution2, BackwardEuler; timedependent_equations = [1], dt_testfunction_operator = [reconstruction_operator], verbosity = 1)

    ## time loop
    change1 = 0.0
    change2 = 0.0
    maxIterations = ceil(T / timestep)
    for iteration = 1 : maxIterations
        try
            change1 = advance!(TCS, timestep; reuse_matrix = [true])
        catch
        end
        try
            change2 = advance!(TCS2, timestep; reuse_matrix = [true])
        catch
        end
        @printf("  iteration %4d of %d",iteration,maxIterations)
        @printf("  time = %.4e",TCS.ctime)
        @printf("  change1/change2 = %.4e/%.4e \n",change1,change2)
    
        ## update velocity plot
        if (iteration % plot_every_nth_step == 1) || plot_every_nth_step == 1
            if plot_velocity
                println("  updating plots...")
                xCoordinates = xgrid[Coordinates]
                try
                    fill!(nodevals,0)
                    nodevalues!(nodevals,Solution[1],FESpaceVelocity)
                    PyPlot.figure("velocity (classical)")
                    ExtendableGrids.plot(xgrid, sqrt.(nodevals[1,:].^2+nodevals[2,:].^2); Plotter = PyPlot, isolines = 5)
                catch
                end
                try
                    fill!(nodevals,0)
                    nodevalues!(nodevals,Solution2[1],FESpaceVelocity)
                    PyPlot.figure("velocity (pressure-robust)")
                    ExtendableGrids.plot(xgrid, sqrt.(nodevals[1,:].^2+nodevals[2,:].^2); Plotter = PyPlot, isolines = 5)
                catch
                end
            end
        end
    end

    ## plot final pressure
    if plot_pressure
        nodevals = zeros(Float64,1,nnodes)
        PyPlot.figure("pressure")
        nodevalues!(nodevals,Solution[2],FESpacePressure)
        ExtendableGrids.plot(xgrid, nodevals[1,:]; Plotter = PyPlot)
    end

end


main()
