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

    uniform_grid = true; nrefinements = 5; maxarea = 4.0^(1-nrefinements)/2
    #uniform_grid = false; maxarea = 2e-3
    ## choose grid

    ## problem parameters
    viscosity = 1e-2 #1e-6
    timestep = 1e-5
    verbosity = 1
    T = 10 # final time
    use_rotationform = true
    use_BDM1_insteadof_RT0 = true
    plot_every_nth_step = 10 #5e-2/timestep # number of steps before redraw
    write_every_nth_step = 1e-2/timestep # number of steps before rewrite into logfile

    #linsolver = DirectUMFPACK
    linsolver = IterativeBigStabl_LUPC
    #convection_timevariant = "IMEX"; maxlureuse = [-1]
    convection_timevariant = "OSEEN"; maxlureuse = [5]
    nonlinear_iterations = 5

    

    ## choose one of these (inf-sup stable) finite element type pairs
    #FETypes = [H1CR{2}, L2P0{1}] # Crouzeix--Raviart
    FETypes = [H1BR{2}, L2P0{1}] # Bernardi--Raugel


    ## postprocess parameters
    plot_grid = false
    plot_pressure = false
    plot_velocity = true
    export_vtk = true

    #####################################################################################    
    #####################################################################################

    # generate grid
    if uniform_grid
        xgrid = uniform_refine(grid_square(4.0),nrefinements)
    else
        xgrid = createMesh(sqrt(maxarea))
    end
    xgrid = split_grid_into(xgrid, Triangle2D)

    ## load Stokes problem prototype and assign data
    StokesProblem = IncompressibleNavierStokesProblem(2; viscosity = viscosity, nonlinear = false)
    add_boundarydata!(StokesProblem, 1, [1,2,3,4], HomogeneousDirichletBoundary)

    ## generate a copy for pressure-robust variant
    StokesProblem2 = deepcopy(StokesProblem)

    ## define bestapproximation problems
    L2VelocityBestapproximationProblem = L2BestapproximationProblem(initial_velocity!, 2, 2; bestapprox_boundary_regions = [1,2,3,4], bonus_quadorder = 5, time = 0)
    
    # define ItemIntegrators for L2/H1 error computation
    L2VelocityErrorEvaluator = L2ErrorIntegrator(initial_velocity!, Identity, 2, 2; bonus_quadorder = 5)

    ## generate FESpaces
    FESpaceVelocity = FESpace{FETypes[1]}(xgrid)
    FESpacePressure = FESpace{FETypes[2]}(xgrid)

    ## generate solution vectors
    Solution = FEVector{Float64}("Stokes velocity",FESpaceVelocity)
    append!(Solution,"Stokes pressure",FESpacePressure)
    Solution2 = deepcopy(Solution)

    if use_BDM1_insteadof_RT0
        reconstruction_operator = ReconstructionIdentity{HDIVBDM1{2}}
    else
        reconstruction_operator = ReconstructionIdentity{HDIVRT0{2}}
    end

    # add IMEX version of nonlinear convection term
    if use_rotationform
        convection_classical = ConvectionRotationFormOperator(1, Identity, 2, 2; testfunction_operator = Identity)
        convection_probust = ConvectionRotationFormOperator(1,  reconstruction_operator, 2, 2; testfunction_operator =  reconstruction_operator)
    else
        convection_classical = ConvectionOperator(1, Identity, 2, 2; testfunction_operator = Identity)
        convection_probust = ConvectionOperator(1,  reconstruction_operator, 2, 2; testfunction_operator =  reconstruction_operator)
    end
    if convection_timevariant == "IMEX"
        add_rhsdata!(StokesProblem, 1, TLFeval(convection_classical, Solution[1], Solution[1],-1))
        add_rhsdata!(StokesProblem2, 1, TLFeval(convection_probust, Solution2[1], Solution2[1],-1))
    elseif convection_timevariant == "OSEEN"
        if viscosity > 0
            StokesProblem.LHSOperators[1,1][1].store_operator = true
            StokesProblem2.LHSOperators[1,1][1].store_operator = true
        end
        add_operator!(StokesProblem, [1,1], convection_classical)
        add_operator!(StokesProblem2, [1,1], convection_probust)
    end
   

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
    TCS = TimeControlSolver(StokesProblem, Solution, BackwardEuler; timedependent_equations = [1], linsolver = linsolver, nonlinear_iterations = nonlinear_iterations, maxlureuse = maxlureuse, dt_testfunction_operator = [Identity], verbosity = verbosity)
    TCS2 = TimeControlSolver(StokesProblem2, Solution2, BackwardEuler; timedependent_equations = [1], linsolver = linsolver, nonlinear_iterations = nonlinear_iterations, maxlureuse = maxlureuse, dt_testfunction_operator = [reconstruction_operator], verbosity = verbosity)

    # prepare log file
    if use_rotationform
        filename_base = "data/incompressible_gresho/area$(maxarea)_dt_$(timestep)_rotform"
    else
        filename_base = "data/incompressible_gresho/area$(maxarea)_dt_$(timestep)_convform"
    end
    mkpath(filename_base)
    filename_log = "$(filename_base)/logfile.txt"
    io = open(filename_log, "w")
    @printf(io, "LOGFILE - incompressible Gresho vortex\n")
    if uniform_grid
        @printf(io, "uniform_grid = true\n")
        @printf(io, "mrefinements = %.4e\n", nrefinements)
    else
        @printf(io, "uniform_grid = false\n")
        @printf(io, "maxarea = %.4e\n", maxarea)
    end
    @printf(io, "ndofs(velo)= %d\n", FESpaceVelocity.ndofs) 
    @printf(io, "ndofs(pressure))= %d\n", FESpacePressure.ndofs) 
    @printf(io, "rotationform = %s\n", use_rotationform ? "true" : "false")
    @printf(io, "convection_timevariant = %s\n", convection_timevariant)
    @printf(io, "mu = %.4e\n", viscosity)
    @printf(io, "dt = %.4e\n", timestep)
    @printf(io, "T = %.4e\n", T)
    @printf(io, "TIME | VELOERROR CLASSICAL | VELO-ERROR ROBUST | MAXLINRESIDUAL CLASSICAL | MAXLINRESIDUAL ROBUST\n")
    close(io)


    ## time loop
    change1 = [0.0,0.0]
    change2 = [0.0,0.0]
    maxresidual1 = 0
    maxresidual2 = 0
    maxIterations = ceil(T / timestep)
    statistics1 = [1e60 1e60;1e60 1e60]
    if maxIterations > 1
        resindex = 3
    else
        resindex = 1
    end
    for iteration = 1 : maxIterations
        try
            statistics1 = advance!(TCS, timestep)
            maxresidual1 = max(maxresidual1, sqrt(sum(statistics1[:,resindex].^2)))
        catch
            maxresidual1 = 1e60
        end
        statistics2 = advance!(TCS2, timestep)
        maxresidual2 = max(maxresidual2, sqrt(sum(statistics2[:,resindex].^2)))
        @printf("  iteration %4d of %d  time = %.4e\n",iteration,maxIterations,TCS.ctime)
        @printf("       residuals1 [v,p] = [%.4e,%.4e] evermax = %.4e\n",statistics1[1,resindex],statistics1[2,resindex],maxresidual1)
        @printf("       residuals2 [v,p] = [%.4e,%.4e] evermax = %.4e\n",statistics2[1,resindex],statistics2[2,resindex],maxresidual2)
        @printf("          change1 [v,p] = [%.4e,%.4e]\n",statistics1[1,2],statistics1[2,2])
        @printf("          change2 [v,p] = [%.4e,%.4e]\n",statistics2[1,2],statistics2[2,2])
        
    
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

        ## write to vtk
        if (iteration % write_every_nth_step == 0) || write_every_nth_step == 1 || iteration == maxIterations
            println("  log...")
            io = open(filename_log, "a")
            @printf(io, "%.4e ", TCS.ctime)
            @printf(io, "%.4e ", sqrt(evaluate(L2VelocityErrorEvaluator,Solution[1])))
            @printf(io, "%.4e ", sqrt(evaluate(L2VelocityErrorEvaluator,Solution2[1])))
            @printf(io, "%.4e ", maxresidual1)
            @printf(io, "%.4e\n", maxresidual2)
            close(io)
            if export_vtk
                println("  vtk export...")
                writeVTK!("$(filename_base)/classical_step_$(iteration).vtk", Solution)
                writeVTK!("$(filename_base)/grobust_step_$(iteration).vtk", Solution2)
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
