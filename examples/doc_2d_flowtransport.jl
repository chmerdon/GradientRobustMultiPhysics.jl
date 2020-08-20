#= 

# 2D Flow + Transport
([source code](SOURCE_URL))

This example solve the Stokes problem in a Omega-shaped pipe and then uses the velocity in a transport equation for a species with a certain inlet concentration.
Altogether, we are looking for a velocity ``\mathbf{u}``, a pressure ``\mathbf{p}`` and a species concentration ``\mathbf{c}`` such that
```math
\begin{aligned}
- \mu \Delta \mathbf{u} + \nabla p & = 0\\
\mathrm{div}(u) & = 0\\
- \kappa \Delta \mathbf{c} + \mathbf{u} \cdot \nabla \mathbf{c} & = 0
\end{aligned}
```
with some viscosity parameter  and diffusion parameter ``\kappa``.

The diffusion coefficient for the species is chosen (almost) zero such that the isolines of the concentration should stay parallel from inlet to outlet. 
For the discretisation of the convection term in the transport equation two three possibilities can be chosen:

1. Classical finite element discretisations ``\mathbf{u}_h \cdot \nabla \mathbf{c}_h``
2. Pressure-robust finite element discretisation ``\Pi_\text{reconst} \mathbf{u}_h \cdot \nabla \mathbf{c}_h`` with some divergence-free reconstruction operator ``\Pi_\text{reconst}``
3. Upwind finite volume discretisation for ``\kappa = 0`` based on normal fluxes along the faces (also divergence-free in finite volume sense)

Observe that a pressure-robust Bernardi--Raugel discretisation preserves this much better than a classical Bernardi--Raugel method. For comparison also a Taylor--Hood method can be switched on
which is comparable to the pressure-robust lowest-order method in this example. 

Note, that the transport equation is very convection-dominated and no stabilisation in the finite element discretisations was used here (but instead a nonzero ``\kappa``). The results are very sensitive to ``\kappa`` and may be different if a stabilisation is used (work in progress).
Also note, that only the finite volume discretisation perfectly obeys the maximum principle for the concentration but the isolines do no stay parallel until the outlet is reached, possibly due to articifial diffusion.
=#

push!(LOAD_PATH, "../src")
using GradientRobustMultiPhysics
using ExtendableGrids
using Triangulate
ENV["MPLBACKEND"]="qt5agg"
using PyPlot
using Printf


## boundary data
function inlet_velocity!(result,x)
    result[1] = 4*x[2]*(1-x[2]);
    result[2] = 0.0;
end
function inlet_concentration!(result,x)
    result[1] = 1-x[2];
end

## grid generator for the bended pipe via Triangulate.jl/ExtendableGrids.jl
## generates triangles and three boundary regions (1 = boundary, 2 = outlet 3 = inlet)
function grid_pipe(maxarea::Float64)
    triin=Triangulate.TriangulateIO()
    triin.pointlist=Matrix{Cdouble}([0 0; 3 0; 3 -3; 7 -3; 7 0; 10 0; 10 1; 6 1; 6 -2; 4 -2; 4 1; 0 1]');
    triin.segmentlist=Matrix{Cint}([1 2 ; 2 3 ; 3 4 ; 4 5; 5 6; 6 7; 7 8; 8 9; 9 10; 10 11; 11 12; 12 1 ]')
    triin.segmentmarkerlist=Vector{Int32}([1,1,1,1,1,2,1,1,1,1,1,4])
    xgrid = simplexgrid("pALVa$(@sprintf("%.15f",maxarea))", triin)
    xgrid[CellRegions] = ones(Int32,num_sources(xgrid[CellNodes]))
    xgrid[CellGeometries] = VectorOfConstants(Triangle2D,num_sources(xgrid[CellNodes]))
    return xgrid
end

## everything is wrapped in a main function
function main()
    #####################################################################################
    #####################################################################################

    ## initial grid
    ## replace Parallelogrm2D by Triangle2D if you like
    xgrid = grid_pipe(1e-3);

    ## problem parameters
    viscosity = 1
    diffusion_FE = 1e-7 # diffusion coefficient for transport equation

    ## choose one of these (inf-sup stable) finite element type pairs for the flow
    #FETypes = [H1P2{2,2}, H1P1{1}]; postprocess_operator = Identity # Taylor--Hood
    #FETypes = [H1BR{2}, L2P0{1}]; postprocess_operator = Identity # Bernardi--Raugel
    FETypes = [H1BR{2}, L2P0{1}]; postprocess_operator = ReconstructionIdentity{HDIVRT0{2}} # Bernardi--Raugel pressure-robust (RT0 reconstruction)
    #FETypes = [H1BR{2}, L2P0{1}]; postprocess_operator = ReconstructionIdentity{HDIVBDM1{2}} # Bernardi--Raugel pressure-robust (BDM1 reconstruction)
    
    ## choose discretisation for the transport equation
    #FETypeTransport = H1P1{1}; FVtransport = false
    FETypeTransport = L2P0{1}; FVtransport = true # note: ingores postprocess_operator, since FV operator does the same on triangles

    ## postprocess parameters
    plot_grid = false
    plot_pressure = false
    plot_velocity = true
    plot_divergence = false
    plot_concentration = true
    write_vtk = true

    #####################################################################################    
    #####################################################################################

    ## load Stokes problem prototype
    ## and assign boundary data (inlet profile in bregion 2, zero Dirichlet at walls 1 and nothing at outlet region 2)
    Problem = IncompressibleNavierStokesProblem(2; viscosity = viscosity, nonlinear = false, no_pressure_constraint = true)
    Problem.name = "Stokes + Transport"
    add_boundarydata!(Problem, 1, [1,3], HomogeneousDirichletBoundary)
    add_boundarydata!(Problem, 1, [4], BestapproxDirichletBoundary; data = inlet_velocity!, bonus_quadorder = 2)

    ## add transport equation of species
    add_unknown!(Problem, 2, 1; unknown_name = "concentration", equation_name = "transport equation")
    if FVtransport == true
        ## finite volume upwind discretisation
        add_operator!(Problem, [3,3], FVConvectionDiffusionOperator(1))
    else
        ## finite element convection and diffusion (very small) operators
        add_operator!(Problem, [3,3], LaplaceOperator(diffusion_FE,2,1))
        add_operator!(Problem, [3,3], ConvectionOperator(1, postprocess_operator, 2, 1))
    end
    ## with boundary data (i.e. inlet concentration)
    add_boundarydata!(Problem, 3, [4], InterpolateDirichletBoundary; data = inlet_concentration!, bonus_quadorder = 0)
    Base.show(Problem)
    
    ## generate FESpaces
    FESpaceVelocity = FESpace{FETypes[1]}(xgrid)
    FESpacePressure = FESpace{FETypes[2]}(xgrid)
    FESpaceConcentration = FESpace{FETypeTransport}(xgrid)

    ## solve the decoupled flow problem
    Solution = FEVector{Float64}("velocity",FESpaceVelocity)
    append!(Solution,"pressure",FESpacePressure)
    append!(Solution,"species concentration",FESpaceConcentration)
    solve!(Solution, Problem; subiterations = [[1,2]], verbosity = 1, maxIterations = 5, maxResidual = 1e-12)

    ## solve the transport by finite volumes or finite elements
    if FVtransport == true
        ## pseudo-timestepping until stationarity detected
        TCS = TimeControlSolver(Problem, Solution, BackwardEuler; subiterations = [[3]], timedependent_equations = [3], verbosity = 1)
        change = 0.0
        timestep = 1000
        maxResidual = 1e-10
        for iteration = 1 : 100
            ## in the advance! step we can tell that the matrices of which subiterations do not change
            ## and so allow for reuse of the lu factorization
            change = advance!(TCS, timestep; reuse_matrix = [true])
            @printf("  iteration %4d",iteration)
            @printf("  time = %.4e",TCS.ctime)
            @printf("  change = %.4e \n",change)
            if change < maxResidual
                println("  terminated (below tolerance)")
                break;
            end
        end
    else
        ## solve directly
        solve!(Solution, Problem; subiterations = [[3]], verbosity = 1, maxIterations = 5, maxResidual = 1e-12)
    end



    ## print minimal and maximal concentration
    ## (maximum principle says it should be [0,1])
    println("[min(c),max(c)] = [$(minimum(Solution[3][:])),$(maximum(Solution[3][:]))]")

    ## plots
    nnodes = size(xgrid[Coordinates],2)
    if plot_grid
        PyPlot.figure("grid")
        ExtendableGrids.plot(xgrid, Plotter = PyPlot)
    end
    if plot_pressure
        nodevals = zeros(Float64,1,nnodes)
        PyPlot.figure("pressure")
        nodevalues!(nodevals,Solution[2],FESpacePressure)
        ExtendableGrids.plot(xgrid, nodevals[1,:]; Plotter = PyPlot)
    end
    if plot_velocity
        xCoordinates = xgrid[Coordinates]
        nodevals = zeros(Float64,2,nnodes)
        nodevalues!(nodevals,Solution[1],FESpaceVelocity)
        PyPlot.figure("velocity")
        ExtendableGrids.plot(xgrid, sqrt.(nodevals[1,:].^2+nodevals[2,:].^2); Plotter = PyPlot, isolines = 3)
        quiver(xCoordinates[1,:],xCoordinates[2,:],nodevals[1,:],nodevals[2,:])
    end
    if plot_divergence
        nodevals = zeros(Float64,2,nnodes)
        PyPlot.figure("divergence")
        nodevalues!(nodevals,Solution[1],FESpaceVelocity,Divergence)
        ExtendableGrids.plot(xgrid, nodevals[1,:]; Plotter = PyPlot, cmap = "cool")
    end
    if plot_concentration
        nodevals = zeros(Float64,1,nnodes)
        PyPlot.figure("species concentration")
        nodevalues!(nodevals,Solution[3],FESpaceConcentration)
        ExtendableGrids.plot(xgrid, nodevals[1,:]; Plotter = PyPlot, cmap = "cool")
    end

    if write_vtk
        writeVTK!("data_flowtransport.vtk", Solution)
    end
end

main()
