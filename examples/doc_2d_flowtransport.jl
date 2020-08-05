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
with some viscosity parameter ``\mu`` and diffusion parameter ``\kappa``.

The diffusion coefficient for the species is so small that the isolines of the concentration should stay parallel from inlet to outlet. Observe that a pressure-robust
    Bernardi--Raugel method preserves this much better than a classical Bernardi--Raugel method. For comparison also a Taylor--Hood method can be switched on
    which is comparable to the pressure-robust lowest-order method in this example. Also note, that the transport equation is very convection-dominated
    and no stabilisation was used here. The results are very sensitive to ``\kappa`` and may be different if a stabilisation is used (work in progress).
    Another approach would be to use a finite volume discretisation (coupling is also work in progress).
=#

push!(LOAD_PATH, "../src")
using ExtendableGrids
using Triangulate
#using VTKView
ENV["MPLBACKEND"]="qt5agg"
using PyPlot
using Printf
using GradientRobustMultiPhysics

## file that includes the mesh definition
include("../src/testgrids.jl")

function inlet_velocity!(result,x)
    result[1] = 4*x[2]*(1-x[2]);
    result[2] = 0.0;
end
function inlet_concentration!(result,x)
    result[1] = 1-x[2];
end

function beta!(result,x)
    result[1] = 1.0
    result[2] = 0.0
end

## grid generator for the bended pipe via Triangulate.jl/ExtendableGrids.jl
## generates triangles and three boundary regions (1 = boundary, 2 = outlet 3 = inlet)
function grid_pipe(maxarea::Float64)
    triin=Triangulate.TriangulateIO()
    triin.pointlist=Matrix{Cdouble}([0 0; 3 0; 3 -3; 7 -3; 7 0; 10 0; 10 1; 6 1; 6 -2; 4 -2; 4 1; 0 1]');
    triin.segmentlist=Matrix{Cint}([1 2 ; 2 3 ; 3 4 ; 4 5; 5 6; 6 7; 7 8; 8 9; 9 10; 10 11; 11 12; 12 1 ]')
    triin.segmentmarkerlist=Vector{Int32}([1,1,1,1,1,2,1,1,1,1,1,3])
    xgrid = simplexgrid("pALVa$(@sprintf("%.15f",maxarea))", triin)
    xgrid[CellRegions] = ones(Int32,num_sources(xgrid[CellNodes]))
    xgrid[CellGeometries] = VectorOfConstants(Triangle2D,num_sources(xgrid[CellNodes]))
    return xgrid
end

function main()
    #####################################################################################
    #####################################################################################

    ## initial grid
    ## replace Parallelogrm2D by Triangle2D if you like
    xgrid = grid_pipe(2e-3);

    ## problem parameters
    viscosity = 1
    diffusion = 5e-7

    ## choose one of these (inf-sup stable) finite element type pairs
    #FETypes = [H1P2{2,2}, H1P1{1}, H1P1{1}]; postprocess_operator = Identity # Taylor--Hood
    #FETypes = [H1BR{2}, L2P0{1}, H1P1{1}]; postprocess_operator = Identity # Bernardi--Raugel
    FETypes = [H1BR{2}, L2P0{1}, H1P1{1}]; postprocess_operator = ReconstructionIdentity{HDIVRT0{2}} # Bernardi--Raugel pressure-robust (RT0 reconstruction)
    #FETypes = [H1BR{2}, L2P0{1}, H1P1{1}]; postprocess_operator = ReconstructionIdentity{HDIVBDM1{2}} # Bernardi--Raugel pressure-robust (BDM1 reconstruction)
    
    ## postprocess parameters
    plot_grid = false
    plot_pressure = false
    plot_velocity = true
    plot_divergence = false
    plot_concentration = true

    #####################################################################################    
    #####################################################################################

    ## load Stokes problem prototype
    ## and assign boundary data (inlet profile in bregion 2, zero Dichlet at walls 1 and nothing at outlet region 2)
    Problem = IncompressibleNavierStokesProblem(2; viscosity = viscosity, nonlinear = false, no_pressure_constraint = true)
    add_boundarydata!(Problem, 1, [1], HomogeneousDirichletBoundary)
    add_boundarydata!(Problem, 1, [3], BestapproxDirichletBoundary; data = inlet_velocity!, bonus_quadorder = 2)

    ## add transport equation of species
    ## with boundary data (i.e. inlet concentration)
    add_unknown!(Problem, 2, 1)
    add_operator!(Problem, [3,3], LaplaceOperator(diffusion,2,1))
    add_operator!(Problem, [3,3], ConvectionOperator(1, postprocess_operator, 2, 1))
    add_boundarydata!(Problem, 3, [3], InterpolateDirichletBoundary; data = inlet_concentration!, bonus_quadorder = 0)
    Base.show(Problem)
    
    ## generate FESpaces
    FESpaceVelocity = FESpace{FETypes[1]}(xgrid)
    FESpacePressure = FESpace{FETypes[2]}(xgrid)
    FESpaceConcentration = FESpace{FETypes[3]}(xgrid)

    ## solve the problem
    ## since the flow is independent of the concentration, we first can solve the
    ## Stokes problem (equation set [1,2]) and then the transport equation ([3]) and
    ## therefore define them as an array of subiterations (but one could also solve them together)
    Solution = FEVector{Float64}("velocity",FESpaceVelocity)
    append!(Solution,"pressure",FESpacePressure)
    append!(Solution,"species concentration",FESpaceConcentration)
    solve!(Solution, Problem; subiterations = [[1,2],[3]], verbosity = 1, maxIterations = 5, maxResidual = 1e-12)

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
end

main()
