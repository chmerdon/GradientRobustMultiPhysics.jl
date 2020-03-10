using Triangulate
using Grid
using Quadrature
using ExtendableSparse
using LinearAlgebra
using SparseArrays
using FiniteElements
using FESolveCommon
using FESolveStokes
using FESolveNavierStokes
using FESolvePoisson
using ForwardDiff
ENV["MPLBACKEND"]="tkagg"
using PyPlot
using Printf


function triangulate_unitsquare(maxarea, refine_barycentric = false)
    triin=Triangulate.TriangulateIO()
    triin.pointlist=Matrix{Cdouble}([-1 -1; 1 -1; 1 1; -1 1]');
    triin.segmentlist=Matrix{Cint}([1 2 ; 2 3 ; 3 4 ; 4 1 ]')
    triin.segmentmarkerlist=Vector{Int32}([1, 1, 1, 1])
    (triout, vorout)=triangulate("pALVa$(@sprintf("%.16f", maxarea))", triin)
    coords4nodes = Array{Float64,2}(triout.pointlist');
    nodes4cells = Array{Int64,2}(triout.trianglelist');
    if refine_barycentric
        coords4nodes, nodes4cells = Grid.barycentric_refinement(Grid.ElemType2DTriangle(),coords4nodes,nodes4cells)
    end    
    grid = Grid.Mesh{Float64}(coords4nodes,nodes4cells,Grid.ElemType2DTriangle());
    Grid.assign_boundaryregions!(grid,triout.segmentlist,triout.segmentmarkerlist);
    return grid
end


function main()

#fem = "CR"
#fem = "CRipm"
#fem = "CR+" # with RT0 reconstruction
#fem = "CR++" # with BDM1 reconstruction
#fem = "MINI"
#fem = "TH"
fem = "SV"
#fem = "SVipm"
#fem = "P2P0"
#fem = "P2B"
#fem = "BR"
#fem = "BR+" # with RT0 reconstruction
#fem = "BR++" # with BDM1 reconstruction

reflevel = 4
dt = 0.01
final_time = 1.0
nonlinear = true
timesteps::Int64 = floor(final_time / dt)
energy_computation_gaps = 2
u_order =
 5
nu = 1e-4
error_order = 10

solve_iterative = (fem == "SVipm" || fem == "CRipm") ? true : false
compare_with_bestapproximations = false
show_plots = true
show_convergence_history = true

function zero_data!(result,x)
    result[1] = 0.0;
    result[2] = 0.0;
end

function exact_velocity!()
    r = 0.0
    function closure(result,x)
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
end

# write data into problem description structure
PD = FESolveStokes.StokesProblemDescription()
PD.name = "Gresho vortex"
PD.viscosity = nu;
PD.time_dependent_data = false;
PD.volumedata4region = Vector{Function}(undef,1)
PD.boundarydata4bregion = Vector{Function}(undef,1)
PD.boundarytype4bregion = [1]
PD.quadorder4region = [0]
PD.volumedata4region[1] = (result,x) -> 0.0
PD.boundarydata4bregion[1] = exact_velocity!()
PD.quadorder4bregion = [0]
FESolveStokes.show(PD);

println("Solving Stokes problem on refinement level...", reflevel);
println("Generating grid by triangle...");
maxarea = 4.0^(-reflevel)
grid = triangulate_unitsquare(maxarea, (fem == "SV" || fem == "SVipm") ? true : false)
Grid.show(grid)

# load finite element
use_reconstruction = 0
if fem == "TH"
    # Taylor--Hood
    FE_velocity = FiniteElements.getP2FiniteElement(grid,2);
    FE_pressure = FiniteElements.getP1FiniteElement(grid,1);
elseif (fem == "SV") || (fem == "SVipm")
    # Scott-Vogelius
    FE_velocity = FiniteElements.getP2FiniteElement(grid,2);
    FE_pressure = FiniteElements.getP1discFiniteElement(grid,1);
elseif fem == "P2B"
    # P2-bubble
    FE_velocity = FiniteElements.getP2BFiniteElement(grid,2,2);
    FE_pressure = FiniteElements.getP1discFiniteElement(grid,1);
elseif fem == "MINI"
    # MINI
    FE_velocity = FiniteElements.getMINIFiniteElement(grid,2,2);
    FE_pressure = FiniteElements.getP1FiniteElement(grid,1);
elseif (fem == "CR") || (fem == "CRipm")
    # Crouzeix--Raviart
    FE_velocity = FiniteElements.getCRFiniteElement(grid,2,2);
    FE_pressure = FiniteElements.getP0FiniteElement(grid,1);
elseif fem == "CR+"
    # Crouzeix--Raviart
    FE_velocity = FiniteElements.getCRFiniteElement(grid,2,2);
    FE_pressure = FiniteElements.getP0FiniteElement(grid,1);
    use_reconstruction = 1
elseif fem == "CR++"
    # Crouzeix--Raviart
    FE_velocity = FiniteElements.getCRFiniteElement(grid,2,2);
    FE_pressure = FiniteElements.getP0FiniteElement(grid,1);
    use_reconstruction = 2
elseif fem == "BR"
    # Bernardi--Raugel
    FE_velocity = FiniteElements.getBRFiniteElement(grid,2);
    FE_pressure = FiniteElements.getP0FiniteElement(grid,1);
elseif fem == "BR+"
    # Bernardi--Raugel with RT0 reconstruction
    FE_velocity = FiniteElements.getBRFiniteElement(grid,2);
    FE_pressure = FiniteElements.getP0FiniteElement(grid,1);
    use_reconstruction = 1
elseif fem == "BR++"
    # Bernardi--Raugel with RT0 reconstruction
    FE_velocity = FiniteElements.getBRFiniteElement(grid,2);
    FE_pressure = FiniteElements.getP0FiniteElement(grid,1);
    use_reconstruction = 2
elseif fem == "P2P0"
    # P2P0
    FE_velocity = FiniteElements.getP2FiniteElement(grid,2);
    FE_pressure = FiniteElements.getP0FiniteElement(grid,1);
end    
FiniteElements.show(FE_velocity)
FiniteElements.show(FE_pressure)
#FiniteElements.show_dofmap(FE_velocity)
#FiniteElements.show_dofmap(FE_pressure)
ndofs_velocity = FiniteElements.get_ndofs(FE_velocity);
ndofs_pressure = FiniteElements.get_ndofs(FE_pressure);
ndofs = ndofs_velocity + ndofs_pressure;

# solve for initial value by best approximation 
val4dofs = zeros(Base.eltype(grid.coords4nodes),ndofs);
residual = FESolveStokes.computeDivFreeBestApproximation!(val4dofs,exact_velocity!(),exact_velocity!(),FE_velocity,FE_pressure,u_order)

TSS = FESolveStokes.setupTransientStokesSolver(PD,FE_velocity,FE_pressure,val4dofs,use_reconstruction)

velocity_energy = []
energy_times = []


if (show_plots)
    pygui(true)
    
    # evaluate velocity and pressure at grid points
    velo = FESolveCommon.eval_at_nodes(val4dofs,FE_velocity);
    speed = sqrt.(sum(velo.^2, dims = 2))
    #pressure = FESolveCommon.eval_at_nodes(val4dofs,FE_pressure,FiniteElements.get_ndofs(FE_velocity));

    PyPlot.figure(1)
    tcf = PyPlot.tricontourf(view(grid.coords4nodes,:,1),view(grid.coords4nodes,:,2),speed[:])
    PyPlot.axis("equal")
    PyPlot.title("Stokes Problem Solution - velocity speed")
    PyPlot.colorbar(tcf)
end    

for j = 0 : timesteps

    if mod(j,energy_computation_gaps) == 0
        println("computing errors")
        # compute errors
        integral4cells = zeros(size(grid.nodes4cells,1),1);
        integral4cells = zeros(size(grid.nodes4cells,1),2);
        integrate!(integral4cells,eval_L2_interpolation_error!(zero_data!, val4dofs[1:ndofs_velocity], FE_velocity), grid, error_order, 2);
        append!(velocity_energy,sqrt(abs(sum(integral4cells[:]))));
        append!(energy_times,TSS.current_time);
    end    

    if nonlinear == false
        FESolveStokes.PerformTimeStep(TSS,dt)
    else
        FESolveNavierStokes.PerformIMEXTimeStep(TSS,dt)
    end
    val4dofs[:] = TSS.current_solution[:]

#plot
if (show_plots)
    pygui(true)
    
    # evaluate velocity and pressure at grid points
    velo = FESolveCommon.eval_at_nodes(val4dofs,FE_velocity);
    speed = sqrt.(sum(velo.^2, dims = 2))
    #pressure = FESolveCommon.eval_at_nodes(val4dofs,FE_pressure,FiniteElements.get_ndofs(FE_velocity));

    PyPlot.figure(1)
    tcf = PyPlot.tricontourf(view(grid.coords4nodes,:,1),view(grid.coords4nodes,:,2),speed[:])
    PyPlot.axis("equal")
    PyPlot.title("Stokes Problem Solution - velocity speed")
    #PyPlot.figure(1)
    #PyPlot.plot_trisurf(view(grid.coords4nodes,:,1),view(grid.coords4nodes,:,2),view(velo,:,1),cmap=get_cmap("ocean"))
    #PyPlot.title("Stokes Problem Solution - velocity component 1")
    #PyPlot.figure(2)
    #PyPlot.plot_trisurf(view(grid.coords4nodes,:,1),view(grid.coords4nodes,:,2),view(velo,:,2),cmap=get_cmap("ocean"))
    #PyPlot.title("Stokes Problem Solution - velocity component 2")
    #PyPlot.figure(3)
    #PyPlot.plot_trisurf(view(grid.coords4nodes,:,1),view(grid.coords4nodes,:,2),pressure[:],cmap=get_cmap("ocean"))
    #PyPlot.title("Stokes Problem Solution - pressure")
    show()
end    
end    

Base.show(velocity_energy)

if (show_convergence_history)
    PyPlot.figure()
    PyPlot.loglog(energy_times,velocity_energy,"-o")
    PyPlot.legend(("Energy"))
    PyPlot.title("Convergence history (fem=" * fem * ")")
    ax = PyPlot.gca()
    ax.grid(true)
end    

    
end


main()
