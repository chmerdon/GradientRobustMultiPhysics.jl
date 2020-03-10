using Triangulate
using Grid
using Quadrature
using ExtendableSparse
using LinearAlgebra
using SparseArrays
using FiniteElements
using FESolveCommon
using FESolveStokes
using FESolvePoisson
using ForwardDiff
ENV["MPLBACKEND"]="tkagg"
using PyPlot
using Printf


function triangulate_unitsquare(maxarea, refine_barycentric = false)
    triin=Triangulate.TriangulateIO()
    triin.pointlist=Matrix{Cdouble}([0 0; 1 0; 1 1; 0 1]');
    triin.segmentlist=Matrix{Cint}([1 2 ; 2 3 ; 3 4 ; 4 1 ]')
    triin.segmentmarkerlist=Vector{Int32}([1, 2, 3, 4])
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
use_reconstruction = 0
barycentric_refinement = false

#fem_velocity = "CR"; fem_pressure = "P0"
#fem_velocity = "CR"; fem_pressure = "P0"; use_reconstruction = 1
#fem_velocity = "MINI"; fem_pressure = "P1"
fem_velocity = "P2";  fem_pressure = "P1"
#fem_velocity = "P2";  fem_pressure = "P1dc"; barycentric_refinement = true
#fem_velocity = "P2"; fem_pressure = "P0"
#fem_velocity = "P2B"; fem_pressure = "P1dc"
#fem_velocity = "BR"; fem_pressure = "P0"
#fem_velocity = "BR"; fem_pressure = "P0"; use_reconstruction = 1

do_nothing_inlet = true
symmetry_top = true
maxlevel = 5
nu = 1

solve_iterative = false
show_plots = true
show_convergence_history = true


function exact_pressure!(result,x)
    result[1] = -2*x[1];
    if do_nothing_inlet == false
        result[1] += 1.0;
    end    
end


function zero_data!(result,x)
    result[1] = 0.0;
    result[2] = 0.0;
end

function ones_data!(result,x)
    result[1] = 1.0;
    result[2] = 0.0;
end


function exact_velocity!(result,x)
    result[1] = x[2]*(2.0-x[2]);
    result[2] = 0.0;
end

# write data into problem description structure
PD = FESolveStokes.StokesProblemDescription()
PD.name = "Hagen-Poisseuille"
PD.viscosity = nu;
PD.volumedata4region = Vector{Function}(undef,1)
PD.boundarydata4bregion = Vector{Function}(undef,4)
PD.boundarytype4bregion = ones(length(PD.boundarydata4bregion))
PD.quadorder4bregion = zeros(length(PD.boundarydata4bregion))
PD.quadorder4region = [0]

# bottom boundary is constant zero
PD.volumedata4region[1] = zero_data!
PD.boundarydata4bregion[1] = zero_data!

# outlet/right boundary is Dirichlet
PD.boundarydata4bregion[2] = exact_velocity!
PD.quadorder4bregion[2] = 2

# top-bottom is constant one (later symmetric boundary)
if symmetry_top
    PD.boundarydata4bregion[3] = zero_data!
    PD.boundarytype4bregion[3] = 3
else
    PD.boundarydata4bregion[3] = ones_data!
end    

# inlet/left boundary is do-nothing
if do_nothing_inlet == true
    PD.boundarydata4bregion[4] = zero_data!
    PD.boundarytype4bregion[4] = 2
else
    PD.boundarydata4bregion[4] = exact_velocity!
    PD.boundarytype4bregion[4] = 1
    PD.quadorder4bregion[4] = 2
end    
FESolveStokes.show(PD);


function wrap_pressure(result,x)
    result[1] = exact_pressure(use_problem)(x)
end    

L2error_velocity = zeros(Float64,maxlevel)
L2error_divergence = zeros(Float64,maxlevel)
L2error_pressure = zeros(Float64,maxlevel)
ndofs = zeros(Int,maxlevel)

for level = 1 : maxlevel

println("Solving Stokes problem on refinement level...", level);
println("Generating grid by triangle...");
maxarea = 4.0^(-level)
grid = triangulate_unitsquare(maxarea, barycentric_refinement)
Grid.show(grid)

# load finite element
FE_velocity = FiniteElements.string2FE(fem_velocity,grid,2,2)
FE_pressure = FiniteElements.string2FE(fem_pressure,grid,2,1)
FiniteElements.show(FE_velocity)
FiniteElements.show(FE_pressure)
#FiniteElements.show_dofmap(FE_velocity)
#FiniteElements.show_dofmap(FE_pressure)
ndofs_velocity = FiniteElements.get_ndofs(FE_velocity);
ndofs_pressure = FiniteElements.get_ndofs(FE_pressure);
ndofs[level] = ndofs_velocity + ndofs_pressure;

# solve Stokes problem
val4dofs = zeros(Base.eltype(grid.coords4nodes),ndofs[level]);
if solve_iterative
    residual = solveStokesProblem_iterative!(val4dofs,PD,FE_velocity,FE_pressure, use_reconstruction);
else
    residual = solveStokesProblem!(val4dofs,PD,FE_velocity,FE_pressure, use_reconstruction);
end
# check divergence
B = ExtendableSparseMatrix{Float64,Int64}(ndofs_velocity,ndofs_velocity)
FESolveCommon.assemble_operator!(B,FESolveCommon.CELL_DIVUdotDIVV,FE_velocity);
divergence = sqrt(abs(dot(val4dofs[1:ndofs_velocity],B*val4dofs[1:ndofs_velocity])));
println("divergence = ",divergence);
L2error_divergence[level] = divergence;

# compute errors
integral4cells = zeros(size(grid.nodes4cells,1),1);
integrate!(integral4cells,eval_L2_interpolation_error!(exact_pressure!, val4dofs[ndofs_velocity+1:end], FE_pressure), grid, 2, 1);
L2error_pressure[level] = sqrt(abs(sum(integral4cells)));
#println("L2_pressure_error_STOKES = " * string(L2error_pressure[level]));
integral4cells = zeros(size(grid.nodes4cells,1),2);
integrate!(integral4cells,eval_L2_interpolation_error!(exact_velocity!, val4dofs[1:ndofs_velocity], FE_velocity), grid, 4, 2);
L2error_velocity[level] = sqrt(abs(sum(integral4cells[:])));
#println("L2_velocity_error_STOKES = " * string(L2error_velocity[level]));

#plot
if (show_plots) && (level == maxlevel) && ndofs[level] < 7500
    pygui(true)
    
    # evaluate velocity and pressure at grid points
    velo = FESolveCommon.eval_at_nodes(val4dofs,FE_velocity);
    pressure = FESolveCommon.eval_at_nodes(val4dofs,FE_pressure,FiniteElements.get_ndofs(FE_velocity));

    PyPlot.figure(1)
    PyPlot.plot_trisurf(view(grid.coords4nodes,:,1),view(grid.coords4nodes,:,2),view(velo,:,1),cmap=get_cmap("ocean"))
    PyPlot.title("Stokes Problem Solution - velocity component 1")
    PyPlot.figure(2)
    PyPlot.plot_trisurf(view(grid.coords4nodes,:,1),view(grid.coords4nodes,:,2),view(velo,:,2),cmap=get_cmap("ocean"))
    PyPlot.title("Stokes Problem Solution - velocity component 2")
    PyPlot.figure(3)
    PyPlot.plot_trisurf(view(grid.coords4nodes,:,1),view(grid.coords4nodes,:,2),pressure[:],cmap=get_cmap("ocean"))
    PyPlot.title("Stokes Problem Solution - pressure")
    show()
end
end # loop over levels

println("\n L2 pressure error");
show(L2error_pressure)
println("\n L2 velocity error");
show(L2error_velocity)
println("\n L2 velocity divergence error");
show(L2error_divergence)

if (show_convergence_history)
    PyPlot.figure()
    PyPlot.loglog(ndofs,L2error_velocity,"-o")
    PyPlot.loglog(ndofs,L2error_divergence,"-o")
    PyPlot.loglog(ndofs,L2error_pressure,"-o")
    PyPlot.loglog(ndofs,ndofs.^(-1/2),"--",color = "gray")
    PyPlot.loglog(ndofs,ndofs.^(-1),"--",color = "gray")
    PyPlot.loglog(ndofs,ndofs.^(-3/2),"--",color = "gray")
    PyPlot.legend(("L2 error velocity","L2 error divergence","L2 error pressure","O(h)","O(h^2)","O(h^3)"))   
    PyPlot.title("Convergence history (fem=" * fem_velocity * "/" * fem_pressure * ")")
    ax = PyPlot.gca()
    ax.grid(true)
end    

    
end


main()
