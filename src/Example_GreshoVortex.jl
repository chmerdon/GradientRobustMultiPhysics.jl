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
    triin.pointlist=Matrix{Cdouble}([-1 -1; 1 -1; 1 1; -1 1]');
    triin.segmentlist=Matrix{Cint}([1 2 ; 2 3 ; 3 4 ; 4 1 ]')
    triin.segmentmarkerlist=Vector{Int32}([1, 1, 1, 1])
    (triout, vorout)=triangulate("pALVa$(@sprintf("%.16f", maxarea))", triin)
    coords4nodes = Array{Float64,2}(triout.pointlist');
    nodes4cells = Array{Int64,2}(triout.trianglelist');
    if refine_barycentric
        coords4nodes, nodes4cells = Grid.barycentric_refinement(Grid.ElemType2DTriangle(),coords4nodes,nodes4cells)
    end    
    return Grid.Mesh{Float64}(coords4nodes,nodes4cells,Grid.ElemType2DTriangle());
end


function main()

#fem = "CR"
#fem = "CRipm"
#fem = "CR+" # with reconstruction
#fem = "MINI"
#fem = "TH"
fem = "SV"
#fem = "SVipm"
#fem = "P2P0"
#fem = "P2B"
#fem = "BR"
#fem = "BR+" # with reconstruction

reflevel = 4
u_order = 5
nu = 1
error_order = 10

solve_iterative = (fem == "SVipm" || fem == "CRipm") ? true : false
compare_with_bestapproximations = false
show_plots = true
show_convergence_history = true



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
PD.time_dependent = true;
PD.initial_time = 0.0
PD.final_time = 3.0
PD.initial_velocity = exact_velocity!()
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
elseif fem == "BR"
    # Bernardi--Raugel
    FE_velocity = FiniteElements.getBRFiniteElement(grid,2);
    FE_pressure = FiniteElements.getP0FiniteElement(grid,1);
elseif fem == "BR+"
    # Bernardi--Raugel with RT0 reconstruction
    FE_velocity = FiniteElements.getBRFiniteElement(grid,2);
    FE_pressure = FiniteElements.getP0FiniteElement(grid,1);
    use_reconstruction = 1
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


# solve Stokes problem
#if solve_iterative
#    residual = solveStokesProblem_iterative!(val4dofs,PD,FE_velocity,FE_pressure, use_reconstruction);
#else
#    residual = solveStokesProblem!(val4dofs,PD,FE_velocity,FE_pressure, use_reconstruction);
#end

# check divergence
B = ExtendableSparseMatrix{Float64,Int64}(ndofs_velocity,ndofs_velocity)
FESolveCommon.assemble_operator!(B,FESolveCommon.CELL_DIVUdotDIVV,FE_velocity);
divergence = sqrt(abs(dot(val4dofs[1:ndofs_velocity],B*val4dofs[1:ndofs_velocity])));
println("divergence = ",divergence);

# compute errors
integral4cells = zeros(size(grid.nodes4cells,1),1);
integral4cells = zeros(size(grid.nodes4cells,1),2);
integrate!(integral4cells,eval_L2_interpolation_error!(exact_velocity!(), val4dofs[1:ndofs_velocity], FE_velocity), grid, error_order, 2);
L2error_velocity = sqrt(abs(sum(integral4cells[:])));
println("L2_velocity_error_initial = " * string(L2error_velocity));

#plot
if (show_plots)
    pygui(true)
    
    # evaluate velocity and pressure at grid points
    velo = FESolveCommon.eval_at_nodes(val4dofs,FE_velocity);
    speed = sqrt.(sum(velo.^2, dims = 2))
    #pressure = FESolveCommon.eval_at_nodes(val4dofs,FE_pressure,FiniteElements.get_ndofs(FE_velocity));

    PyPlot.figure(1)
    PyPlot.tricontourf(view(grid.coords4nodes,:,1),view(grid.coords4nodes,:,2),speed[:])
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


# if (show_convergence_history)
#     PyPlot.figure()
#     PyPlot.loglog(ndofs,L2error_velocity,"-o")
#     #PyPlot.loglog(ndofs,L2error_divergence,"-o")
#     PyPlot.loglog(ndofs,L2error_pressure,"-o")
#     if compare_with_bestapproximations == true
#         PyPlot.loglog(ndofs,L2error_velocityBA,"-o")
#         PyPlot.loglog(ndofs,L2error_velocityVL,"-o")
#         PyPlot.loglog(ndofs,L2error_pressureBA,"-o")
#     end    
#     PyPlot.loglog(ndofs,ndofs.^(-1/2),"--",color = "gray")
#     PyPlot.loglog(ndofs,ndofs.^(-1),"--",color = "gray")
#     PyPlot.loglog(ndofs,ndofs.^(-3/2),"--",color = "gray")
#     if compare_with_bestapproximations == true
#         PyPlot.legend(("L2 error velocity","L2 error divergence","L2 error pressure","L2 error velocity BA","L2 error velocity Poisson","L2 error pressure BA","O(h)","O(h^2)","O(h^3)"))
#     else
#         PyPlot.legend(("L2 error velocity","L2 error divergence","L2 error pressure","O(h)","O(h^2)","O(h^3)"))
#     end    
#     PyPlot.title("Convergence history (fem=" * fem * " problem=" * use_problem * ")")
#     ax = PyPlot.gca()
#     ax.grid(true)
# end    

    
end


main()
