#############################################
### DEMONSTRATION SCRIPT HAGEN-POISEUILLE ###
#############################################
#
# solves Hagen-Poiseuille test problem
#
# demonstrates:
#   - convergence rates of implemented finite element methods
#     (2nd order finite elements methods should solve it exactly)
#   - multiple boundary conditions (Dirichlet, do-nothing, symmetry boundary)
#

using Triangulate
using Grid
using Quadrature
using LinearAlgebra
using FiniteElements
using FESolveCommon
using FESolveStokes
ENV["MPLBACKEND"]="tkagg"
using PyPlot

# load problem data and common grid generator
include("PROBLEMdefinitions/GRID_unitsquare.jl")
include("PROBLEMdefinitions/STOKES_HagenPoiseuille.jl");


function main()

    # problem modification switches
    do_nothing_inlet = true
    symmetry_top = true
    nu = 1

    # refinement termination criterions
    maxlevel = 7
    maxdofs = 60000

    # other switches
    show_plots = true
    show_convergence_history = true
    use_reconstruction = 0 # do not change here
    barycentric_refinement = false # do not change here


    ########################
    ### CHOOSE FEM BELOW ###
    ########################

    #fem_velocity = "CR"; fem_pressure = "P0"
    fem_velocity = "MINI"; fem_pressure = "P1"
    #fem_velocity = "P2";  fem_pressure = "P1"
    #fem_velocity = "P2";  fem_pressure = "P1dc"; barycentric_refinement = true
    #fem_velocity = "P2"; fem_pressure = "P0"
    #fem_velocity = "P2B"; fem_pressure = "P1dc"
    #fem_velocity = "BR"; fem_pressure = "P0"


    # load problem data
    PD = getProblemData(4);
    PD.viscosity = nu;

    # top-bottom is constant one (later symmetric boundary)
    if symmetry_top
        PD.boundarydata4bregion[3] = (result,x) -> 0.0
        PD.boundarytype4bregion[3] = 3
    end    

    # inlet/left boundary is do-nothing
    if do_nothing_inlet == true
        PD.boundarydata4bregion[4] = (result,x) -> 0.0
        PD.boundarytype4bregion[4] = 2
    end    

    FESolveStokes.show(PD);

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
        ndofs_velocity = FiniteElements.get_ndofs(FE_velocity);
        ndofs_pressure = FiniteElements.get_ndofs(FE_pressure);
        ndofs[level] = ndofs_velocity + ndofs_pressure;

        # stop here if too many dofs
        if ndofs[level] > maxdofs 
            println("terminating (maxdofs exceeded)...");
            maxlevel = level - 1
            break
        end

        # solve Stokes problem
        val4dofs = zeros(Base.eltype(grid.coords4nodes),ndofs[level]);
        residual = solveStokesProblem!(val4dofs,PD,FE_velocity,FE_pressure, use_reconstruction);

        # compute errors
        integral4cells = zeros(size(grid.nodes4cells,1),1);
        integrate!(integral4cells,eval_L2_interpolation_error!(exact_pressure!(nu, do_nothing_inlet ? 0.0 : -1.0), val4dofs[ndofs_velocity+1:end], FE_pressure), grid, 2, 1);
        L2error_pressure[level] = sqrt(abs(sum(integral4cells)));
        integral4cells = zeros(size(grid.nodes4cells,1),2);
        integrate!(integral4cells,eval_L2_interpolation_error!(exact_velocity!, val4dofs[1:ndofs_velocity], FE_velocity), grid, 4, 2);
        L2error_velocity[level] = sqrt(abs(sum(integral4cells[:])));

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
        PyPlot.loglog(ndofs[1:maxlevel],L2error_velocity[1:maxlevel],"-o")
        PyPlot.loglog(ndofs[1:maxlevel],L2error_pressure[1:maxlevel],"-o")
        PyPlot.loglog(ndofs,ndofs.^(-1/2),"--",color = "gray")
        PyPlot.loglog(ndofs,ndofs.^(-1),"--",color = "gray")
        PyPlot.loglog(ndofs,ndofs.^(-3/2),"--",color = "gray")
        PyPlot.legend(("L2 error velocity","L2 error pressure","O(h)","O(h^2)","O(h^3)"))   
        PyPlot.title("Convergence history (fem=" * fem_velocity * "/" * fem_pressure * ")")
        ax = PyPlot.gca()
        ax.grid(true)
    end    

        
end


main()
