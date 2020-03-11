###############################################
### DEMONSTRATION SCRIPT STOKES POLYNOMIALS ###
###############################################
#
# bestapproximates polynomials of order (0,1,2,3) in Hdiv spaces
#
# demonstrates:
#   - convergence rates of implemented finite element methods
#   - exactness
#

using Triangulate
using Grid
using Quadrature
using FiniteElements
using FESolveCommon
using FESolveStokes
ENV["MPLBACKEND"]="tkagg"
using PyPlot

# load problem data and common grid generator
include("PROBLEMdefinitions/GRID_unitsquare.jl")
include("PROBLEMdefinitions/STOKES_2D_polynomials.jl");


function main()

    # problem modification switches
    polynomial_order = 2

    # refinement termination criterions
    maxlevel = 7
    maxdofs = 60000

    # other switches
    show_plots = false
    show_convergence_history = true


    ########################
    ### CHOOSE FEM BELOW ###
    ########################

    # Hdiv-conforming FE
    #fem = "RT0"
    #fem = "RT1"
    fem = "BDM1"

    # H1-conforming FE
    #fem = "CR"
    #fem = "MINI"
    #fem = "P2"
    #fem = "P2B"
    #fem = "BR"

    # load problem data
    PD, exact_velocity! = getProblemData(polynomial_order, 1.0, false, 1);

    L2error_velocity = zeros(Float64,maxlevel)
    ndofs = zeros(Int,maxlevel)
    grid = Nothing
    FE = Nothing
    val4dofs = Nothing
    for level = 1 : maxlevel

        println("Solving Stokes problem on refinement level...", level);
        println("Generating grid by triangle...");
        maxarea = 4.0^(-level)
        grid = gridgen_unitsquare(maxarea)
        Grid.show(grid)

        
        # load finite element
        FE = FiniteElements.string2FE(fem,grid,2,2);
        FiniteElements.show(FE)
        ndofs[level] = FiniteElements.get_ndofs(FE);


        # stop here if too many dofs
        if ndofs[level] > maxdofs 
            println("terminating (maxdofs exceeded)...");
            maxlevel = level - 1
            if (show_plots)
                maxarea = 4.0^(-maxlevel)
                grid = gridgen_unitsquare(maxarea)
                FE = FiniteElements.string2FE(fem,grid,2);
            end    
            break
        end

        # compute Hdiv best-approximation
        val4dofs = FiniteElements.createFEVector(FE);
        computeBestApproximation!(val4dofs,"L2",exact_velocity!,exact_velocity!,FE,polynomial_order + FiniteElements.get_polynomial_order(FE))

        # compute error of Hdiv best-approximation
        integral4cells = zeros(size(grid.nodes4cells,1),2);
        integrate!(integral4cells,eval_L2_interpolation_error!(exact_velocity!, val4dofs, FE), grid, 2*polynomial_order, 2);
        L2error_velocity[level] = sqrt(abs(sum(integral4cells[:])));

    end # loop over levels

    println("\n L2 error");
    show(L2error_velocity)

    #plot
    if (show_plots)
        pygui(true)
        
        # evaluate at grid points
        velo = FESolveCommon.eval_at_nodes(val4dofs,FE);
        
        PyPlot.figure(1)
        PyPlot.plot_trisurf(view(grid.coords4nodes,:,1),view(grid.coords4nodes,:,2),view(velo,:,1),cmap=get_cmap("ocean"))
        PyPlot.title("component 1")
        PyPlot.figure(2)
        PyPlot.plot_trisurf(view(grid.coords4nodes,:,1),view(grid.coords4nodes,:,2),view(velo,:,2),cmap=get_cmap("ocean"))
        PyPlot.title("component 2")
        show()
    end

    if (show_convergence_history)
        PyPlot.figure()
        PyPlot.loglog(ndofs[1:maxlevel],L2error_velocity[1:maxlevel],"-o")
        PyPlot.loglog(ndofs,ndofs.^(-1/2),"--",color = "gray")
        PyPlot.loglog(ndofs,ndofs.^(-1),"--",color = "gray")
        PyPlot.loglog(ndofs,ndofs.^(-3/2),"--",color = "gray")
        PyPlot.legend(("L2 error","O(h)","O(h^2)","O(h^3)"))
        PyPlot.title("Convergence history (fem=" * fem * ")")
        ax = PyPlot.gca()
        ax.grid(true)
    end    

        
end


main()
