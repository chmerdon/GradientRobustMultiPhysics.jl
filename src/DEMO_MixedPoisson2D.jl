#######################################
### DEMONSTRATION SCRIPT POISSON 2D ###
#######################################
#
# solves 2D polynomial Poisson test problems on L-shaped domain
#
# demonstrates:
#   - convergence rates of implemented finite element methods
#   - comparison of Poisson solution and L2 bestapproximation


using Grid
using Triangulate
using Quadrature
using FiniteElements
using FESolveCommon
using FESolvePoisson
using FEEstimate
ENV["MPLBACKEND"]="tkagg"
using PyPlot


# load problem data and common grid generator
include("PROBLEMdefinitions/GRID_unitsquare.jl")
include("PROBLEMdefinitions/POISSON_2D_p4bubble.jl");

function main()

    # grid and refinement termination criterions
    gridgenerator = gridgen_unitsquare
    maxlevel = 15
    maxdofs = 10000

    # other switches
    show_plots = true
    show_convergence_history = true

    ########################
    ### CHOOSE FEM BELOW ###
    ########################

    #fem_stress = "RT0"; fem_divergence = "P0"
    #fem_stress = "RT1"; fem_divergence = "P1dc"
    #fem_stress = "BDM1"; fem_divergence = "P0"
    #fem_stress = "P1"; fem_divergence = "P0"
    fem_stress = "P2"; fem_divergence = "P1dc"

    diffusion = 1.0 # scalar constant diffusion

    # load problem data
    PD, exact_solution!, exact_gradient! = getProblemData(diffusion);
    FESolvePoisson.show(PD);

    L2error = zeros(Float64,maxlevel)
    L2errorBA = zeros(Float64,maxlevel)
    Estimator = zeros(Float64,maxlevel)
    estimator4cells = Nothing
    ndofs = zeros(Int64,maxlevel)            
    val4dofs = Nothing          
    val4dofsBA = Nothing
    FE_stress = Nothing
    FE_divergence = Nothing
    grid = Nothing
    for level = 1 : maxlevel

        # generate grid
        maxarea = 4.0^(-level)
        grid = gridgenerator(maxarea)
        Grid.show(grid)

        # generate FE
        FE_stress = FiniteElements.string2FE(fem_stress, grid, 2, 2)
        FE_divergence = FiniteElements.string2FE(fem_divergence, grid, 2, 1)
        ensure_nodes4faces!(grid);
        ensure_volume4cells!(grid);
        FiniteElements.show(FE_stress)
        FiniteElements.show(FE_divergence)
        ndofs[level] = FiniteElements.get_ndofs(FE_stress) + FiniteElements.get_ndofs(FE_divergence);

        # stop here if too many dofs
        if ndofs[level] > maxdofs 
            println("terminating (maxdofs exceeded)...");
            maxlevel = level - 1
            if show_plots
                maxarea = 4.0^(-maxlevel)
                grid = gridgenerator(maxarea)
                FE_stress = FiniteElements.string2FE(fem_stress, grid, 2, 2)
                FE_divergence = FiniteElements.string2FE(fem_divergence, grid, 2, 1)
            end    
            break
        end

        println("Solving mixed formulation of Poisson problem...");    
        val4dofs = zeros(Float64,ndofs[level]);
        residual = solveMixedPoissonProblem!(val4dofs,PD,FE_stress,FE_divergence);




        # compute velocity best approximation
        println("Solving L2 bestapproximation problem for stress...");    
        val4dofsBA = FiniteElements.createFEVector(FE_stress);
        residual = computeBestApproximation!(val4dofsBA,"L2",exact_gradient!,exact_gradient!,FE_stress, 3)

        # compute errors
        integral4cells = zeros(size(grid.nodes4cells,1),2);
        integrate!(integral4cells,eval_L2_interpolation_error!(exact_gradient!, val4dofs, FE_stress), grid, 6, 2);
        L2error[level] = sqrt(abs(sum(integral4cells)));
        println("L2_error = " * string(L2error[level]));
        integrate!(integral4cells,eval_L2_interpolation_error!(exact_gradient!, val4dofsBA, FE_stress), grid, 6, 2);
        L2errorBA[level] = sqrt(abs(sum(integral4cells)));
        println("L2_error_BA = " * string(L2errorBA[level]));


    end

    # plot
    if (show_plots)
        nodevals = FESolveCommon.eval_at_nodes(val4dofs[1:length(val4dofsBA)],FE_stress);
        pygui(true)
        PyPlot.figure(1)
        PyPlot.plot_trisurf(view(grid.coords4nodes,:,1),view(grid.coords4nodes,:,2),nodevals[:,1],cmap=get_cmap("ocean"))
        PyPlot.title("stress component 1")
        PyPlot.figure(2)
        PyPlot.plot_trisurf(view(grid.coords4nodes,:,1),view(grid.coords4nodes,:,2),nodevals[:,2],cmap=get_cmap("ocean"))
        PyPlot.title("stress component 2")
        #show()
    end    

    if (show_convergence_history)
        PyPlot.figure()
        PyPlot.loglog(ndofs[1:maxlevel],L2error[1:maxlevel],"-o")
        PyPlot.loglog(ndofs[1:maxlevel],L2errorBA[1:maxlevel],"-o")
        PyPlot.loglog(ndofs,ndofs.^(-1/2),"--",color = "gray")
        PyPlot.loglog(ndofs,ndofs.^(-1),"--",color = "gray")
        PyPlot.loglog(ndofs,ndofs.^(-3/2),"--",color = "gray")
        PyPlot.legend(("L2 error","L2 error BA","O(h)","O(h^2)","O(h^3)"))
        PyPlot.title("Convergence history (fem=" * fem_stress * "/" * fem_divergence * ")")
        ax = PyPlot.gca()
        ax.grid(true)
    end 


end


main()
