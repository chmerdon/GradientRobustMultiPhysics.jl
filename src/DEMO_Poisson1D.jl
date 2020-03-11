#######################################
### DEMONSTRATION SCRIPT POISSON 1D ###
#######################################
#
# solves 1D polynomial Poisson test problems of arbitrary order
#
# demonstrates:
#   - convergence rates of implemented finite element methods
#   - comparison of Poisson solution and L2 bestapproximation


using Grid
using Quadrature
using FiniteElements
using FESolveCommon
using FESolvePoisson
ENV["MPLBACKEND"]="tkagg"
using PyPlot


# load problem data and common grid generator
include("PROBLEMdefinitions/GRID_unitinterval.jl")
include("PROBLEMdefinitions/POISSON_1D_polynomials.jl");

function main()

    # refinement termination criterions
    maxlevel = 15
    maxdofs = 20000

    # other switches
    show_plots = true
    show_convergence_history = true

    ########################
    ### CHOOSE FEM BELOW ###
    ########################

    fem = "P1"
    #fem = "P2"

    # choose coefficients of exact solution

    polynomial_coefficients = [0, -3, 2, -1, 1] # quartic
    #polynomial_coefficients = [0, 0, 0, -1]  # cubic
    #polynomial_coefficients = [0, 0, -1]  # quadratic
    #polynomial_coefficients = [0, 1]   # linear
    #polynomial_coefficients = [1, 0]   # constant
    
    # load problem data
    PD, exact_solution! = getProblemData(polynomial_coefficients);
    FESolvePoisson.show(PD);

    L2error = zeros(Float64,maxlevel)
    L2errorBA = zeros(Float64,maxlevel)
    ndofs = zeros(Int64,maxlevel)            
    val4dofs = Nothing
    FE = Nothing
    grid = Nothing
    for level = 1 : maxlevel

        # generate grid
        maxarea = 2.0^(-level)
        grid = gridgen_unitinterval(maxarea)
        Grid.show(grid)

        # generate FE
        FE = FiniteElements.string2FE(fem, grid, 1, 1)
        FiniteElements.show(FE)
        ndofs[level] = FiniteElements.get_ndofs(FE);

        # stop here if too many dofs
        if ndofs[level] > maxdofs 
            println("terminating (maxdofs exceeded)...");
            maxlevel = level - 1
            if show_plots
                maxarea = 2.0^(-maxlevel)
                grid = gridgen_unitinterval(maxarea)
                FE = FiniteElements.string2FE(fem, grid, 1, 1)
            end    
            break
        end

        println("Solving Poisson problem...");    
        val4dofs = FiniteElements.createFEVector(FE);
        residual = solvePoissonProblem!(val4dofs,PD,FE);

        # compute velocity best approximation
        println("Solving L2 bestapproximation problem...");    
        val4dofsBA = FiniteElements.createFEVector(FE);
        residual = computeBestApproximation!(val4dofsBA,"L2",exact_solution!,exact_solution!,FE,length(polynomial_coefficients) + FiniteElements.get_polynomial_order(FE))

        # compute errors
        integral4cells = zeros(size(grid.nodes4cells,1),1);
        integrate!(integral4cells,eval_L2_interpolation_error!(exact_solution!, val4dofs, FE), grid, 2*length(polynomial_coefficients), 1);
        L2error[level] = sqrt(abs(sum(integral4cells)));
        println("L2_error = " * string(L2error[level]));
        integrate!(integral4cells,eval_L2_interpolation_error!(exact_solution!, val4dofsBA, FE), grid, 2*length(polynomial_coefficients), 1);
        L2errorBA[level] = sqrt(abs(sum(integral4cells)));
        println("L2_error_BA = " * string(L2errorBA[level]));

    end


    # plot
    if (show_plots)
        nodevals = FESolveCommon.eval_at_nodes(val4dofs,FE);
        pygui(true)
        PyPlot.figure(1)
        I = sortperm(grid.coords4nodes[:])
        PyPlot.plot(grid.coords4nodes[I],nodevals[I])
        PyPlot.title("Poisson Problem Solution")
        #show()
    end    

    if (show_convergence_history)
        PyPlot.figure(2)
        PyPlot.loglog(ndofs[1:maxlevel],L2error[1:maxlevel],"-o")
        PyPlot.loglog(ndofs[1:maxlevel],L2errorBA[1:maxlevel],"-o")
        PyPlot.loglog(ndofs,ndofs.^(-1),"--",color = "gray")
        PyPlot.loglog(ndofs,ndofs.^(-2),"--",color = "gray")
        PyPlot.loglog(ndofs,ndofs.^(-3),"--",color = "gray")
        PyPlot.legend(("L2 error","L2 error BA","O(h)","O(h^2)","O(h^3)"))
        PyPlot.title("Convergence history (fem=" * fem * ")")
        ax = PyPlot.gca()
        ax.grid(true)
    end 


end


main()
