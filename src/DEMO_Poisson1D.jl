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
using VTKView


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

    fem = "P1"; expectedorder = 1
    #fem = "P2"; expectedorder = 2

    # choose coefficients of exact solution

    polynomial_coefficients = [0, -3.0, 2.0, -1.0, 1.0] # quartic
    #polynomial_coefficients = [0, 0, 0, -1.0]  # cubic
    #polynomial_coefficients = [0, 0, -1.0]  # quadratic
    #polynomial_coefficients = [0, 1.0]   # linear
    #polynomial_coefficients = [1.0, 0]   # constant
    
    # load problem data
    PD, exact_solution! = getProblemData(polynomial_coefficients);
    FESolvePoisson.show(PD);

    L2error = zeros(Float64,maxlevel)
    L2errorBA = zeros(Float64,maxlevel)
    ndofs = zeros(Int64,maxlevel)            
    val4dofs = Nothing
    val4dofsBA = Nothing
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
        L2error[level] = sqrt(FESolveCommon.assemble_operator!(FESolveCommon.DOMAIN_L2_FplusA,exact_solution!,FE,val4dofs; degreeF = length(polynomial_coefficients)-1, factorA = -1.0))
        println("L2_error = " * string(L2error[level]));
        L2errorBA[level] = sqrt(FESolveCommon.assemble_operator!(FESolveCommon.DOMAIN_L2_FplusA,exact_solution!,FE,val4dofsBA; degreeF = length(polynomial_coefficients)-1, factorA = -1.0))
        println("L2_error_BA = " * string(L2errorBA[level]));

    end

    # plot
    if (show_plots)
        frame=VTKView.StaticFrame()
        clear!(frame)
        layout!(frame,2,1)
        size!(frame,1000,500)
        frametitle!(frame,"  discrete solution  | error convergence history")
        

        # XY solution plot
        plot=VTKView.XYPlot()
        addview!(frame,plot,1)
        clear!(plot)
        nodevals = FESolveCommon.eval_at_nodes(val4dofs,FE);
        I = sortperm(grid.coords4nodes[:])
        plotlegend!(plot,"Poisson solution")
        plotcolor!(plot,1,0,0)
        addplot!(plot,grid.coords4nodes[I],nodevals[I])
        nodevalsBA = FESolveCommon.eval_at_nodes(val4dofsBA,FE);
        I = sortperm(grid.coords4nodes[:])
        plotlegend!(plot,"L2 best approximation")
        plotcolor!(plot,0,0,1)
        addplot!(plot,grid.coords4nodes[I],nodevalsBA[I])
        
        # XY error plot
        plot=VTKView.XYPlot()
        addview!(frame,plot,2)
        clear!(plot)
        plotlegend!(plot,"||u-u_h|| ($fem)")
        plotcolor!(plot,1,0,0)
        addplot!(plot,Array{Float64,1}(log10.(ndofs[1:maxlevel])),log10.(L2error[1:maxlevel]))
        plotlegend!(plot,"||u-u_best|| ($fem)")
        plotcolor!(plot,0,0,1)
        addplot!(plot,Array{Float64,1}(log10.(ndofs[1:maxlevel])),log10.(L2errorBA[1:maxlevel]))
    
        expectedorderL2 = expectedorder + 1
        plotlegend!(plot,"O(h^$expectedorderL2)")
        plotcolor!(plot,0.5,0.5,0.5)
        addplot!(plot,Array{Float64,1}(log10.(ndofs[1:maxlevel])),Array{Float64,1}(log10.(ndofs[1:maxlevel].^(-Float64(expectedorderL2)))))
    
        legendsize!(plot,0.3,0.15)
        legendposition!(plot,0.28,0.12)

        # show
        display(frame)
    end    


end


main()
