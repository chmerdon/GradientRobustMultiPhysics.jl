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
using VTKView


# load problem data and common grid generator
include("PROBLEMdefinitions/GRID_lshape.jl")
include("PROBLEMdefinitions/GRID_unitsquare.jl")
include("PROBLEMdefinitions/POISSON_2D_polynomials.jl");

function main()

    # grid and refinement termination criterions
    gridgenerator = gridgen_unitsquare
    #gridgenerator = gridgen_lshape
    maxlevel = 15
    maxdofs = 20000

    # other switches
    show_plots = true
    do_estimation = true

    ########################
    ### CHOOSE FEM BELOW ###
    ########################

    #fem = "CR"; expectedorder = 1
    #fem = "P1"; expectedorder = 1
    #fem = "MINI"; expectedorder = 1
    fem = "P2"; expectedorder = 2
    #fem = "P2B"; expectedorder = 2

    # choose coefficients of exact solution

    polynomial_coefficients = [0 -3 2 -1 1; 0 2 -1 0 -0.5] # quartic
    #polynomial_coefficients = [0 0 0 -1.0; 0 1.0 0 1.0] # cubic
    #polynomial_coefficients = [0 0 -1; 1 0.5 0.5]  # quadratic
    #polynomial_coefficients = [0 1; 0.5 -1]   # linear
    #polynomial_coefficients = [1 0; 0.5 0]   # constant

    #diffusion = 10.0 # scalar constant diffusion
    #diffusion = [2.0 0.5] # diagonal constant diffusion matrix
    #diffusion = [2.0 0.0; 0.0 0.5] # arbitrary non-diagonal constant diffusion matrix
    diffusion = [2.0 0.5; 0.5 0.5] # arbitrary non-diagonal constant diffusion matrix (symmetric positive definit)
    
    # load problem data
    PD, exact_solution! = getProblemData(polynomial_coefficients, diffusion);
    FESolvePoisson.show(PD);

    L2error = zeros(Float64,maxlevel)
    L2errorBA = zeros(Float64,maxlevel)
    Estimator = zeros(Float64,maxlevel)
    estimator4cells = Nothing
    ndofs = zeros(Int64,maxlevel)            
    val4dofs = Nothing
    FE = Nothing
    grid = Nothing
    for level = 1 : maxlevel

        # generate grid
        maxarea = 4.0^(-level)
        grid = gridgenerator(maxarea)
        Grid.show(grid)

        # generate FE
        FE = FiniteElements.string2FE(fem, grid, 2, 1)
        ensure_nodes4faces!(grid);
        ensure_volume4cells!(grid);
        FiniteElements.show(FE)
        ndofs[level] = FiniteElements.get_ndofs(FE);

        # stop here if too many dofs
        if ndofs[level] > maxdofs 
            println("terminating (maxdofs exceeded)...");
            maxlevel = level - 1
            if show_plots
                maxarea = 4.0^(-maxlevel)
                grid = gridgenerator(maxarea)
                FE = FiniteElements.string2FE(fem, grid, 2, 1)
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

        # compute L2 error estimator
        if do_estimation
            estimator4cells = zeros(Float64,size(FE.grid.nodes4cells,1))
            Estimator[level] = FEEstimate.estimate_cellwise!(estimator4cells,PD,FE,val4dofs,0)
            println("estimator = " * string(Estimator[level]));
        end    

    end


    # plot
    if (show_plots)
        frame=VTKView.StaticFrame()
        clear!(frame)
        layout!(frame,3,1)
        size!(frame,1500,500)

        # grid view
        frametitle!(frame,"    final grid     |  discrete solution  | error convergence history")
        dataset=VTKView.DataSet()
        VTKView.simplexgrid!(dataset,Array{Float64,2}(grid.coords4nodes'),Array{Int32,2}(grid.nodes4cells'))
        gridview=VTKView.GridView()
        data!(gridview,dataset)
        addview!(frame,gridview,1)

        # scalar view
        scalarview=VTKView.ScalarView()
        nodevals = FESolveCommon.eval_at_nodes(val4dofs,FE);
        pointscalar!(dataset,nodevals[:,1],"U")
        data!(scalarview,dataset,"U")
        addview!(frame,scalarview,2)

        # XY plot
        plot=VTKView.XYPlot()
        addview!(frame,plot,3)
        clear!(plot)
        plotlegend!(plot,"L2 error Poisson ($fem)")
        plotcolor!(plot,1,0,0)
        addplot!(plot,Array{Float64,1}(log10.(ndofs[1:maxlevel])),log10.(L2error[1:maxlevel]))
        plotlegend!(plot,"L2 error L2BestApprox ($fem)")
        plotcolor!(plot,0,0,1)
        addplot!(plot,Array{Float64,1}(log10.(ndofs[1:maxlevel])),log10.(L2errorBA[1:maxlevel]))
        plotlegend!(plot,"Estimator ($fem)")
        plotcolor!(plot,0,1,0)
        addplot!(plot,Array{Float64,1}(log10.(ndofs[1:maxlevel])),log10.(Estimator[1:maxlevel]))
        
        expectedorderL2 = expectedorder + 1
        plotlegend!(plot,"O(h^$expectedorderL2)")
        plotcolor!(plot,0.5,0.5,0.5)
        addplot!(plot,Array{Float64,1}(log10.(ndofs[1:maxlevel])),Array{Float64,1}(log10.(ndofs[1:maxlevel].^(-expectedorderL2/2))))
    
        # show
        display(frame)
    end    
end


main()
