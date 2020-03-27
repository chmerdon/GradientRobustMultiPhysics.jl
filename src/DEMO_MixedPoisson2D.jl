#############################################
### DEMONSTRATION SCRIPT MIXED POISSON 2D ###
#############################################
#
# solves 2D polynomial Poisson test problems in mixed formulation
# meaning find some sigma in Hdiv with
#
# sigma = grad u   &   div(sigma) = -f
#
# demonstrates:
#   - convergence rates of implemented finite element methods
#   - comparison of mixed Poisson solution and L2 bestapproximation


using Grid
using Triangulate
using Quadrature
using FiniteElements
using FESolveCommon
using FESolvePoisson
using FEEstimate
using PyPlot


# load problem data and common grid generator
include("PROBLEMdefinitions/GRID_unitsquare.jl")
include("PROBLEMdefinitions/POISSON_2D_p4bubble.jl");

function main()

    # grid and refinement termination criterions
    gridgenerator = gridgen_unitsquare
    maxlevel = 15
    maxdofs = 25000

    # other switches
    show_plots = true
    do_estimation = true

    ########################
    ### CHOOSE FEM BELOW ###
    ########################

    fem_stress = "RT0"; fem_divergence = "P0"; expectedorder = 1
    #fem_stress = "RT1"; fem_divergence = "P1dc"; expectedorder = 2
    #fem_stress = "BDM1"; fem_divergence = "P0"; expectedorder = 2
    #fem_stress = "P1"; fem_divergence = "P0"; expectedorder = 1
    #fem_stress = "P2"; fem_divergence = "P1dc"; expectedorder = 2

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

        # compute L2 error estimator
        if do_estimation
            estimator4cells = zeros(Float64,size(FE_stress.grid.nodes4cells,1))
            Estimator[level] = FEEstimate.estimate_cellwise!(estimator4cells,PD,FE_stress,FE_divergence,val4dofs)
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
        nodevals = FESolveCommon.eval_at_nodes(val4dofsBA,FE_stress);
        absval = sqrt.(sum(nodevals.^2, dims = 1))
        pointscalar!(dataset,nodevals[:,1],"|sigma|")
        data!(scalarview,dataset,"|sigma|")
        addview!(frame,scalarview,2)

        # quiver view
        vectorview=VTKView.VectorView()
        pointvector!(dataset,Array{Float64,2}(nodevals'),"sigma")
        data!(vectorview,dataset,"sigma")
        quiver!(vectorview,10,10)
        addview!(frame,vectorview,2)

        # XY plot
        plot=VTKView.XYPlot()
        addview!(frame,plot,3)
        clear!(plot)
        plotlegend!(plot,"L2 error Poisson ($fem_stress)")
        plotcolor!(plot,1,0,0)
        addplot!(plot,Array{Float64,1}(log10.(ndofs[1:maxlevel])),log10.(L2error[1:maxlevel]))
        plotlegend!(plot,"L2 error L2BestApprox ($fem_stress)")
        plotcolor!(plot,0,0,1)
        addplot!(plot,Array{Float64,1}(log10.(ndofs[1:maxlevel])),log10.(L2errorBA[1:maxlevel]))
        if (do_estimation)
            plotlegend!(plot,"Estimator ($fem_stress)")
            plotcolor!(plot,0,1,0)
            addplot!(plot,Array{Float64,1}(log10.(ndofs[1:maxlevel])),log10.(Estimator[1:maxlevel]))
        end    
        
        plotlegend!(plot,"O(h^$expectedorder)")
        plotcolor!(plot,0.5,0.5,0.5)
        addplot!(plot,Array{Float64,1}(log10.(ndofs[1:maxlevel])),Array{Float64,1}(log10.(ndofs[1:maxlevel].^(-expectedorder/2))))
    
        # show
        display(frame)
    end    


end


main()
