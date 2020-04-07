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
using VTKView

# load problem data and common grid generator
include("PROBLEMdefinitions/GRID_unitsquare.jl")
include("PROBLEMdefinitions/STOKES_2D_polynomials.jl");


function main()

    # problem modification switches
    polynomial_order = 2

    # refinement termination criterions
    maxlevel = 4
    maxdofs = 60000

    # other switches
    show_plots = true
    use_square_grid = false # do not change here


    ########################
    ### CHOOSE FEM BELOW ###
    ########################

    # Hdiv-conforming FE
    #fem = "RT0"; expectedorder = 1
    #fem = "RT1"; expectedorder = 2
    #fem = "BDM1"; expectedorder = 2

    # Hdiv on squares
    fem = "ABF0"; expectedorder = 1; use_square_grid = true

    # H1-conforming FE
    #fem = "CR"; expectedorder = 2
    #fem = "MINI"; expectedorder = 2
    #fem = "P2"; expectedorder = 3
    #fem = "P2B"; expectedorder = 3
    #fem = "BR"; expectedorder = 2

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
        if use_square_grid == true
            grid = gridgen_unitsquare_squares(maxarea)
        else
            grid = gridgen_unitsquare(maxarea)
        end    
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
                if use_square_grid == true
                    grid = gridgen_unitsquare_squares(maxarea)
                else
                    grid = gridgen_unitsquare(maxarea)
                end    
                FE = FiniteElements.string2FE(fem,grid,2);
            end    
            break
        end

        # compute Hdiv best-approximation
        val4dofs = FiniteElements.createFEVector(FE);
        computeBestApproximation!(val4dofs,"L2",exact_velocity!,exact_velocity!,FE,polynomial_order + FiniteElements.get_polynomial_order(FE))

        # compute error of Hdiv best-approximation
        L2error_velocity[level] = sqrt(FESolveCommon.assemble_operator!(FESolveCommon.DOMAIN_L2_FplusA,exact_velocity!,FE ,val4dofs; degreeF = polynomial_order, factorA = -1.0))


    end # loop over levels

    println("\n L2 error");
    show(L2error_velocity)

    # plot
    if (show_plots)
        frame=VTKView.StaticFrame()
        clear!(frame)
        layout!(frame,3,1)
        size!(frame,1500,500)
        
        velo = FESolveCommon.eval_at_nodes(val4dofs,FE);
        speed = sqrt.(sum(velo.^2, dims = 2))
        if use_square_grid
            grid.nodes4cells = Grid.divide_into_triangles(Grid.ElemType2DParallelogram(),grid.nodes4cells)
        end    

        # grid view
        frametitle!(frame,"    final grid     |  discrete solution | error convergence history")
        dataset=VTKView.DataSet()
        VTKView.simplexgrid!(dataset,Array{Float64,2}(grid.coords4nodes'),Array{Int32,2}(grid.nodes4cells'))
        gridview=VTKView.GridView()
        data!(gridview,dataset)
        addview!(frame,gridview,1)

        # scalar view
        scalarview=VTKView.ScalarView()
        pointscalar!(dataset,speed[:],"|U|")
        data!(scalarview,dataset,"|U|")
        addview!(frame,scalarview,2)
        
        vectorview=VTKView.VectorView()
        pointvector!(dataset,Array{Float64,2}(velo'),"U")
        data!(vectorview,dataset,"U")
        quiver!(vectorview,10,10)
        addview!(frame,vectorview,2)

        # XY plot
        plot=VTKView.XYPlot()
        addview!(frame,plot,3)
        clear!(plot)
        plotlegend!(plot,"L2 error ($fem)")
        plotcolor!(plot,1,0,0)
        addplot!(plot,Array{Float64,1}(log10.(ndofs[1:maxlevel])),log10.(L2error_velocity[1:maxlevel]))

        plotlegend!(plot,"O(h^$expectedorder)")
        plotcolor!(plot,0.67,0.67,0.67)
        addplot!(plot,Array{Float64,1}(log10.(ndofs[1:maxlevel])),Array{Float64,1}(log10.(ndofs[1:maxlevel].^(-expectedorder/2))))
        
        # show
        display(frame)
    end    
        
end


main()
