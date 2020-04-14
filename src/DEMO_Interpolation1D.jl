#############################################
### DEMONSTRATION SCRIPT INTERPOLATION 1D ###
#############################################
#
# interpolates test polynomials arbitrary order
#
# demonstrates:
#   - finite element interpolation


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

    # refinement
    reflevel = 2
    reflevel_exact = 6

    # other switches
    show_plots = true

    ########################
    ### CHOOSE FEM BELOW ###
    ########################

    #fem = "P1"
    fem = "P2"

    # choose coefficients of exact solution

    #polynomial_coefficients = [0, -3.0, 2.0, -1.0, 1.0] # quartic
    #polynomial_coefficients = [0, 0, 0, -1.0]  # cubic
    polynomial_coefficients = [1.0, 1.0, 1.0]  # quadratic
    #polynomial_coefficients = [0, 1.0]   # linear
    #polynomial_coefficients = [1.0, 0]   # constant
    
    # generate problem data (Poisson data is used just to get exact_solution!)
    PD, exact_solution! = getProblemData(polynomial_coefficients);

    # generate grid
    grid = gridgen_unitinterval(2.0^(-reflevel))
    Grid.show(grid)

    # generate FE
    FE = FiniteElements.string2FE(fem, grid, 1, 1)
    FiniteElements.show(FE)
    ndofs = FiniteElements.get_ndofs(FE);

    # interpolate
    println("Interpolating...");    
    val4dofs = FiniteElements.createFEVector(FE);
    computeFEInterpolation!(val4dofs, exact_solution!, FE);

    # compute interpolation error
    L2error = sqrt(FESolveCommon.assemble_operator!(FESolveCommon.DOMAIN_L2_FplusA,exact_solution!,FE,val4dofs; degreeF = length(polynomial_coefficients)-1, factorA = -1.0))
    println("L2_interpolation_error = " * string(L2error));
    
    # plot
    if (show_plots)
        frame=VTKView.StaticFrame()
        clear!(frame)
        layout!(frame,1,1)
        size!(frame,1000,500)
        frametitle!(frame,"interpolation vs. exact")
        
        # XY solution plot
        plot=VTKView.XYPlot()
        addview!(frame,plot,1)
        clear!(plot)

        # interpolation
        nodevals = FESolveCommon.eval_at_nodes(val4dofs,FE);
        I = sortperm(grid.coords4nodes[:])
        plotlegend!(plot,"interpolation")
        plotcolor!(plot,1,0,0)
        addplot!(plot,grid.coords4nodes[I],nodevals[I])

        # exact 
        grid_exact = gridgen_unitinterval(2.0^(-reflevel_exact))
        nodevals_exact = zeros(Float64,size(grid_exact.coords4nodes,1))
        result = [0.0]
        for j = 1 : size(grid_exact.coords4nodes,1)
            exact_solution!(result,grid_exact.coords4nodes[j,:]); 
            nodevals_exact[j] = result[1]
        end
        I = sortperm(grid_exact.coords4nodes[:])
        plotlegend!(plot,"exact")
        plotcolor!(plot,0,0,1)
        addplot!(plot,grid_exact.coords4nodes[I],nodevals_exact[I])

        # legend size/position
        legendsize!(plot,0.3,0.15)
        legendposition!(plot,0.6,0.2)
        
        # show
        display(frame)
    end    
end


main()
