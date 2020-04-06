#############################################
### DEMONSTRATION SCRIPT INTERPOLATION 2D ###
#############################################
#
# interpolates test polynomials arbitrary order
#
# demonstrates:
#   - finite element interpolation


using Grid
using Triangulate
using Quadrature
using FiniteElements
using FESolveCommon
using FESolvePoisson
using VTKView


# load problem data and common grid generator
include("PROBLEMdefinitions/GRID_unitsquare.jl")
include("PROBLEMdefinitions/POISSON_2D_polynomials.jl");

function main()

    # refinement
    #gridgenerator = gridgen_unitsquare;
    gridgenerator = gridgen_unitsquare_squares;

    reflevel = 4
    reflevel_exact = 6

    # other switches
    show_plots = true

    ########################
    ### CHOOSE FEM BELOW ###
    ########################

    fem = "P1"

    #polynomial_coefficients = [0 -3 2 -1 1; 0 2 -1 0 -0.5] # quartic
    #polynomial_coefficients = [0 0 0 -1.0; 0 1.0 0 1.0] # cubic
    polynomial_coefficients = [0 0 -1; 1 0.5 0.5]  # quadratic
    #polynomial_coefficients = [0 1; 0.5 -1]   # linear
    #polynomial_coefficients = [1 0; 0.5 0]   # constant

    # generate problem data (Poisson data is used just to get exact_solution!)
    PD, exact_solution! = getProblemData(polynomial_coefficients, 1.0);

    # generate grid
    grid = gridgenerator(4.0^(-reflevel))
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
    L2error = sqrt(FESolveCommon.assemble_operator!(FESolveCommon.DOMAIN_L2_FplusA,exact_solution!,FE,val4dofs; degreeF = 4, factorA = -1.0))
    println("L2_interpolation_error = " * string(L2error));

end


main()
