##########################################
### DEMONSTRATION SCRIPT COOK MEMBRANE ###
##########################################
#
# solves Cook's membrane test problem
#
# demonstrates:
#   - convergence rates of implemented finite element methods
#   - multiple boundary conditions (Dirichlet, do-nothing, symmetry boundary)
#

using Triangulate
using Grid
using Quadrature
using FiniteElements
using FESolveCommon
using FESolveLinElasticity
using VTKView

# load problem data and common grid generator
include("PROBLEMdefinitions/GRID_CookMembrane.jl")
include("PROBLEMdefinitions/ELASTICITY_2D_CookMembrane.jl");


function main()

    # problem modification switches
    E = 100000 # elasticity modulus
    nu = 0.4 # Poisson number
    lambda = (nu/(1-2*nu))*(1/(1+nu))*E
    shear_modulus = (1/(1+nu))*E
    factor_plotdisplacement = 5.0

    # refinement termination criterions
    reflevel = 3

    # other switches
    show_plots = true


    ########################
    ### CHOOSE FEM BELOW ###
    ########################

    #fem = "CR"; expectedorder = 1
    fem = "P1"; expectedorder = 1


    # load problem data
    PD = getProblemData(shear_modulus, lambda);

    FESolveLinElasticity.show(PD);

    println("Solving linear elasticity problem on refinement level...", reflevel);
    println("Generating grid by triangle...");
    maxarea = 4.0^(-reflevel+4)
    grid = gridgen_cookmembrane(maxarea)
    Grid.show(grid)

    # load finite element
    FE = FiniteElements.string2FE(fem,grid,2,2)
    FiniteElements.show(FE)

    # solve elasticity problem
    val4dofs = FiniteElements.createFEVector(FE)
    residual = solveLinElasticityProblem!(val4dofs,PD,FE);
    
    # plot
    if (show_plots)
        frame=VTKView.StaticFrame()
        clear!(frame)
        layout!(frame,3,1)
        size!(frame,1500,500)

        # grid view
        frametitle!(frame,"    final grid   |     discrete solution     |     displacement     ")
        dataset=VTKView.DataSet()
        VTKView.simplexgrid!(dataset,Array{Float64,2}(grid.coords4nodes'),Array{Int32,2}(grid.nodes4cells'))
        gridview=VTKView.GridView()
        data!(gridview,dataset)
        addview!(frame,gridview,1)

        # scalar view
        scalarview=VTKView.ScalarView()
        displacement = FESolveCommon.eval_at_nodes(val4dofs,FE);
        speed = sqrt.(sum(displacement.^2, dims = 2))
        pointscalar!(dataset,speed[:],"|U|")
        data!(scalarview,dataset,"|U|")
        addview!(frame,scalarview,2)
        
        vectorview=VTKView.VectorView()
        pointvector!(dataset,Array{Float64,2}(displacement'),"U")
        data!(vectorview,dataset,"U")
        quiver!(vectorview,10,10)
        addview!(frame,vectorview,2)
        
        # displacement
        dataset2=VTKView.DataSet()
        VTKView.simplexgrid!(dataset2,Array{Float64,2}((grid.coords4nodes + factor_plotdisplacement*displacement)'),Array{Int32,2}(grid.nodes4cells'))
        gridview2=VTKView.GridView()
        data!(gridview2,dataset2)
        addview!(frame,gridview2,3)

        # show
        display(frame)
    end    

        
end


main()
