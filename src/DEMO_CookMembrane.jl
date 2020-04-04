##########################################
### DEMONSTRATION SCRIPT COOK MEMBRANE ###
##########################################
#
# solves Cook's membrane test problem
#
# demonstrates:
#   - locking of P1-element for nu=0.4999 (bends not as high as other elements)
#   - composite finite elements (Kouhia-Stenberg P1xCR, CRxP1)
#

using Triangulate # third-party; for mesh generation
using Grid # for storing mesh data
using FiniteElements # for handling finite element
using FESolveCommon # contains basic stuff like best approximations, interpolations
using FESolveLinElasticity # contains solver for linear elasticity
using VTKView # third-party, for plotting

# load problem data and common grid generator
include("PROBLEMdefinitions/GRID_CookMembrane.jl")
include("PROBLEMdefinitions/ELASTICITY_2D_CookMembrane.jl");

function main()

    # problem parameters
    elasticity_modulus = 100000 # elasticity modulus
    poisson_number = 0.4999 # Poisson number
    shear_modulus = (1/(1+poisson_number))*elasticity_modulus
    lambda = (poisson_number/(1-2*poisson_number))*shear_modulus
    reflevel = 3 # refinement level of Cook membrane
    show_plots = true # plot grid and displacement?
    factor_plotdisplacement = 10.0 # scaling of displacement in deformed grid plot

    ########################
    ### CHOOSE FEM BELOW ###
    ########################

    #fem = "P1"; expectedorder = 1
    #fem = "P2"; expectedorder = 2
    fem = ["CR","P1"]; expectedorder = 1 # Kouhia-Stenberg composite
    #fem = ["P1","CR"]; expectedorder = 1 # Kouhia-Stenberg composite

    # load problem data
    PD = getProblemData(shear_modulus, lambda);
    FESolveLinElasticity.show(PD);

    # load grid
    grid = gridgen_cookmembrane(4.0^(-reflevel+4))
    Grid.show(grid)

    # load finite element
    if typeof(fem) == Array{String,1}
        FE = FiniteElements.string2FE(fem,grid,2)
    else
        FE = FiniteElements.string2FE(fem,grid,2,2)
    end    
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
        frametitle!(frame,"        grid            |      discrete solution      |       grid+displacement  ")
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
