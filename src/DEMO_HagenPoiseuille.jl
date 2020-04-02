#############################################
### DEMONSTRATION SCRIPT HAGEN-POISEUILLE ###
#############################################
#
# solves Hagen-Poiseuille test problem
#
# demonstrates:
#   - convergence rates of implemented finite element methods
#     (2nd order finite elements methods should solve it exactly)
#   - multiple boundary conditions (Dirichlet, do-nothing, symmetry boundary)
#

using Triangulate
using Grid
using Quadrature
using FiniteElements
using FESolveCommon
using FESolveStokes
using FESolveNavierStokes
using VTKView

# load problem data and common grid generator
include("PROBLEMdefinitions/GRID_unitsquare.jl")
include("PROBLEMdefinitions/STOKES_HagenPoiseuille.jl");


function main()

    # problem modification switches
    do_nothing_inlet = true
    symmetry_top = true
    nu = 1
    nonlinear = true
    maxIterations = 20

    # refinement termination criterions
    maxlevel = 7
    maxdofs = 60000

    # other switches
    show_plots = true
    use_reconstruction = 0 # do not change here
    barycentric_refinement = false # do not change here


    ########################
    ### CHOOSE FEM BELOW ###
    ########################

    #fem_velocity = "CR"; fem_pressure = "P0"; expectedorder = 1
    #fem_velocity = "CR"; fem_pressure = "P0"; use_reconstruction = 1; expectedorder = 1
    #fem_velocity = "MINI"; fem_pressure = "P1"; expectedorder = 1
    #fem_velocity = "P2";  fem_pressure = "P1"; expectedorder = 2
    #fem_velocity = "P2";  fem_pressure = "P1dc"; barycentric_refinement = true; expectedorder = 2
    #fem_velocity = "P2"; fem_pressure = "P0"; expectedorder = 2
    #fem_velocity = "P2B"; fem_pressure = "P1dc"; expectedorder = 2
    #fem_velocity = "BR"; fem_pressure = "P0"; expectedorder = 1
    fem_velocity = "BR"; fem_pressure = "P0"; use_reconstruction = 1; expectedorder = 1


    # load problem data
    PD, exact_velocity!, exact_pressure! = getProblemData(nu, do_nothing_inlet ? 0.0 : -1.0, 4);
    PD.viscosity = nu;

    # top-bottom is constant one (later symmetric boundary)
    if symmetry_top
        PD.boundarydata4bregion[3] = (result,x) -> 0.0
        PD.boundarytype4bregion[3] = 3
    end    

    # inlet/left boundary is do-nothing
    if do_nothing_inlet == true
        PD.boundarydata4bregion[4] = (result,x) -> 0.0
        PD.boundarytype4bregion[4] = 2
    end    

    FESolveStokes.show(PD);

    L2error_velocity = zeros(Float64,maxlevel)
    L2error_pressure = zeros(Float64,maxlevel)
    ndofs = zeros(Int,maxlevel)
    grid = Nothing
    FE_velocity = Nothing
    FE_pressure = Nothing
    val4dofs = Nothing
    for level = 1 : maxlevel

        println("Solving Stokes problem on refinement level...", level);
        println("Generating grid by triangle...");
        maxarea = 4.0^(-level)
        grid = gridgen_unitsquare(maxarea, barycentric_refinement)
        Grid.show(grid)

        # load finite element
        FE_velocity = FiniteElements.string2FE(fem_velocity,grid,2,2)
        FE_pressure = FiniteElements.string2FE(fem_pressure,grid,2,1)
        FiniteElements.show(FE_velocity)
        FiniteElements.show(FE_pressure)
        ndofs_velocity = FiniteElements.get_ndofs(FE_velocity);
        ndofs_pressure = FiniteElements.get_ndofs(FE_pressure);
        ndofs[level] = ndofs_velocity + ndofs_pressure;

        # stop here if too many dofs
        if ndofs[level] > maxdofs 
            println("terminating (maxdofs exceeded)...");
            maxlevel = level - 1
            if (show_plots)
                maxarea = 4.0^(-maxlevel)
                grid = gridgen_unitsquare(maxarea, barycentric_refinement)
                FE_velocity = FiniteElements.string2FE(fem_velocity,grid,2,2)
                FE_pressure = FiniteElements.string2FE(fem_pressure,grid,2,1)
            end    
            break
        end

        # solve Stokes problem
        val4dofs = zeros(Base.eltype(grid.coords4nodes),ndofs[level]);
        if nonlinear
            residual = solveNavierStokesProblem!(val4dofs,PD,FE_velocity,FE_pressure, use_reconstruction, maxIterations);
        else    
            residual = solveStokesProblem!(val4dofs,PD,FE_velocity,FE_pressure; reconst_variant = use_reconstruction);
        end
        

        # compute errors
        L2error_pressure[level] = sqrt(FESolveCommon.assemble_operator!(FESolveCommon.DOMAIN_L2_FplusA,exact_pressure!,FE_pressure ,val4dofs[ndofs_velocity+1:end]; degreeF = 1, factorA = -1.0))
        L2error_velocity[level] = sqrt(FESolveCommon.assemble_operator!(FESolveCommon.DOMAIN_L2_FplusA,exact_velocity!,FE_velocity ,val4dofs; degreeF = 2, factorA = -1.0))

    end # loop over levels

    println("\n L2 pressure error");
    show(L2error_pressure)
    println("\n L2 velocity error");
    show(L2error_velocity)

    # plot
    if (show_plots)
        frame=VTKView.StaticFrame()
        clear!(frame)
        layout!(frame,4,1)
        size!(frame,1500,500)

        # grid view
        frametitle!(frame,"    final grid     |  discrete solution (speed, pressure)  | error convergence history")
        dataset=VTKView.DataSet()
        VTKView.simplexgrid!(dataset,Array{Float64,2}(grid.coords4nodes'),Array{Int32,2}(grid.nodes4cells'))
        gridview=VTKView.GridView()
        data!(gridview,dataset)
        addview!(frame,gridview,1)

        # scalar view
        scalarview=VTKView.ScalarView()
        velo = FESolveCommon.eval_at_nodes(val4dofs,FE_velocity);
        speed = sqrt.(sum(velo.^2, dims = 2))
        pointscalar!(dataset,speed[:],"|U|")
        data!(scalarview,dataset,"|U|")
        addview!(frame,scalarview,2)
        
        vectorview=VTKView.VectorView()
        pointvector!(dataset,Array{Float64,2}(velo'),"U")
        data!(vectorview,dataset,"U")
        quiver!(vectorview,10,10)
        addview!(frame,vectorview,2)

        scalarview2=VTKView.ScalarView()
        pres = FESolveCommon.eval_at_nodes(val4dofs[FiniteElements.get_ndofs(FE_velocity)+1:end],FE_pressure);
        pointscalar!(dataset,pres[:],"p")
        data!(scalarview2,dataset,"p")
        addview!(frame,scalarview2,3)

        # XY plot
        plot=VTKView.XYPlot()
        addview!(frame,plot,4)
        clear!(plot)
        plotlegend!(plot,"L2 error velocity ($fem_velocity)")
        plotcolor!(plot,1,0,0)
        addplot!(plot,Array{Float64,1}(log10.(ndofs[1:maxlevel])),log10.(L2error_velocity[1:maxlevel]))
        plotlegend!(plot,"L2 error pressure ($fem_pressure)")
        plotcolor!(plot,0,0,1)
        addplot!(plot,Array{Float64,1}(log10.(ndofs[1:maxlevel])),log10.(L2error_pressure[1:maxlevel]))

        expectedorderL2velo = expectedorder + 1
        plotlegend!(plot,"O(h^$expectedorder)")
        plotcolor!(plot,0.67,0.67,0.67)
        addplot!(plot,Array{Float64,1}(log10.(ndofs[1:maxlevel])),Array{Float64,1}(log10.(ndofs[1:maxlevel].^(-expectedorder/2))))
        plotlegend!(plot,"O(h^$expectedorderL2velo)")
        plotcolor!(plot,0.33,0.33,0.33)
        addplot!(plot,Array{Float64,1}(log10.(ndofs[1:maxlevel])),Array{Float64,1}(log10.(ndofs[1:maxlevel].^(-expectedorderL2velo/2))))

        # show
        display(frame)
    end    

        
end


main()
