##################################################################
### DEMONSTRATION SCRIPT COMPRESSIBLE STOKES STRATIFIEDNO-FLOW ###
##################################################################
#
# solves compressible Stokes test problem stratified no-flow
#
# means: exact velocity solution is zero, density is non-constant and only y-dependent
#
#
# demonstrates:
#   - gradient-robustness (use_reconstruction > 0) important to get much more accuracy of hydrostic solution
#

using Triangulate
using Grid
using Quadrature
using FiniteElements
using FESolveCommon
using FESolveStokes
using FESolveCompressibleStokes
ENV["MPLBACKEND"]="tkagg"
using PyPlot

# load problem data and common grid generator
include("PROBLEMdefinitions/GRID_unitsquare.jl")


function main()

    # problem modification switches
    shear_modulus = 1
    symmetric_gradient = true
    nonlinear_convection = true
    lambda = 0.0
    c = 1
    total_mass = 2.0
    gamma = 1 # exact_density only exact for gamma = 1 !!!
    dt = shear_modulus*0.5/c
    maxT = 1000
    stationarity_tolerance = 1e-11

    function equation_of_state!(pressure,density)
        for j=1:length(density)
            pressure[j] = c*density[j]^gamma
        end    
    end    

    # refinement termination criterions
    maxlevel = 3
    maxdofs = 40000

    # other switches
    show_plots = true
    use_reconstruction = 0 # do not change here
    barycentric_refinement = false # do not change here

    ########################
    ### CHOOSE FEM BELOW ###
    ########################

    #fem_velocity = "CR"; fem_densitypressure = "P0"
    #fem_velocity = "CR"; fem_densitypressure = "P0"; use_reconstruction = 1
    #fem_velocity = "BR"; fem_densitypressure = "P0"
    fem_velocity = "BR"; fem_densitypressure = "P0"; use_reconstruction = 1



    function zero_data!(result,x)
        fill!(result,0.0)
    end    
    

    d = log(total_mass/(c*(exp(1)^(1/c)-1.0)))
    function exact_density!(result,x) # only exact for gamma = 1
        result[1] = exp((1-x[2])/c + d)
    end 

    function gravity!(result,x)
        result[1] = 0
        result[2] = -1.0
    end    


    # transform into compressible getProblemData
    PD = FESolveCompressibleStokes.CompressibleStokesProblemDescription()
    PD.name = "stratified no-flow";
    PD.shear_modulus = shear_modulus
    PD.use_symmetric_gradient = symmetric_gradient
    PD.use_nonlinear_convection = nonlinear_convection
    PD.lambda = lambda
    PD.total_mass = total_mass
    PD.volumedata4region = [zero_data!]
    PD.gravity = gravity!
    PD.quadorder4gravity = 0
    PD.quadorder4region = [0]
    PD.boundarydata4bregion = [zero_data!,zero_data!,zero_data!,zero_data!]
    PD.boundarytype4bregion = [1,1,1,1]
    PD.quadorder4bregion = [0,0,0,0]
    PD.equation_of_state = equation_of_state!
    FESolveCompressibleStokes.show(PD);

    L2error_velocity = zeros(Float64,maxlevel)
    L2error_density = zeros(Float64,maxlevel)
    nrIterations = zeros(Int64,maxlevel)
    ndofs = zeros(Int,maxlevel)
    grid = Nothing
    FE_velocity = Nothing
    FE_densitypressure = Nothing
    velocity = Nothing
    density = Nothing
    for level = 1 : maxlevel

        println("Solving compressible Stokes problem on refinement level...", level);
        println("Generating grid by triangle...");
        maxarea = 4.0^(-level-2)
        grid = gridgen_unitsquare(maxarea, barycentric_refinement)
        Grid.show(grid)

        # load finite element
        FE_velocity = FiniteElements.string2FE(fem_velocity,grid,2,2)
        FE_densitypressure = FiniteElements.string2FE(fem_densitypressure,grid,2,1)
        FiniteElements.show(FE_velocity)
        FiniteElements.show(FE_densitypressure)
        ndofs_velocity = FiniteElements.get_ndofs(FE_velocity);
        ndofs_densitypressure = FiniteElements.get_ndofs(FE_densitypressure);
        ndofs[level] = ndofs_velocity + 2*ndofs_densitypressure;

        # stop here if too many dofs
        if ndofs[level] > maxdofs 
            println("terminating (maxdofs exceeded)...");
            maxlevel = level - 1
            if (show_plots)
                maxarea = 4.0^(-maxlevel)
                grid = gridgen_unitsquare(maxarea, barycentric_refinement)
                FE_velocity = FiniteElements.string2FE(fem_velocity,grid,2,2)
                FE_densitypressure = FiniteElements.string2FE(fem_densitypressure,grid,2,1)
            end    
            break
        end


        # initial velocity is zero
        velocity = zeros(Float64,ndofs_velocity);
        
        # initial density is constant
        density = FiniteElements.createFEVector(FE_densitypressure)
        density[:] .= total_mass
        CSS = FESolveCompressibleStokes.setupCompressibleStokesSolver(PD,FE_velocity,FE_densitypressure,velocity,density,use_reconstruction)

        change = 1
        while ((change > stationarity_tolerance) && (maxT > CSS.current_time))
            nrIterations[level] += 1
            change = FESolveCompressibleStokes.PerformTimeStep(CSS,dt)
        end    

        velocity[:] = CSS.current_velocity[:]
        density[:] = CSS.current_density[:]

        # compute errors
        L2error_density[level] = sqrt(FESolveCommon.assemble_operator!(FESolveCommon.DOMAIN_L2_FplusA,exact_density!,FE_densitypressure,density; factorA = -1.0, degreeF = 10))
        L2error_velocity[level] = sqrt(FESolveCommon.assemble_operator!(FESolveCommon.DOMAIN_L2_FplusA,zero_data!,FE_velocity,velocity; factorA = -1.0, degreeF = 0))

    end # loop over levels

    println("\n L2 density error");
    show(L2error_density)
    println("\n L2 velocity error");
    show(L2error_velocity)
    println("\n nrIterations");
    show(nrIterations)

    
    # plot
    if (show_plots)
        frame=VTKView.StaticFrame()
        clear!(frame)
        layout!(frame,4,1)
        size!(frame,1500,500)

        # grid view
        frametitle!(frame,"    final grid     |  discrete solution (speed, density)  | error convergence history")
        dataset=VTKView.DataSet()
        VTKView.simplexgrid!(dataset,Array{Float64,2}(grid.coords4nodes'),Array{Int32,2}(grid.nodes4cells'))
        gridview=VTKView.GridView()
        data!(gridview,dataset)
        addview!(frame,gridview,1)

        # scalar view
        scalarview=VTKView.ScalarView()
        velo = FESolveCommon.eval_at_nodes(velocity,FE_velocity);
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
        density = FESolveCommon.eval_at_nodes(density,FE_densitypressure);
        pointscalar!(dataset,density[:],"rho")
        data!(scalarview2,dataset,"rho")
        addview!(frame,scalarview2,3)

        # XY plot
        plot=VTKView.XYPlot()
        addview!(frame,plot,4)
        clear!(plot)
        plotlegend!(plot,"L2 error velocity ($fem_velocity)")
        plotcolor!(plot,1,0,0)
        addplot!(plot,Array{Float64,1}(log10.(ndofs[1:maxlevel])),log10.(L2error_velocity[1:maxlevel]))
        plotlegend!(plot,"L2 error density ($fem_densitypressure)")
        plotcolor!(plot,0,0,1)
        addplot!(plot,Array{Float64,1}(log10.(ndofs[1:maxlevel])),log10.(L2error_density[1:maxlevel]))

        expectedorder = 1
        expectedorderL2velo = 2
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
