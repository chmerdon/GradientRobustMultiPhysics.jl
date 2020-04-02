##########################################################
### DEMONSTRATION SCRIPT STOKES P7-VORTEX COMPRESSIBLE ###
##########################################################
#
# solves compressible Stokes test problem p7vortex for densities of arbitrary polynomial degree
#
# theta = stream function as in p7vortex example
# velocity = curl(theta) / density
#
# hence density*velocity is divergence-free
#
# demonstrates:
#   - convergence rates of compressible Stokes solver
#   - benefits of gradient-robustness (use_reconstruction > 0) for small shear_moduli
#

using Triangulate
using Grid
using Quadrature
using FiniteElements
using FESolveCommon
using FESolveStokes
using FESolveCompressibleStokes
using VTKView

# load problem data and common grid generator
include("PROBLEMdefinitions/GRID_unitsquare.jl")
include("PROBLEMdefinitions/CSTOKES_p7vortex.jl");


function main()

    # problem modification switches
    shear_modulus = 1.0 # coefficient for Laplacian/div(eps(u))
    lambda = -2.0/3.0*shear_modulus # coefficient for grad(div(u))
    total_mass = 1
    nonlinear_convection = true
    c = 1 # coefficient of density in equation of state (inverse of squared Mach number)
    gamma = 2 # exponent of density in equation of state
    density_power = 2 # density will be a polynomial of this degree

    # discretisation parameter
    dt = (2*shear_modulus + lambda)*0.4/c # time step has to be small enough for convergence
    stationarity_tolerance = 1e-10 # termination condition for time loop
    symmetric_gradient = true # use div(eps(u)) instead of Laplacian
    maxT = 1000 # termination condition for time loop
    initial_density_bestapprox = true # otherwise constant initial density is used
    use_gravity = true # adds a density-dependent gravity term

    # refinement termination criterions
    maxlevel = 3
    maxdofs = 40000

    # other switches
    show_plots = true
    use_reconstruction = 0 # do not change here
    barycentric_refinement = false # do not change here
    order_error = 10

    ########################
    ### CHOOSE FEM BELOW ###
    ########################

    #fem_velocity = "CR"; fem_densitypressure = "P0"
    #fem_velocity = "CR"; fem_densitypressure = "P0"; use_reconstruction = 1
    fem_velocity = "BR"; fem_densitypressure = "P0"
    #fem_velocity = "BR"; fem_densitypressure = "P0"; use_reconstruction = 1


    # load problem data
    PD, exact_velocity!, exact_density!, exact_pressure! = getProblemData(shear_modulus, lambda; use_nonlinear_convection = nonlinear_convection, use_gravity = use_gravity, symmetric_gradient = symmetric_gradient, gamma = gamma, c = c, density_power = density_power, nrBoundaryRegions = 4);
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
        maxarea = 4.0^(-level-1)
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


        # solve for initial value by best approximation 
        velocity = zeros(Float64,ndofs[level]);
        density = zeros(Float64,ndofs_densitypressure);
        residual = FESolveStokes.computeDivFreeBestApproximation!(velocity,exact_velocity!,exact_velocity!,FE_velocity,FE_densitypressure,7)
        velocity = velocity[1:ndofs_velocity]

        initial_density = FiniteElements.createFEVector(FE_densitypressure)
        if initial_density_bestapprox
            residual = FESolveCommon.computeBestApproximation!(density,"L2",exact_density!,Nothing,FE_densitypressure,density_power)    
        else
            initial_density[:] .= total_mass
        end

        CSS = FESolveCompressibleStokes.setupCompressibleStokesSolver(PD,FE_velocity,FE_densitypressure,velocity,density,use_reconstruction)

        change = 1
        while ((change > stationarity_tolerance) && (maxT > CSS.current_time))
            nrIterations[level] += 1
            change = FESolveCompressibleStokes.PerformTimeStep(CSS,dt)
        end    

        velocity[:] = CSS.current_velocity[:]
        density[:] = CSS.current_density[:]

        # compute errors
        L2error_density[level] = sqrt(FESolveCommon.assemble_operator!(FESolveCommon.DOMAIN_L2_FplusA,exact_density!,FE_densitypressure,density; factorA = -1.0, degreeF = density_power))
        L2error_velocity[level] = sqrt(FESolveCommon.assemble_operator!(FESolveCommon.DOMAIN_L2_FplusA,exact_velocity!,FE_velocity,velocity; factorA = -1.0, degreeF = 7))

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
