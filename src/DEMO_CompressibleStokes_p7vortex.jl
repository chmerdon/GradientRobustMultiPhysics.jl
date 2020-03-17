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
ENV["MPLBACKEND"]="tkagg"
using PyPlot

# load problem data and common grid generator
include("PROBLEMdefinitions/GRID_unitsquare.jl")
include("PROBLEMdefinitions/CSTOKES_p7vortex.jl");


function main()

    # problem modification switches
    shear_modulus = 1.0 # coefficient for Laplacian/div(eps(u))
    lambda = -2.0/3.0*shear_modulus # coefficient for grad(div(u))
    total_mass = 1
    c = 1 # coefficient of density in equation of state (inverse of squared Mach number)
    gamma = 2 # exponent of density in equation of state
    density_power = 2 # density will be a polynomial of this degree

    # discretisation parameter
    dt = (2*shear_modulus + lambda)*0.2/c # time step has to be small enough for convergence
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
    show_convergence_history = true
    use_reconstruction = 0 # do not change here
    barycentric_refinement = false # do not change here
    order_error = 10

    ########################
    ### CHOOSE FEM BELOW ###
    ########################

    #fem_velocity = "CR"; fem_densitypressure = "P0"
    #fem_velocity = "CR"; fem_densitypressure = "P0"; use_reconstruction = 1
    #fem_velocity = "BR"; fem_densitypressure = "P0"
    fem_velocity = "BR"; fem_densitypressure = "P0"; use_reconstruction = 1


    # load problem data
    PD, exact_velocity!, exact_density!, exact_pressure! = getProblemData(shear_modulus, lambda; use_gravity = use_gravity, symmetric_gradient = symmetric_gradient, gamma = gamma, c = c, density_power = density_power, nrBoundaryRegions = 4);
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
        integral4cells = zeros(size(grid.nodes4cells,1),1);
        integrate!(integral4cells,eval_L2_interpolation_error!(exact_density!, density, FE_densitypressure), grid, order_error, 1);
        L2error_density[level] = sqrt(abs(sum(integral4cells)));
        integral4cells = zeros(size(grid.nodes4cells,1),2);
        integrate!(integral4cells,eval_L2_interpolation_error!(exact_velocity!, velocity, FE_velocity), grid, order_error, 2);
        L2error_velocity[level] = sqrt(abs(sum(integral4cells[:])));

    end # loop over levels

    println("\n L2 density error");
    show(L2error_density)
    println("\n L2 velocity error");
    show(L2error_velocity)
    println("\n nrIterations");
    show(nrIterations)

    #plot
    if (show_plots)
        pygui(true)
        
        # evaluate velocity and pressure at grid points
        velo = FESolveCommon.eval_at_nodes(velocity,FE_velocity);
        density = FESolveCommon.eval_at_nodes(density,FE_densitypressure);
        speed = sqrt.(sum(velo.^2, dims = 2))
        
        PyPlot.figure(1)
        tcf = PyPlot.tricontourf(view(grid.coords4nodes,:,1),view(grid.coords4nodes,:,2),speed[:])
        PyPlot.axis("equal")
        PyPlot.title("velocity speed")

        PyPlot.figure(2)
        PyPlot.plot_trisurf(view(grid.coords4nodes,:,1),view(grid.coords4nodes,:,2),density[:],cmap=get_cmap("ocean"))
        PyPlot.title("density")
        show()
    end

    if (show_convergence_history)
        PyPlot.figure()
        PyPlot.loglog(ndofs[1:maxlevel],L2error_velocity[1:maxlevel],"-o")
        PyPlot.loglog(ndofs[1:maxlevel],L2error_density[1:maxlevel],"-o")
        PyPlot.loglog(ndofs,ndofs.^(-1/2),"--",color = "gray")
        PyPlot.loglog(ndofs,ndofs.^(-1),"--",color = "gray")
        PyPlot.loglog(ndofs,ndofs.^(-3/2),"--",color = "gray")
        PyPlot.legend(("L2 error velocity","L2 error density","O(h)","O(h^2)","O(h^3)"))   
        PyPlot.title("Convergence history (fem=" * fem_velocity * "/" * fem_densitypressure * ")")
        ax = PyPlot.gca()
        ax.grid(true)
    end    

        
end


main()
