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
    lambda = 0.0
    c = 10
    total_mass = 2.0
    gamma = 1 # exact_density only exact for gamma = 1 !!!
    dt = shear_modulus*0.2/c
    maxT = 1000
    stationarity_tolerance = 1e-11

    function equation_of_state!(pressure,density)
        for j=1:length(density)
            pressure[j] = c*density[j]^gamma
        end    
    end    

    # refinement termination criterions
    maxlevel = 5
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



    function zero_data!(result,x)
        fill!(result,0.0)
    end    
    

    d = log(total_mass/(c*(exp(1)^(1/c)-1.0)))
    show(exp(d))
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
    ndofs = zeros(Int,maxlevel)
    grid = Nothing
    FE_velocity = Nothing
    FE_densitypressure = Nothing
    val4dofs = Nothing
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


        # initial velocity is zero
        val4dofs = zeros(Float64,ndofs[level]);
        
        Grid.ensure_volume4cells!(grid)
        initial_density = FiniteElements.createFEVector(FE_densitypressure)
        initial_density[:] .= total_mass
        CSS = FESolveCompressibleStokes.setupCompressibleStokesSolver(PD,FE_velocity,FE_densitypressure,val4dofs[1:ndofs_velocity],initial_density,use_reconstruction)

        change = 1
        while ((change > stationarity_tolerance) && (maxT > CSS.current_time))

            change = FESolveCompressibleStokes.PerformTimeStep(CSS,dt)
        end    

        val4dofs[:] = CSS.current_solution[:]


        # compute errors
        integral4cells = zeros(size(grid.nodes4cells,1),1);
        integrate!(integral4cells,eval_L2_interpolation_error!(exact_density!, val4dofs[ndofs_velocity+1:end], FE_densitypressure), grid, order_error, 1);
        L2error_density[level] = sqrt(abs(sum(integral4cells)));
        integral4cells = zeros(size(grid.nodes4cells,1),2);
        integrate!(integral4cells,eval_L2_interpolation_error!(zero_data!, val4dofs[1:ndofs_velocity], FE_velocity), grid, order_error, 2);
        L2error_velocity[level] = sqrt(abs(sum(integral4cells[:])));

    end # loop over levels

    println("\n L2 density error");
    show(L2error_density)
    println("\n L2 velocity error");
    show(L2error_velocity)

    #plot
    if (show_plots)
        pygui(true)
        
        # evaluate velocity and pressure at grid points
        velo = FESolveCommon.eval_at_nodes(val4dofs,FE_velocity);
        density = FESolveCommon.eval_at_nodes(val4dofs,FE_densitypressure,FiniteElements.get_ndofs(FE_velocity));
        speed = sqrt.(sum(velo.^2, dims = 2))
        #pressure = FESolveCommon.eval_at_nodes(val4dofs,FE_pressure,FiniteElements.get_ndofs(FE_velocity));

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
