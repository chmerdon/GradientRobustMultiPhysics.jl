##########################################
### DEMONSTRATION SCRIPT GRESH0-VORTEX ###
##########################################
#
# solves transient Gresho vortex test Problem
#
# demonstrates:
#   - setup and run of a transient Stokes solver
#   - benefit of pressure-robustness (usre_reconstruction > 0) in material derivative#
#     (concerning shape preservation and energy decay)
#

using Triangulate
using Grid
using Quadrature
using FiniteElements
using FESolveCommon
using FESolveStokes
using FESolveNavierStokes
ENV["MPLBACKEND"]="tkagg"
using PyPlot

# load problem data and common grid generator
include("PROBLEMdefinitions/GRID_square.jl")
include("PROBLEMdefinitions/STOKES_GreshoVortex.jl");

function main()

    # problem configuration
    nu = 2e-4
    reflevel = 5
    dt = 0.01
    final_time = 0.25
    nonlinear = true
    timesteps::Int64 = floor(final_time / dt)
    energy_computation_gaps = 2
    u_order = 10
    error_order = 10

    # other switches
    show_plots = true
    show_convergence_history = true
    use_reconstruction = [0] # do not change here
    barycentric_refinement = false # do not change here


    ########################
    ### CHOOSE FEM BELOW ###
    ########################

    #fem_velocity = "CR"; fem_pressure = "P0"
    #fem_velocity = "MINI"; fem_pressure = "P1"
    #fem_velocity = "P2";  fem_pressure = "P1"
    #fem_velocity = "P2";  fem_pressure = "P1dc"; barycentric_refinement = true
    #fem_velocity = "P2"; fem_pressure = "P0"
    #fem_velocity = "P2B"; fem_pressure = "P1dc"
    #fem_velocity = "BR"; fem_pressure = "P0"
    fem_velocity = "BR"; fem_pressure = "P0";  use_reconstruction = [0,1]


    # load problem data
    PD, exact_velocity! = getProblemData(nu,4);


    println("Solving transient Navier-Stokes problem on refinement level...", reflevel);
    println("Generating grid by triangle...");
    maxarea = 4.0^(-reflevel)
    grid = gridgen_unitsquare(maxarea, barycentric_refinement)
    Grid.show(grid)

    # load finite element
    FE_velocity = FiniteElements.string2FE(fem_velocity,grid,2,2)
    FE_pressure = FiniteElements.string2FE(fem_pressure,grid,2,1)
    FiniteElements.show(FE_velocity)
    FiniteElements.show(FE_pressure)
    ndofs_velocity = FiniteElements.get_ndofs(FE_velocity);
    ndofs_pressure = FiniteElements.get_ndofs(FE_pressure);
    ndofs = ndofs_velocity + ndofs_pressure;

    # solve for initial value by best approximation 
    val4dofs = Array{Array{Float64,1}}(undef,length(use_reconstruction))
    for k= 1 : length(use_reconstruction)
        val4dofs[k] = zeros(Base.eltype(grid.coords4nodes),ndofs);
    end    
    residual = FESolveStokes.computeDivFreeBestApproximation!(val4dofs[1],exact_velocity!,exact_velocity!,FE_velocity,FE_pressure,u_order)
    for k= 2 : length(use_reconstruction)
        val4dofs[k] = deepcopy(val4dofs[1])
    end    

    function zero_data!(result,x)
        fill!(result,0.0)
    end
    
    TSS = Array{FESolveStokes.TransientStokesSolver,1}(undef,length(use_reconstruction))
    velocity_energy = Array{Array{Float64,1},1}(undef,length(use_reconstruction))
    energy_times = []

    for k = 1 : length(use_reconstruction)
        velocity_energy[k] = []
        TSS[k] = FESolveStokes.setupTransientStokesSolver(PD,FE_velocity,FE_pressure,val4dofs[k],use_reconstruction[k])
  
        if (show_plots)
            pygui(true)
            
            # evaluate velocity and pressure at grid points
            velo = FESolveCommon.eval_at_nodes(val4dofs[k],FE_velocity);
            speed = sqrt.(sum(velo.^2, dims = 2))
            
            PyPlot.figure(k)
            tcf = PyPlot.tricontourf(view(grid.coords4nodes,:,1),view(grid.coords4nodes,:,2),speed[:])
            PyPlot.axis("equal")
            PyPlot.title("velocity speed (reconst = " * string(use_reconstruction[k]) * ")")
            PyPlot.colorbar(tcf)
        end      
    end

    

    for j = 0 : timesteps
        if mod(j,energy_computation_gaps) == 0
            append!(energy_times,TSS[1].current_time)
        end

        for k = 1 : length(use_reconstruction)
            
            if mod(j,energy_computation_gaps) == 0
                println("computing errors")
                # compute errors
                integral4cells = zeros(size(grid.nodes4cells,1),1);
                integral4cells = zeros(size(grid.nodes4cells,1),2);
                integrate!(integral4cells,eval_L2_interpolation_error!(zero_data!, val4dofs[k][1:ndofs_velocity], FE_velocity), grid, error_order, 2);
                append!(velocity_energy[k],sqrt(abs(sum(integral4cells[:]))));
            end    

            if nonlinear == false
                FESolveStokes.PerformTimeStep(TSS[k],dt)
            else
                FESolveNavierStokes.PerformIMEXTimeStep(TSS[k],dt)
            end
            val4dofs[k] = deepcopy(TSS[k].current_solution[:])

            #plot
            if (show_plots)
                pygui(true)
                
                # evaluate velocity and pressure at grid points
                velo = FESolveCommon.eval_at_nodes(val4dofs[k],FE_velocity);
                speed = sqrt.(sum(velo.^2, dims = 2))
                #pressure = FESolveCommon.eval_at_nodes(val4dofs,FE_pressure,FiniteElements.get_ndofs(FE_velocity));

                PyPlot.figure(k)
                tcf = PyPlot.tricontourf(view(grid.coords4nodes,:,1),view(grid.coords4nodes,:,2),speed[:])
                PyPlot.axis("equal")
                PyPlot.title("velocity speed (reconst = " * string(use_reconstruction[k]) * ")")
                show()
            end    
        end
    end    

    #Base.show(energy_times)
    #Base.show(velocity_energy)

    if (show_convergence_history)
        PyPlot.figure()
        labels = []
        for k = 1 : length(use_reconstruction)
            PyPlot.loglog(energy_times[:],velocity_energy[k][:],"-o")
            append!(labels,["Energy reconst = " * string(use_reconstruction[k])])
        end     
        PyPlot.legend(labels)
        ax = PyPlot.gca()
        ax.grid(true)
    end    

end


main()
