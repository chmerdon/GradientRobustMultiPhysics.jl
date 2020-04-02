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
using LinearAlgebra
using ExtendableSparse
using FiniteElements
using FESolveCommon
using FESolveStokes
using FESolveNavierStokes
using VTKView

# load problem data and common grid generator
include("PROBLEMdefinitions/GRID_square.jl")
include("PROBLEMdefinitions/STOKES_GreshoVortex.jl");

function main()

    # problem configuration
    nu = 2e-4
    reflevel = 5
    dt = 0.01
    final_time = 0.2
    nonlinear = true
    timesteps::Int64 = floor(final_time / dt)
    u_order = 10
    error_order = 10

    # other switches
    show_plots = true
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
    end

    # plot
    scalarview = Array{VTKView.ScalarView,1}(undef,length(use_reconstruction))
    vectorview = Array{VTKView.VectorView,1}(undef,length(use_reconstruction))
    velo = zeros(Float64,size(FE_velocity.grid.coords4nodes,1),2)
    speed = zeros(Float64,size(FE_velocity.grid.coords4nodes,1),1)
    dataset=VTKView.DataSet()
    xyplot=VTKView.XYPlot()
    if (show_plots)
        frame=VTKView.StaticFrame()
        frametitle!(frame,"t = " * string(TSS[1].current_time))
        clear!(frame)
        layout!(frame,length(use_reconstruction)+1,1)
        size!(frame,1500,500)
        VTKView.simplexgrid!(dataset,Array{Float64,2}(grid.coords4nodes'),Array{Int32,2}(grid.nodes4cells'))

        # scalar view
        for k = 1 : length(use_reconstruction)
            scalarview[k]=VTKView.ScalarView()
            velo = FESolveCommon.eval_at_nodes(val4dofs[k],FE_velocity);
            speed = sqrt.(sum(velo.^2, dims = 2))
            pointscalar!(dataset,speed[:],"|U" * string(use_reconstruction[k]) *"|")
            data!(scalarview[k],dataset,"|U" * string(use_reconstruction[k]) *"|")
            addview!(frame,scalarview[k],k)
        
            vectorview[k]=VTKView.VectorView()
            pointvector!(dataset,Array{Float64,2}(velo'),"U" * string(use_reconstruction[k]))
            data!(vectorview[k],dataset,"U" * string(use_reconstruction[k]))
            quiver!(vectorview[k],10,10)
            addview!(frame,vectorview[k],k)
        end  

         # XY plot
         addview!(frame,xyplot,length(use_reconstruction)+1)
         clear!(xyplot)
         
        # show
        display(frame)
    end    

    # compute mass matrix for energy calculation
    MassMatrix = ExtendableSparseMatrix{Float64,Int64}(ndofs,ndofs) # +1 due to Lagrange multiplier for integral mean    
    FESolveCommon.assemble_operator!(MassMatrix,FESolveCommon.CELL_UdotV,FE_velocity)
    ExtendableSparse.flush!(MassMatrix)

    for j = 0 : timesteps
        append!(energy_times,TSS[1].current_time)
    
        for k = 1 : length(use_reconstruction)
            
            # compute errors
            append!(velocity_energy[k],sqrt.(sum(val4dofs[k].*(MassMatrix.cscmatrix*val4dofs[k]), dims = 1)))

            if nonlinear == false
                FESolveStokes.PerformTimeStep(TSS[k],dt)
            else
                FESolveNavierStokes.PerformIMEXTimeStep(TSS[k],dt)
            end
            val4dofs[k] = deepcopy(TSS[k].current_solution[:])

            #plot
            if (k == length(use_reconstruction)) && (show_plots)
                frametitle!(frame,"t = " * string(TSS[1].current_time))
                # update scalar view
                for k = 1 : length(use_reconstruction)
                    velo[:] = FESolveCommon.eval_at_nodes(val4dofs[k],FE_velocity);
                    speed[:] = sqrt.(sum(velo.^2, dims = 2))
                    pointscalar!(dataset,speed[:],"|U" * string(use_reconstruction[k]) *"|")
                    
                    pointvector!(dataset,Array{Float64,2}(velo'),"U" * string(use_reconstruction[k]))
                    data!(vectorview[k],dataset,"U" * string(use_reconstruction[k]))
                    quiver!(vectorview[k],10,10)

                    clear!(xyplot)
                    for k = 1 : length(use_reconstruction)
                        plotlegend!(xyplot,"Energy ( " * string(use_reconstruction[k]) * ")")
                        plotcolor!(xyplot,(k == 1) ? 1 : 0,(k == 2) ? 1 : 0,(k == 3) ? 1 : 0)
                        addplot!(xyplot,Array{Float64,1}(energy_times[:]),velocity_energy[k][:]) 
                    end
                end    

                # show
                display(frame)
            end    
        end
    end    

end


main()
