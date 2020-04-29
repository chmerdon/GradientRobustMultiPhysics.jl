#############################################
### DEMONSTRATION SCRIPT STOKES P7-VORTEX ###
#############################################
#
# solves Stokes test problem p7vortex for multiple viscosities
# and reconstruction operators
#
# demonstrates:
#   - lack/presence of pressure-robustness for small nu
#

using Triangulate
using Grid
using Quadrature
using FiniteElements
using FESolveCommon
using FESolveStokes
ENV["MPLBACKEND"]="tkagg"
using PyPlot

# load problem data and common grid generator
include("PROBLEMdefinitions/GRID_unitsquare.jl")
include("PROBLEMdefinitions/STOKES_p7vortex.jl");


function main()

    # problem modification switches
    nu = [1e1,1,1e-1,1e-2,1e-3,1e-4]

    # refinement termination criterions
    maxlevel = 4
    maxdofs = 10000

    # other switches
    show_plots = true
    show_convergence_history = true
    uniform_mesh = true
    use_reconstruction = [0] # do not change here
    barycentric_refinement = false # do not change here
    use_square_grid = false # do not change here


    ########################
    ### CHOOSE FEM BELOW ###
    ########################

    #fem_velocity = "CR"; fem_pressure = "P0"; use_reconstruction = [0,1,2]
    #fem_velocity = "MINI"; fem_pressure = "P1"
    #fem_velocity = "P2";  fem_pressure = "P1"
    #fem_velocity = "P2";  fem_pressure = "P1dc"; barycentric_refinement = true
    #fem_velocity = "P2"; fem_pressure = "P0"
    #fem_velocity = "P2B"; fem_pressure = "P1dc"
    fem_velocity = "BR"; fem_pressure = "P0"; use_reconstruction = [0,1,2]

    ### elements on square grids
    #fem_velocity = "Q1"; fem_pressure = "P0"; expectedorder = 1; use_reconstruction = [0,1]; use_square_grid = true
    #fem_velocity = "BR"; fem_pressure = "P0"; expectedorder = 1; use_reconstruction = [0,1]; use_square_grid = true


    L2error_velocity = zeros(Float64,maxlevel,length(nu),length(use_reconstruction))
    L2error_pressure = zeros(Float64,maxlevel,length(nu),length(use_reconstruction))
    ndofs = zeros(Int,maxlevel)
    grid = Nothing
    FE_velocity = Nothing
    FE_pressure = Nothing
    val4dofs = Nothing


    for n = 1 : length(nu)
        # load problem data
        PD, exact_velocity!, exact_pressure! = getProblemData(nu[n], 4);
        FESolveStokes.show(PD);

        for level = 1 : maxlevel

            println("Solving Stokes problem on refinement level...", level);
            println("Generating grid by triangle...");
            maxarea = 4.0^(-level)
            if use_square_grid == true
                grid = gridgen_unitsquare_squares(maxarea,0.4,0.6)
            else
                if uniform_mesh == true
                    grid = gridgen_unitsquare_uniform(maxarea, true, true)    
                else    
                    grid = gridgen_unitsquare(maxarea, barycentric_refinement)
                end    
            end 
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
                    if use_square_grid == true
                        grid = gridgen_unitsquare_squares(maxarea,0.4,0.6)
                    else
                        if uniform_mesh == true
                            grid = gridgen_unitsquare_uniform(maxarea, true, true)    
                        else        
                            grid = gridgen_unitsquare(maxarea, barycentric_refinement)
                        end    
                    end 
                    FE_velocity = FiniteElements.string2FE(fem_velocity,grid,2,2)
                    FE_pressure = FiniteElements.string2FE(fem_pressure,grid,2,1)
                end    
                break
            end
            
            for r = 1 : length(use_reconstruction)
                
                # solve Stokes problem
                val4dofs = zeros(Base.eltype(grid.coords4nodes),ndofs[level]);
                residual = solveStokesProblem!(val4dofs,PD,FE_velocity,FE_pressure, reconst_variant = use_reconstruction[r]);

                # compute errors
                L2error_pressure[level,n,r] = sqrt(FESolveCommon.assemble_operator!(FESolveCommon.DOMAIN_L2_FplusA,exact_pressure!,FE_pressure ,val4dofs[ndofs_velocity+1:end]; degreeF = 3, factorA = -1.0))
                L2error_velocity[level,n,r] = sqrt(FESolveCommon.assemble_operator!(FESolveCommon.DOMAIN_L2_FplusA,exact_velocity!,FE_velocity ,val4dofs; degreeF = 7, factorA = -1.0))


            end # loop over reconstruction
        end # loop over levels
    end # loop over nu

    println("\n L2 pressure error");
    show(L2error_pressure)
    println("\n L2 velocity error");
    show(L2error_velocity)

    #plot
    if (show_plots)
        pygui(true)
        
        # evaluate velocity and pressure at grid points
        velo = FESolveCommon.eval_at_nodes(val4dofs,FE_velocity);
        pressure = FESolveCommon.eval_at_nodes(val4dofs,FE_pressure,FiniteElements.get_ndofs(FE_velocity));

        PyPlot.figure(1)
        PyPlot.plot_trisurf(view(grid.coords4nodes,:,1),view(grid.coords4nodes,:,2),view(velo,:,1),cmap=get_cmap("ocean"))
        PyPlot.title("Stokes Problem Solution - velocity component 1")
        PyPlot.figure(2)
        PyPlot.plot_trisurf(view(grid.coords4nodes,:,1),view(grid.coords4nodes,:,2),view(velo,:,2),cmap=get_cmap("ocean"))
        PyPlot.title("Stokes Problem Solution - velocity component 2")
        PyPlot.figure(3)
        PyPlot.plot_trisurf(view(grid.coords4nodes,:,1),view(grid.coords4nodes,:,2),pressure[:],cmap=get_cmap("ocean"))
        PyPlot.title("Stokes Problem Solution - pressure")
        show()
    end

    if (show_convergence_history)
        for r = 1 : length(use_reconstruction)
            PyPlot.figure(r)
            labels = [];
            for n = 1 : length(nu)
                PyPlot.loglog(ndofs[1:maxlevel],L2error_velocity[1:maxlevel,n,r],"-o")
                #PyPlot.loglog(ndofs[1:maxlevel],L2error_pressure[1:maxlevel,n,r],"-o")
                append!(labels,["nu = " * string(nu[n])])
            end
            PyPlot.loglog(ndofs,ndofs.^(-1/2),"--",color = "gray")
            PyPlot.loglog(ndofs,ndofs.^(-1),"--",color = "gray")
            PyPlot.loglog(ndofs,ndofs.^(-3/2),"--",color = "gray")
            append!(labels,["O(h)","O(h^2)","O(h^3)"])
            PyPlot.legend(labels)   
            PyPlot.title("Convergence history (fem=" * fem_velocity * "/" * fem_pressure * ", reconstruction = " * string(use_reconstruction[r]) *")")
            ax = PyPlot.gca()
            ax.grid(true)
        end
    end    

        
end


main()
