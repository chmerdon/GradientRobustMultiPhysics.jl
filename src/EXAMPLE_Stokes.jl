
using FEXGrid
using ExtendableGrids
using ExtendableSparse
using FiniteElements
using FEAssembly
using PDETools
using QuadratureRules
#using VTKView
ENV["MPLBACKEND"]="qt5agg"
using PyPlot
using Printf


include("testgrids.jl")

# data for Hagen-Poiseuille flow
viscosity = 1.0

function exact_pressure!(result,x)
    result[1] = viscosity*(-2*x[1]+1.0)
end
function exact_velocity!(result,x)
    result[1] = x[2]*(1.0-x[2]);
    result[2] = 0.0;
end
function exact_velocity_gradient!(result,x)
    result[1] = 0.0
    result[2] = (1.0-2.0*x[2]);
    result[3] = 0.0;
    result[4] = 0.0;
end
function bnd_data_rest!(result,x)
    result[1] = 0.0
    result[2] = 0.0
end    
function exact_divergence!(result,x)
    result[1] = 0.0
end    


function main()
    #####################################################################################
    #####################################################################################

    # meshing parameters
    xgrid = testgrid_mixedEG(); # initial grid
    #xgrid = split_grid_into(xgrid,Triangle2D) # if you want just triangles
    nlevels = 5 # number of refinement levels

    # problem parameters
    viscosity = 1e-2
    nonlinear = true

    # fem/solver parameters
    fem = "BR" # Bernardi--Raugel
    #fem = "CR" # Crouzeix--Raviart
    #fem = "MINI" # MINI element
    #fem = "TH" # Taylor--Hood
    maxIterations = 10  # termination criterion 1 for nonlinear mode
    maxResidual = 1e-12 # termination criterion 2 for nonlinear mode
    verbosity = 1 # deepness of messaging (the larger, the more)

    #####################################################################################    
    #####################################################################################

    # load Stokes problem prototype and assign data
    StokesProblem = IncompressibleStokesProblem(2; viscosity = viscosity, nonlinear = nonlinear)
    append!(StokesProblem.BoundaryOperators[1], [1,3], HomogeneousDirichletBoundary)
    append!(StokesProblem.BoundaryOperators[1], [2,4], BestapproxDirichletBoundary; data = exact_velocity!, bonus_quadorder = 2)

    # define bestapproximation problems
    L2VelocityBestapproximationProblem = L2BestapproximationProblem(exact_velocity!, 2, 2; bestapprox_boundary_regions = [1,2,3,4], bonus_quadorder = 2)
    L2PressureBestapproximationProblem = L2BestapproximationProblem(exact_pressure!, 2, 1; bestapprox_boundary_regions = [], bonus_quadorder = 1)
    H1VelocityBestapproximationProblem = H1BestapproximationProblem(exact_velocity_gradient!, exact_velocity!, 2, 2; bestapprox_boundary_regions = [1,2,3,4], bonus_quadorder = 1, bonus_quadorder_boundary = 2)

    # define ItemIntegrators for L2/H1 error computation
    L2VelocityErrorEvaluator = L2ErrorIntegrator(exact_velocity!, Identity, 2, 2; bonus_quadorder = 2)
    L2PressureErrorEvaluator = L2ErrorIntegrator(exact_pressure!, Identity, 2, 1; bonus_quadorder = 1)
    H1VelocityErrorEvaluator = L2ErrorIntegrator(exact_velocity_gradient!, Gradient, 2, 4; bonus_quadorder = 1)
    L2error_velocity = []
    L2error_pressure = []
    L2errorInterpolation_velocity = []
    L2errorInterpolation_pressure = []
    L2errorBestApproximation_velocity = []
    L2errorBestApproximation_pressure = []
    H1error_velocity = []
    H1errorInterpolation_velocity = []
    H1errorBestApproximation_velocity = []
    NDofs = []
    
    # loop over levels
    for level = 1 : nlevels

        # uniform mesh refinement
        if (level > 1) 
            xgrid = uniform_refine(xgrid)
        end

        # generate FE
        if fem == "BR" # Bernardi--Raugel
            FE_velocity = FiniteElements.getH1BRFiniteElement(xgrid)
            FE_pressure = FiniteElements.getP0FiniteElement(xgrid,1)
        elseif fem == "MINI" # MINI element
            FE_velocity = FiniteElements.getH1MINIFiniteElement(xgrid,2)
            FE_pressure = FiniteElements.getH1P1FiniteElement(xgrid,1)
        elseif fem == "CR" # Crouzeix--Raviart
            FE_velocity = FiniteElements.getH1CRFiniteElement(xgrid,2)
            FE_pressure = FiniteElements.getP0FiniteElement(xgrid,1)
        elseif fem == "TH" # Taylor--Hood/Q2xP1
            FE_velocity = FiniteElements.getH1P2FiniteElement(xgrid,2)
            FE_pressure = FiniteElements.getH1P1FiniteElement(xgrid,1)
        end        
        if verbosity > 2
            FiniteElements.show(FE_velocity)
            FiniteElements.show(FE_pressure)
        end    

        # solve Stokes problem
        Solution = FEVector{Float64}("Stokes velocity",FE_velocity)
        append!(Solution,"Stokes pressure",FE_pressure)
        solve!(Solution, StokesProblem; verbosity = verbosity, maxIterations = maxIterations, maxResidual = maxResidual)
        push!(NDofs,length(Solution.entries))

        # interpolate
        Interpolation = FEVector{Float64}("Interpolation velocity",FE_velocity)
        append!(Interpolation,"Interpolation pressure",FE_pressure)
        interpolate!(Interpolation[1], exact_velocity!; verbosity = verbosity, bonus_quadorder = 2)
        interpolate!(Interpolation[2], exact_pressure!; verbosity = verbosity, bonus_quadorder = 1)

        # solve bestapproximation problems
        L2VelocityBestapproximation = FEVector{Float64}("L2-Bestapproximation velocity",FE_velocity)
        L2PressureBestapproximation = FEVector{Float64}("L2-Bestapproximation pressure",FE_pressure)
        H1VelocityBestapproximation = FEVector{Float64}("H1-Bestapproximation velocity",FE_velocity)
        solve!(L2VelocityBestapproximation, L2VelocityBestapproximationProblem; verbosity = verbosity)
        solve!(L2PressureBestapproximation, L2PressureBestapproximationProblem; verbosity = verbosity)
        solve!(H1VelocityBestapproximation, H1VelocityBestapproximationProblem; verbosity = verbosity)

        # compute L2 and H1 error
        append!(L2error_velocity,sqrt(evaluate(L2VelocityErrorEvaluator,Solution[1])))
        append!(L2errorInterpolation_velocity,sqrt(evaluate(L2VelocityErrorEvaluator,Interpolation[1])))
        append!(L2errorBestApproximation_velocity,sqrt(evaluate(L2VelocityErrorEvaluator,L2VelocityBestapproximation[1])))
        append!(L2error_pressure,sqrt(evaluate(L2PressureErrorEvaluator,Solution[2])))
        append!(L2errorInterpolation_pressure,sqrt(evaluate(L2PressureErrorEvaluator,Interpolation[2])))
        append!(L2errorBestApproximation_pressure,sqrt(evaluate(L2PressureErrorEvaluator,L2PressureBestapproximation[1])))
        append!(H1error_velocity,sqrt(evaluate(H1VelocityErrorEvaluator,Solution[1])))
        append!(H1errorInterpolation_velocity,sqrt(evaluate(H1VelocityErrorEvaluator,Interpolation[1])))
        append!(H1errorBestApproximation_velocity,sqrt(evaluate(H1VelocityErrorEvaluator,H1VelocityBestapproximation[1])))
        
        # plot final solution
        if (level == nlevels)
            println("\n         |   L2ERROR   |   L2ERROR   |   L2ERROR")
            println("   NDOF  | VELO-STOKES | VELO-INTERP | VELO-L2BEST");
            for j=1:nlevels
                @printf("  %6d |",NDofs[j]);
                @printf(" %.5e |",L2error_velocity[j])
                @printf(" %.5e |",L2errorInterpolation_velocity[j])
                @printf(" %.5e\n",L2errorBestApproximation_velocity[j])
            end
            println("\n         |   H1ERROR   |   H1ERROR   |   H1ERROR")
            println("   NDOF  | VELO-STOKES | VELO-INTERP | VELO-H1BEST");
            for j=1:nlevels
                @printf("  %6d |",NDofs[j]);
                @printf(" %.5e |",H1error_velocity[j])
                @printf(" %.5e |",H1errorInterpolation_velocity[j])
                @printf(" %.5e\n",H1errorBestApproximation_velocity[j])
            end
            println("\n         |   L2ERROR   |   L2ERROR   |   L2ERROR")
            println("   NDOF  | PRES-STOKES | PRES-INTERP | PRES-L2BEST");
            for j=1:nlevels
                @printf("  %6d |",NDofs[j]);
                @printf(" %.5e |",L2error_pressure[j])
                @printf(" %.5e |",L2errorInterpolation_pressure[j])
                @printf(" %.5e\n",L2errorBestApproximation_pressure[j])
            end
            println("\nLEGEND\n======")
            println("VELO-STOKES : discrete Stokes velocity solution ($(FE_velocity.name))")
            println("VELO-INTERP : interpolation of exact velocity")
            println("VELO-L2BEST : L2-Bestapproximation of exact velocity (with boundary data)")
            println("VELO-H1BEST : H1-Bestapproximation of exact velocity (with boudnary data)")
            println("PRES-STOKES : discrete Stokes pressure solution ($(FE_pressure.name))")
            println("PRES-INTERP : interpolation of exact pressure")
            println("PRES-L2BEST : L2-Bestapproximation of exact pressure (without boundary data)")
            # split grid into triangles for plotter
            xgrid = split_grid_into(xgrid,Triangle2D)

            # plot triangulation
            PyPlot.figure(1)
            ExtendableGrids.plot(xgrid, Plotter = PyPlot)

            # plot solution
            nnodes = size(xgrid[Coordinates],2)
            nodevals = zeros(Float64,2,nnodes)
            nodevalues!(nodevals,Solution[1],FE_velocity)
            PyPlot.figure(2)
            ExtendableGrids.plot(xgrid, nodevals[1,:][1:nnodes]; Plotter = PyPlot)
            PyPlot.figure(3)
            nodevalues!(nodevals,Solution[2],FE_pressure)
            ExtendableGrids.plot(xgrid, nodevals[1,:]; Plotter = PyPlot)
        end    
    end    


end


main()
