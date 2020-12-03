#= 

# 2D Hagen-Poiseuille (Navier--Stokes)
([source code](SOURCE_URL))

This example computes a velocity ``\mathbf{u}`` and pressure ``\mathbf{p}`` of the incompressible Navier--Stokes problem
```math
\begin{aligned}
- \mu \Delta \mathbf{u} + (\mathbf{u} \cdot \nabla) \mathbf{u} + \nabla p & = \mathbf{f}\\
\mathrm{div}(u) & = 0
\end{aligned}
```
with exterior force ``\mathbf{f}`` and some viscosity parameter ``\mu``.

Here we solve the simple Hagen-Poiseuille flow on the two-dimensional unit square domain on a series of uniform refined meshes.

After that the errors of the Stokes solution, interpolations and bestapproximation into the FESpace is printed for easy comparison.
(Note that second order schemes are exact in this example and that the mesh is structured.)
=#

module Example_2DHagenPoiseuille

using GradientRobustMultiPhysics
using ExtendableGrids
using Printf


## data for Hagen-Poiseuille flow
function exact_pressure!(viscosity)
    function closure(result,x)
        result[1] = viscosity*(-2*x[1]+1.0)
    end
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

## everything is wrapped in a main function
function main(; verbosity = 1, Plotter = nothing, nonlinear = false)

    ## initial grid
    ## replace Parallelogram2D by Triangle2D if you like
    xgrid = uniform_refine(grid_unitsquare(Parallelogram2D),0);

    ## problem parameters
    viscosity = 1.0
    nlevels = 4 # maximal number of refinement levels

    ## choose one of these (inf-sup stable) finite element type pairs
    #FETypes = [H1P2{2,2}, H1P1{1}] # Taylor--Hood
    #FETypes = [H1P2B{2,2}, L2P1{1}] # P2-bubble
    #FETypes = [H1CR{2}, L2P0{1}] # Crouzeix--Raviart
    #FETypes = [H1MINI{2,2}, H1P1{1}] # MINI element on triangles only
    FETypes = [H1BR{2}, L2P0{1}] # Bernardi--Raugel
 
    ## solver parameters for nonlinear solve
    maxIterations = 20  # termination criterion 1 for nonlinear mode
    maxResidual = 1e-10 # termination criterion 2 for nonlinear mode

    #####################################################################################    
    #####################################################################################

    ## negotiate data functions to the package
    user_function_velocity = DataFunction(exact_velocity!, [2,2]; dependencies = "X", quadorder = 2)
    user_function_pressure = DataFunction(exact_pressure!(viscosity), [1,2]; dependencies = "X", quadorder = 1)
    user_function_velocity_gradient = DataFunction(exact_velocity_gradient!, [4,2]; dependencies = "X", quadorder = 1)

    ## load Stokes problem prototype and assign data
    StokesProblem = IncompressibleNavierStokesProblem(2; viscosity = viscosity, nonlinear = nonlinear)
    if nonlinear
        ## store matrix of Laplace operator for nonlinear solver
        StokesProblem.LHSOperators[1,1][1].store_operator = true
    end
    add_boundarydata!(StokesProblem, 1, [1,3], HomogeneousDirichletBoundary)
    add_boundarydata!(StokesProblem, 1, [2,4], BestapproxDirichletBoundary; data = user_function_velocity)
    Base.show(StokesProblem)

    ## define bestapproximation problems
    L2VelocityBestapproximationProblem = L2BestapproximationProblem(user_function_velocity; bestapprox_boundary_regions = [1,2,3,4])
    L2PressureBestapproximationProblem = L2BestapproximationProblem(user_function_pressure; bestapprox_boundary_regions = [])
    H1VelocityBestapproximationProblem = H1BestapproximationProblem(user_function_velocity_gradient, user_function_velocity; bestapprox_boundary_regions = [1,2,3,4])

    ## define ItemIntegrators for L2/H1 error computation and arrays to store them
    L2VelocityErrorEvaluator = L2ErrorIntegrator(Float64, user_function_velocity, Identity)
    L2PressureErrorEvaluator = L2ErrorIntegrator(Float64, user_function_pressure, Identity)
    H1VelocityErrorEvaluator = L2ErrorIntegrator(Float64, user_function_velocity_gradient, Gradient)
    L2error_velocity = []; L2error_pressure = []; L2errorInterpolation_velocity = []; NDofs = []
    L2errorInterpolation_pressure = []; L2errorBestApproximation_velocity = []; L2errorBestApproximation_pressure = []
    H1error_velocity = []; H1errorInterpolation_velocity = []; H1errorBestApproximation_velocity = []
    
    ## loop over levels
    for level = 1 : nlevels

        ## uniform mesh refinement
        if (level > 1) 
            xgrid = uniform_refine(xgrid)
        end

        ## generate FESpaces
        FESpaceVelocity = FESpace{FETypes[1]}(xgrid)
        FESpacePressure = FESpace{FETypes[2]}(xgrid)

        ## solve Stokes problem
        Solution = FEVector{Float64}("Stokes velocity",FESpaceVelocity)
        append!(Solution,"Stokes pressure",FESpacePressure)
        solve!(Solution, StokesProblem; verbosity = verbosity, linsolver = IterativeBigStabl_LUPC, maxlureuse = [10], maxIterations = maxIterations, maxResidual = maxResidual)
        push!(NDofs,length(Solution.entries))

        ## interpolate
        Interpolation = FEVector{Float64}("Interpolation velocity",FESpaceVelocity)
        append!(Interpolation,"Interpolation pressure",FESpacePressure)
        interpolate!(Interpolation[1], user_function_velocity)
        interpolate!(Interpolation[2], user_function_pressure)

        ## solve bestapproximation problems
        L2VelocityBestapproximation = FEVector{Float64}("L2-Bestapproximation velocity",FESpaceVelocity)
        L2PressureBestapproximation = FEVector{Float64}("L2-Bestapproximation pressure",FESpacePressure)
        H1VelocityBestapproximation = FEVector{Float64}("H1-Bestapproximation velocity",FESpaceVelocity)
        solve!(L2VelocityBestapproximation, L2VelocityBestapproximationProblem)
        solve!(L2PressureBestapproximation, L2PressureBestapproximationProblem)
        solve!(H1VelocityBestapproximation, H1VelocityBestapproximationProblem)

        ## compute L2 and H1 error
        append!(L2error_velocity,sqrt(evaluate(L2VelocityErrorEvaluator,Solution[1])))
        append!(L2errorInterpolation_velocity,sqrt(evaluate(L2VelocityErrorEvaluator,Interpolation[1])))
        append!(L2errorBestApproximation_velocity,sqrt(evaluate(L2VelocityErrorEvaluator,L2VelocityBestapproximation[1])))
        append!(L2error_pressure,sqrt(evaluate(L2PressureErrorEvaluator,Solution[2])))
        append!(L2errorInterpolation_pressure,sqrt(evaluate(L2PressureErrorEvaluator,Interpolation[2])))
        append!(L2errorBestApproximation_pressure,sqrt(evaluate(L2PressureErrorEvaluator,L2PressureBestapproximation[1])))
        append!(H1error_velocity,sqrt(evaluate(H1VelocityErrorEvaluator,Solution[1])))
        append!(H1errorInterpolation_velocity,sqrt(evaluate(H1VelocityErrorEvaluator,Interpolation[1])))
        append!(H1errorBestApproximation_velocity,sqrt(evaluate(H1VelocityErrorEvaluator,H1VelocityBestapproximation[1])))
        
        ## plot and ouput errors
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
            println("VELO-STOKES : discrete Stokes velocity solution ($(FESpaceVelocity.name))")
            println("VELO-INTERP : interpolation of exact velocity")
            println("VELO-L2BEST : L2-Bestapproximation of exact velocity (with boundary data)")
            println("VELO-H1BEST : H1-Bestapproximation of exact velocity (with boudnary data)")
            println("PRES-STOKES : discrete Stokes pressure solution ($(FESpacePressure.name))")
            println("PRES-INTERP : interpolation of exact pressure")
            println("PRES-L2BEST : L2-Bestapproximation of exact pressure (without boundary data)")

            ## plot
            GradientRobustMultiPhysics.plot(Solution, [1,2], [Identity, Identity]; Plotter = Plotter, verbosity = verbosity, use_subplots = true)
            
        end    
    end    
end

end