#= 

# 2D Pressure-robustness
([source code](SOURCE_URL))

This example studies two benchmarks for pressure-robust discretisations of the stationary
    Navier-Stokes equations that seek a velocity ``\mathbf{u}`` and pressure ``\mathbf{p}`` such that
```math
\begin{aligned}
- \mu \Delta \mathbf{u} + (\mathbf{u} \cdot \nabla) \mathbf{u} + \nabla p & = \mathbf{f}\\
\mathrm{div}(u) & = 0
\end{aligned}
```
with (possibly time-dependent) exterior force ``\mathbf{f}`` and some viscosity parameter ``\mu``.

Pressure-robustness is concerned with gradient forces that may appear in the right-hand side or the material derivative and
should be balanced by the pressure (as divergence-free vector fields are orthogonal on gradient fields). Here, two test problems
are considered:

1. HydrostaticTestProblem() : Stokes (without convection term) and ``\mathbf{f} = \nabla p`` such that  ``\mathbf{u} = 0``
2. PotentialFlowTestProblem() : Navier-Stokes with ``\mathbf{f} = 0`` and ``\mathbf{u} = \nabla h`` for some harmonic function

In both test problems the errors of non-pressure-robust discretisations scale with  ``1/\mu``, while the pressure-robust
discretisation solves ``\mathbf{u} = 0`` exactly in test problem 1 and gives much better results in test problem 2.

=#


module Example_2DPressureRobustness

using GradientRobustMultiPhysics
using Printf

## problem data
function HydrostaticTestProblem()
    ## Stokes problem with f = grad(p)
    ## u = 0
    ## p = x^3+y^3 - 1//2
    function P1_pressure!(result,x::Array{<:Real,1})
        result[1] = x[1]^3 + x[2]^3 - 1//2
    end
    function P1_velo!(result)
        result[1] = 0;
        result[2] = 0;
    end
    function P1_velogradient!(result)
        result[1] = 0
        result[2] = 0;
        result[3] = 0;
        result[4] = 0;
    end
    function P1_rhs!(result,x::Array{<:Real,1})
        result[1] = 3*x[1]^2
        result[2] = 3*x[2]^2
    end
    user_function_velocity = DataFunction(P1_velo!, [2,2]; dependencies = "", quadorder = 0)
    user_function_pressure = DataFunction(P1_pressure!, [1,2]; dependencies = "X", quadorder = 3)
    user_function_velocity_gradient = DataFunction(P1_velogradient!, [4,2]; dependencies = "", quadorder = 0)
    user_function_rhs = DataFunction(P1_rhs!, [2,2]; dependencies = "X", quadorder = 2)

    return user_function_pressure,user_function_velocity,user_function_velocity_gradient,user_function_rhs, false
end

function PotentialFlowTestProblem()
    ## NavierStokes with f = 0
    ## u = grad(h) with h = x^3 - 3xy^2
    ## p = - |grad(h)|^2 + 14//5
    function P2_pressure!(result,x::Array{<:Real,1})
        result[1] = - 1//2 * (9*(x[1]^4 + x[2]^4) + 18*x[1]^2*x[2]^2) + 14//5
    end
    function P2_velo!(result,x::Array{<:Real,1})
        result[1] = 3*x[1]^2 - 3*x[2]^2;
        result[2] = -6*x[1]*x[2];
    end
    function P2_velogradient!(result,x::Array{<:Real,1})
        result[1] = 6*x[1]
        result[2] = -6*x[2];
        result[3] = -6*x[2];
        result[4] = -6*x[1];
    end
    function P2_rhs!(result)
        result[1] = 0
        result[2] = 0
    end
    user_function_velocity = DataFunction(P2_velo!, [2,2]; dependencies = "X", quadorder = 2)
    user_function_pressure = DataFunction(P2_pressure!, [1,2]; dependencies = "X", quadorder = 4)
    user_function_velocity_gradient = DataFunction(P2_velogradient!, [4,2]; dependencies = "X", quadorder = 1)
    user_function_rhs = DataFunction(P2_rhs!, [2,2]; dependencies = "", quadorder = 0)

    return user_function_pressure,user_function_velocity,user_function_velocity_gradient,user_function_rhs, true
end


function solve(Problem, xgrid, FETypes, viscosity = 1e-2; nlevels = 3, print_results = true, verbosity = 1)

    ## load problem data and set solver parameters
    exact_pressure!, exact_velocity!, exact_velocity_gradient!, rhs!, nonlinear = Problem()
    maxIterations = 20  # termination criterion 1 for nonlinear mode
    maxResidual = 1e-12 # termination criterion 2 for nonlinear mode

    ## load Stokes problem prototype and assign data
    StokesProblem = IncompressibleNavierStokesProblem(2; viscosity = viscosity, nonlinear = nonlinear)
    add_boundarydata!(StokesProblem, 1, [1,2,3,4], BestapproxDirichletBoundary; data = exact_velocity!)
    add_rhsdata!(StokesProblem, 1, RhsOperator(ReconstructionIdentity, [0], rhs!))

    ## define bestapproximation problems
    L2VelocityBestapproximationProblem = L2BestapproximationProblem(exact_velocity!; bestapprox_boundary_regions = [1,2,3,4])
    L2PressureBestapproximationProblem = L2BestapproximationProblem(exact_pressure!; bestapprox_boundary_regions = [])
    H1VelocityBestapproximationProblem = H1BestapproximationProblem(exact_velocity_gradient!, exact_velocity!; bestapprox_boundary_regions = [1,2,3,4])

    ## define ItemIntegrators for L2/H1 error computation
    L2VelocityErrorEvaluator = L2ErrorIntegrator(Float64, exact_velocity!, Identity)
    L2PressureErrorEvaluator = L2ErrorIntegrator(Float64, exact_pressure!, Identity)
    H1VelocityErrorEvaluator = L2ErrorIntegrator(Float64, exact_velocity_gradient!, Gradient)
    L2error_velocity = []; L2error_pressure = []; L2error_velocity2 = []; L2error_pressure2 = []
    L2errorInterpolation_velocity = []; L2errorInterpolation_pressure = []; L2errorBestApproximation_velocity = []; L2errorBestApproximation_pressure = []
    H1error_velocity = []; H1error_velocity2 = []; H1errorBestApproximation_velocity = []; NDofs = []
    
    ## setup classical (StokesProblem) and pressure-robust scheme (StokesProblem2)
    StokesProblem2 = deepcopy(StokesProblem)
    StokesProblem.RHSOperators[1][1] = RhsOperator(Identity, [0], rhs!)
    ReconstructionOperator = FETypes[3]
    StokesProblem2.RHSOperators[1][1] = RhsOperator(ReconstructionOperator, [0], rhs!)
    if nonlinear
        StokesProblem.LHSOperators[1,1][1].store_operator = true # store matrix of Laplace operator
        StokesProblem2.LHSOperators[1,1][1].store_operator = true # store matrix of Laplace operator
        StokesProblem.LHSOperators[1,1][2] = ConvectionOperator(1, Identity, 2, 2)
        StokesProblem2.LHSOperators[1,1][2] = ConvectionOperator(1, ReconstructionOperator, 2, 2; testfunction_operator = ReconstructionOperator)
    end
    
    ## loop over refinement levels
    for level = 1 : nlevels

        ## uniform mesh refinement
        xgrid = uniform_refine(xgrid)
        xFaceVolumes = xgrid[FaceVolumes]

        ## get FESpaces
        FESpaceVelocity = FESpace{FETypes[1]}(xgrid)
        FESpacePressure = FESpace{FETypes[2]}(xgrid)

        Solution = FEVector{Float64}("Stokes velocity classical",FESpaceVelocity)
        append!(Solution,"Stokes pressure (classical)",FESpacePressure)
        solve!(Solution, StokesProblem; maxIterations = maxIterations, maxResidual = maxResidual, verbosity = verbosity)
        push!(NDofs,length(Solution.entries))

        Solution2 = FEVector{Float64}("Stokes velocity p-robust",FESpaceVelocity)
        append!(Solution2,"Stokes pressure (p-robust)",FESpacePressure)
        solve!(Solution2, StokesProblem2; maxIterations = maxIterations, maxResidual = maxResidual)

        ## solve bestapproximation problems
        L2VelocityBestapproximation = FEVector{Float64}("L2-Bestapproximation velocity",FESpaceVelocity)
        L2PressureBestapproximation = FEVector{Float64}("L2-Bestapproximation pressure",FESpacePressure)
        H1VelocityBestapproximation = FEVector{Float64}("H1-Bestapproximation velocity",FESpaceVelocity)
        solve!(L2VelocityBestapproximation, L2VelocityBestapproximationProblem)
        solve!(L2PressureBestapproximation, L2PressureBestapproximationProblem)
        solve!(H1VelocityBestapproximation, H1VelocityBestapproximationProblem)

        ## compute L2 and H1 error
        append!(L2error_velocity,sqrt(evaluate(L2VelocityErrorEvaluator,Solution[1])))
        append!(L2error_velocity2,sqrt(evaluate(L2VelocityErrorEvaluator,Solution2[1])))
        append!(L2errorBestApproximation_velocity,sqrt(evaluate(L2VelocityErrorEvaluator,L2VelocityBestapproximation[1])))
        append!(L2error_pressure,sqrt(evaluate(L2PressureErrorEvaluator,Solution[2])))
        append!(L2error_pressure2,sqrt(evaluate(L2PressureErrorEvaluator,Solution2[2])))
        append!(L2errorBestApproximation_pressure,sqrt(evaluate(L2PressureErrorEvaluator,L2PressureBestapproximation[1])))
        append!(H1error_velocity,sqrt(evaluate(H1VelocityErrorEvaluator,Solution[1])))
        append!(H1error_velocity2,sqrt(evaluate(H1VelocityErrorEvaluator,Solution2[1])))
        append!(H1errorBestApproximation_velocity,sqrt(evaluate(H1VelocityErrorEvaluator,H1VelocityBestapproximation[1])))
        
        ## print results
        if (level == nlevels) && (print_results)
            println("\n         |   L2ERROR    |   L2ERROR    |   L2ERROR")
            println("   NDOF  | VELO-CLASSIC | VELO-PROBUST | VELO-L2BEST");
            for j=1:nlevels
                @printf("  %6d |",NDofs[j]);
                @printf(" %.6e |",L2error_velocity[j])
                @printf(" %.6e |",L2error_velocity2[j])
                @printf(" %.6e\n",L2errorBestApproximation_velocity[j])
            end
            println("\n         |   H1ERROR    |   H1ERROR    |   H1ERROR")
            println("   NDOF  | VELO-CLASSIC | VELO-PROBUST | VELO-H1BEST");
            for j=1:nlevels
                @printf("  %6d |",NDofs[j]);
                @printf(" %.6e |",H1error_velocity[j])
                @printf(" %.6e |",H1error_velocity2[j])
                @printf(" %.6e\n",H1errorBestApproximation_velocity[j])
            end
            println("\n         |   L2ERROR    |   L2ERROR    |   L2ERROR")
            println("   NDOF  | PRES-CLASSIC | PRES-PROBUST | PRES-L2BEST");
            for j=1:nlevels
                @printf("  %6d |",NDofs[j]);
                @printf(" %.6e |",L2error_pressure[j])
                @printf(" %.6e |",L2error_pressure2[j])
                @printf(" %.6e\n",L2errorBestApproximation_pressure[j])
            end
            println("\nLEGEND\n======")
            println("VELO-CLASSIC : discrete Stokes velocity solution ($(FESpaceVelocity.name)) with classical discretisation")
            println("VELO-PROBUST : discrete Stokes velocity solution ($(FESpaceVelocity.name)) with p-robust discretisation")
            println("VELO-L2BEST : L2-Bestapproximation of exact velocity (with boundary data)")
            println("VELO-H1BEST : H1-Bestapproximation of exact velocity (with boudnary data)")
            println("PRES-CLASSIC : discrete Stokes pressure solution ($(FESpacePressure.name)) with classical discretisation")
            println("PRES-PROBUST : discrete Stokes pressure solution ($(FESpaceVelocity.name)) with p-robust discretisation")
            println("PRES-L2BEST : L2-Bestapproximation of exact pressure (without boundary data)")
        end    
    end    

    ## return last error for testing
    return L2error_velocity2[end]
end


## everything is wrapped in a main function
function main(; verbosity = 0, nlevels = 4, viscosity = 1e-2)
    ## set problem to solve
    #Problem = HydrostaticTestProblem
    Problem = PotentialFlowTestProblem

    ## set grid and problem parameters
    xgrid = grid_unitsquare_mixedgeometries() # initial grid

    ## choose finite element discretisation
    #FETypes = [H1BR{2}, L2P0{1}, ReconstructionIdentity{HDIVRT0{2}}] # Bernardi--Raugel with RT0 reconstruction
    FETypes = [H1BR{2}, L2P0{1}, ReconstructionIdentity{HDIVBDM1{2}}] # Bernardi--Raugel with BDM1 reconstruction
    #FETypes = [H1CR{2}, L2P0{1}, ReconstructionIdentity{HDIVRT0{2}}] # Crouzeix--Raviart with RT0 reconstruction

    ## run
    solve(Problem, xgrid, FETypes, viscosity; nlevels = nlevels, verbosity = verbosity)
end


## test function that is called by test unit
## tests if hydrostatic problem is solved exactly by pressure-robust methods
function test(; verbosity = 0)
    xgrid = uniform_refine(grid_unitsquare_mixedgeometries())

    testspaces = [[H1CR{2}, L2P0{1}, ReconstructionIdentity{HDIVRT0{2}}],
                  [H1BR{2}, L2P0{1}, ReconstructionIdentity{HDIVRT0{2}}],
                  [H1BR{2}, L2P0{1}, ReconstructionIdentity{HDIVBDM1{2}}]]
    error = []
    for FETypes in testspaces
        push!(error, solve(HydrostaticTestProblem, xgrid, FETypes, 1; nlevels = 1, print_results = false, verbosity = verbosity))
        println("FETypes = $FETypes   error = $(error[end])")
    end
    return maximum(error)
end

end