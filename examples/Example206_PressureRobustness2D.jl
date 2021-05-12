#= 

# 206 : Pressure-robustness 2D
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


module Example206_PressureRobustness2D

using GradientRobustMultiPhysics

## problem data
function HydrostaticTestProblem()
    ## Stokes problem with f = grad(p)
    ## u = 0
    ## p = x^3+y^3 - 1//2
    function P1_pressure!(result,x::Array{<:Real,1})
        result[1] = x[1]^3 + x[2]^3 - 1//2
    end
    function P1_rhs!(result,x::Array{<:Real,1})
        result[1] = 3*x[1]^2
        result[2] = 3*x[2]^2
    end
    user_function_velocity = DataFunction([0,0]; name = "u_exact")
    user_function_pressure = DataFunction(P1_pressure!, [1,2]; name = "p_exact", dependencies = "X", quadorder = 3)
    user_function_velocity_gradient = DataFunction([0,0,0,0]; name = "grad(u_exact)")
    user_function_rhs = DataFunction(P1_rhs!, [2,2]; name = "f", dependencies = "X", quadorder = 2)

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
    user_function_velocity = DataFunction(P2_velo!, [2,2]; name = "u_exact", dependencies = "X", quadorder = 2)
    user_function_pressure = DataFunction(P2_pressure!, [1,2]; name = "p_exact", dependencies = "X", quadorder = 4)
    user_function_velocity_gradient = DataFunction(P2_velogradient!, [4,2]; name = "grad(u_exact)", dependencies = "X", quadorder = 1)
    user_function_rhs = DataFunction([0,0]; name = "f")

    return user_function_pressure,user_function_velocity,user_function_velocity_gradient,user_function_rhs, true
end


function solve(Problem, xgrid, FETypes, viscosity = 1e-2; nlevels = 4, print_results = true, verbosity = 1, target_residual = 1e-10, maxiterations = 20)

    ## load problem data and set solver parameters
    ReconstructionOperator = FETypes[3]
    exact_pressure!, exact_velocity!, exact_velocity_gradient!, rhs!, nonlinear = Problem()

    ## setup classical (Problem) and pressure-robust scheme (Problem2)
    Problem = IncompressibleNavierStokesProblem(2; viscosity = viscosity, nonlinear = false)
    add_boundarydata!(Problem, 1, [1,2,3,4], BestapproxDirichletBoundary; data = exact_velocity!)
    Problem2 = deepcopy(Problem)
    Problem.name = "Stokes problem (classical)"
    Problem2.name = "Stokes problem (p-robust)"

    ## assign right-hand side
    add_rhsdata!(Problem, 1, RhsOperator(Identity, [0], rhs!))
    add_rhsdata!(Problem2, 1, RhsOperator(ReconstructionOperator, [0], rhs!))

    ## assign convection term
    if nonlinear
        add_operator!(Problem,[1,1], ConvectionOperator(1, Identity, 2, 2))
        add_operator!(Problem2,[1,1], ConvectionOperator(1, ReconstructionOperator, 2, 2; testfunction_operator = ReconstructionOperator))
    end

    ## define bestapproximation problems
    L2VelocityBestapproximationProblem = L2BestapproximationProblem(exact_velocity!; bestapprox_boundary_regions = [1,2,3,4])
    L2PressureBestapproximationProblem = L2BestapproximationProblem(exact_pressure!; bestapprox_boundary_regions = [])
    H1VelocityBestapproximationProblem = H1BestapproximationProblem(exact_velocity_gradient!, exact_velocity!; bestapprox_boundary_regions = [1,2,3,4])

    ## define ItemIntegrators for L2/H1 error computation
    L2VelocityErrorEvaluator = L2ErrorIntegrator(Float64, exact_velocity!, Identity)
    L2PressureErrorEvaluator = L2ErrorIntegrator(Float64, exact_pressure!, Identity)
    H1VelocityErrorEvaluator = L2ErrorIntegrator(Float64, exact_velocity_gradient!, Gradient)
    Results = zeros(Float64, nlevels, 9)
    NDofs = zeros(Int, nlevels)
    
    ## loop over refinement levels
    for level = 1 : nlevels

        ## uniform mesh refinement
        xgrid = uniform_refine(xgrid)
        xFaceVolumes = xgrid[FaceVolumes]

        ## get FESpaces
        FES = [FESpace{FETypes[1]}(xgrid), FESpace{FETypes[2]}(xgrid; broken = true)]
        Solution = FEVector{Float64}(["u_c (classic)", "p_c (classic)"],FES)
        Solution2 = FEVector{Float64}(["u_r (p-robust)", "p_r (p-robust)"],FES)

        ## solve both problems
        solve!(Solution, Problem; maxiterations = maxiterations, target_residual = target_residual, anderson_iterations = 5)
        solve!(Solution2, Problem2; maxiterations = maxiterations, target_residual = target_residual, anderson_iterations = 5)

        ## solve bestapproximation problems
        L2VelocityBestapproximation = FEVector{Float64}("Πu",FES[1])
        L2PressureBestapproximation = FEVector{Float64}("πp",FES[2])
        H1VelocityBestapproximation = FEVector{Float64}("Su",FES[1])
        solve!(L2VelocityBestapproximation, L2VelocityBestapproximationProblem)
        solve!(L2PressureBestapproximation, L2PressureBestapproximationProblem)
        solve!(H1VelocityBestapproximation, H1VelocityBestapproximationProblem)

        ## compute L2 and H1 errors and save data
        NDofs[level] = length(Solution.entries)
        Results[level,1] = sqrt(evaluate(L2VelocityErrorEvaluator,Solution[1]))
        Results[level,2] = sqrt(evaluate(L2VelocityErrorEvaluator,Solution2[1]))
        Results[level,3] = sqrt(evaluate(L2VelocityErrorEvaluator,L2VelocityBestapproximation[1]))
        Results[level,4] = sqrt(evaluate(L2PressureErrorEvaluator,Solution[2]))
        Results[level,5] = sqrt(evaluate(L2PressureErrorEvaluator,Solution2[2]))
        Results[level,6] = sqrt(evaluate(L2PressureErrorEvaluator,L2PressureBestapproximation[1]))
        Results[level,7] = sqrt(evaluate(H1VelocityErrorEvaluator,Solution[1]))
        Results[level,8] = sqrt(evaluate(H1VelocityErrorEvaluator,Solution2[1]))
        Results[level,9] = sqrt(evaluate(H1VelocityErrorEvaluator,H1VelocityBestapproximation[1]))
    end

    ## print convergence history
    print_convergencehistory(NDofs, Results[:,1:3]; X_to_h = X -> X.^(-1/2), ylabels = ["||u-u_c||", "||u-u_r||", "||u-Πu||"])
    print_convergencehistory(NDofs, Results[:,4:6]; X_to_h = X -> X.^(-1/2), ylabels = ["||p-p_c||", "||p-p_r||", "||p-πp||"])
    print_convergencehistory(NDofs, Results[:,7:9]; X_to_h = X -> X.^(-1/2), ylabels = ["||∇(u-u_c)||", "||∇(u-u_r)||", "||∇(u-Su)||"])

    ## return last L2 error of p-robust method for testing
    return Results[end,2]
end


## everything is wrapped in a main function
function main(; problem = 2, verbosity = 0, nlevels = 4, viscosity = 1e-2)

    ## set log level
    set_verbosity(verbosity)
    
    ## set problem to solve
    if problem == 1
        Problem = HydrostaticTestProblem
    elseif problem == 2
        Problem = PotentialFlowTestProblem
    else
        @error "No problem defined for this number!"
    end

    ## set grid and problem parameters
    xgrid = grid_unitsquare_mixedgeometries() # initial grid

    ## choose finite element discretisation
    #FETypes = [H1BR{2}, H1P0{1}, ReconstructionIdentity{HDIVRT0{2}}] # Bernardi--Raugel with RT0 reconstruction
    FETypes = [H1BR{2}, H1P0{1}, ReconstructionIdentity{HDIVBDM1{2}}] # Bernardi--Raugel with BDM1 reconstruction
    #FETypes = [H1CR{2}, H1P0{1}, ReconstructionIdentity{HDIVRT0{2}}] # Crouzeix--Raviart with RT0 reconstruction

    ## run
    solve(Problem, xgrid, FETypes, viscosity; nlevels = nlevels)

    return nothing
end


## test function that is called by test unit
## tests if hydrostatic problem is solved exactly by pressure-robust methods
function test()
    xgrid = uniform_refine(grid_unitsquare_mixedgeometries())

    testspaces = [[H1CR{2}, H1P0{1}, ReconstructionIdentity{HDIVRT0{2}}],
                  [H1BR{2}, H1P0{1}, ReconstructionIdentity{HDIVRT0{2}}],
                  [H1BR{2}, H1P0{1}, ReconstructionIdentity{HDIVBDM1{2}}]]
    error = []
    for FETypes in testspaces
        push!(error, solve(HydrostaticTestProblem, xgrid, FETypes, 1; nlevels = 1, print_results = false))
        println("FETypes = $FETypes   error = $(error[end])")
    end
    return maximum(error)
end

end