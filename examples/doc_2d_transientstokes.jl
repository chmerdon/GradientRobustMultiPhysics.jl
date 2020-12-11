#= 

# 2D Transient Stokes-Problem
([source code](SOURCE_URL))

This example computes a velocity ``\mathbf{u}`` and pressure ``\mathbf{p}`` of the incompressible Navier--Stokes problem
```math
\begin{aligned}
\mathbf{u}_t - \mu \Delta \mathbf{u} + \nabla p & = \mathbf{f}\\
\mathrm{div}(u) & = 0
\end{aligned}
```
with (possibly time-dependent) exterior force ``\mathbf{f}`` and some viscosity parameter ``\mu``.

In this example we solve an analytical toy problem with prescribed solution
```math
\begin{aligned}
\mathbf{u}(\mathbf{x},t) & = (1+t)(\cos(x_2), \sin(x_1))^T\\
p(\mathbf{x}) &= \sin(x_1+x_2) - 2\sin(1) + \sin(2)
\end{aligned}
```
with time-dependent right-hand side and inhomogeneous Dirichlet boundary data. The example showcases the
benefits of pressure-robustness in time-dependent linear Stokes problem in presence of complicated pressures and small viscosities.
The problem is solved on series of finer and finer unstructured simplex meshes and compares the error of the discrete Stokes solution,
an interpolation into the same space and the best-approximations into the same space. While a pressure-robust variant shows optimally
converging errors close to the best-approximations, a non pressure-robust discretisations show suboptimal (or no) convergence!
Compare e.g. Bernardi--Raugel and Bernardi--Raugel pressure-robust by (un)commenting the responsible lines in this example.
=#


module Example_2DTransientStokes

using GradientRobustMultiPhysics
using ExtendableGrids
using Printf


## problem data
function exact_pressure!(result,x::Array{<:Real,1})
    result[1] = sin(x[1]+x[2]) - 2*sin(1)+sin(2)
end
function exact_velocity!(result,x::Array{<:Real,1},t::Real)
    result[1] = (1+t)*cos(x[2]);
    result[2] = (1+t)*sin(x[1]);
end
function exact_velocity_gradient!(result,x::Array{<:Real,1},t::Real)
    result[1] = 0.0
    result[2] = -(1+t)*sin(x[2]);
    result[3] = (1+t)*cos(x[1]);
    result[4] = 0.0;
end
function exact_rhs!(viscosity)
    function closure(result,x::Array{<:Real,1},t::Real)
        result[1] = viscosity*(1+t)*cos(x[2]) + cos(x[1]+x[2]) + cos(x[2])
        result[2] = viscosity*(1+t)*sin(x[1]) + cos(x[1]+x[2]) + sin(x[1])
    end
end

## everything is wrapped in a main function
function main(; verbosity = 2, Plotter = nothing)

    ## problem parameters
    viscosity = 1e-6
    timestep = 1e-3
    T = 1e-2 # final time
    nlevels = 5 # maximal number of refinement levels
    reconstruct = false # do not change
    graddiv = 0
    broken_p = false # is pressure space broken ?

    ## initial grid
    xgrid = grid_unitsquare(Triangle2D);

    ## choose one of these (inf-sup stable) finite element type pairs
    #FETypes = [H1P2{2,2}, H1P1{1}] # Taylor--Hood
    #FETypes = [H1P2B{2,2}, H1P1{1}]; broken_p = true # P2-bubble
    #FETypes = [H1CR{2}, H1P0{1}]; broken_p = true # Crouzeix--Raviart
    #FETypes = [H1CR{2}, H1P0{1}]; broken_p = true; reconstruct = true # Crouzeix-Raviart gradient-robust
    #FETypes = [H1MINI{2,2}, H1P1{1}] # MINI element on triangles only
    #FETypes = [H1MINI{2,2}, H1CR{1}] # MINI element on triangles/quads
    #FETypes = [H1BR{2}, H1P0{1}]; broken_p = true # Bernardi--Raugel
    FETypes = [H1BR{2}, H1P0{1}]; broken_p = true; reconstruct = true # Bernardi--Raugel gradient-robust

    #####################################################################################    
    #####################################################################################

    ## set testfunction operator for certain testfunctions
    ## (pressure-robustness chooses a reconstruction that can exploit the L2-orthogonality onto gradients)
    ## (Scott-Vogelius is divergence-free and is pressure-robust without modifications)
    if reconstruct
        testfunction_operator = ReconstructionIdentity{HDIVBDM1{2}}
    else
        testfunction_operator = Identity
    end

    ## negotiate data functions to the package
    ## note that dependencies "XT" marks the function to be x- and t-dependent
    ## that causes the solver to automatically reassemble associated operators in each time step
    user_function_velocity = DataFunction(exact_velocity!, [2,2]; name = "u_exact", dependencies = "XT", quadorder = 5)
    user_function_pressure = DataFunction(exact_pressure!, [1,2]; name = "p_exact", dependencies = "X", quadorder = 5)
    user_function_velocity_gradient = DataFunction(exact_velocity_gradient!, [4,2]; name = "grad(u_exact)", dependencies = "XT", quadorder = 4)
    user_function_rhs = DataFunction(exact_rhs!(viscosity), [2,2]; name = "f", dependencies = "XT", quadorder = 5)

    ## load Stokes problem prototype and assign data
    StokesProblem = IncompressibleNavierStokesProblem(2; viscosity = viscosity, nonlinear = false)
    add_boundarydata!(StokesProblem, 1, [1,2,3,4], BestapproxDirichletBoundary; data = user_function_velocity)
    add_rhsdata!(StokesProblem, 1, RhsOperator(testfunction_operator, [1], user_function_rhs))

    ## add grad-div stabilisation
    if graddiv > 0
        add_operator!(StokesProblem, [1,1], AbstractBilinearForm("graddiv-stabilisation (div x div)", Divergence, Divergence, MultiplyScalarAction(graddiv)))
    end

    ## define bestapproximation problems
    L2PressureBestapproximationProblem = L2BestapproximationProblem(user_function_pressure; bestapprox_boundary_regions = [])
    L2VelocityBestapproximationProblem = L2BestapproximationProblem(user_function_velocity; bestapprox_boundary_regions = [1,2,3,4])
    H1VelocityBestapproximationProblem = H1BestapproximationProblem(user_function_velocity_gradient, user_function_velocity; bestapprox_boundary_regions = [1,2,3,4])

    ## define ItemIntegrators for L2/H1 error computation and arrays to store them
    L2VelocityErrorEvaluator = L2ErrorIntegrator(Float64, user_function_velocity, Identity; time = T)
    L2PressureErrorEvaluator = L2ErrorIntegrator(Float64, user_function_pressure, Identity)
    H1VelocityErrorEvaluator = L2ErrorIntegrator(Float64, user_function_velocity_gradient, Gradient; time = T)
    L2error_velocity = []; L2error_pressure = []; NDofs = []
    L2errorBestApproximation_velocity = []; L2errorBestApproximation_pressure = []
    H1error_velocity = []; H1errorBestApproximation_velocity = []
    
    ## loop over levels
    for level = 1 : nlevels

        xgrid = uniform_refine(xgrid)

        ## generate FESpaces
        FESpaceVelocity = FESpace{FETypes[1]}(xgrid)
        FESpacePressure = FESpace{FETypes[2]}(xgrid; broken = broken_p)

        ## generate solution fector
        Solution = FEVector{Float64}("Stokes velocity",FESpaceVelocity)
        append!(Solution,"Stokes pressure",FESpacePressure)
        push!(NDofs,length(Solution.entries))

        ## set initial solution ( = bestapproximation at time 0)
        L2VelocityBestapproximation = FEVector{Float64}("L2-Bestapproximation velocity",FESpaceVelocity)
        solve!(L2VelocityBestapproximation, L2VelocityBestapproximationProblem; time = 0)
        Solution[1][:] = L2VelocityBestapproximation[1][:]

        ## generate time-dependent solver and chance rhs data
        TCS = TimeControlSolver(StokesProblem, Solution, BackwardEuler; timedependent_equations = [1], maxlureuse = [-1], dt_testfunction_operator = [testfunction_operator], verbosity = verbosity)
        advance_until_time!(TCS, timestep, T)

        ## solve bestapproximation problems at final time for comparison
        L2PressureBestapproximation = FEVector{Float64}("L2-Bestapproximation pressure",FESpacePressure)
        H1VelocityBestapproximation = FEVector{Float64}("H1-Bestapproximation velocity",FESpaceVelocity)
        solve!(L2VelocityBestapproximation, L2VelocityBestapproximationProblem; time = T)
        solve!(L2PressureBestapproximation, L2PressureBestapproximationProblem;)
        solve!(H1VelocityBestapproximation, H1VelocityBestapproximationProblem; time = T)

        ## compute L2 and H1 error of all solutions
        append!(L2error_velocity,sqrt(evaluate(L2VelocityErrorEvaluator,Solution[1])))
        append!(L2errorBestApproximation_velocity,sqrt(evaluate(L2VelocityErrorEvaluator,L2VelocityBestapproximation[1])))
        append!(L2error_pressure,sqrt(evaluate(L2PressureErrorEvaluator,Solution[2])))
        append!(L2errorBestApproximation_pressure,sqrt(evaluate(L2PressureErrorEvaluator,L2PressureBestapproximation[1])))
        append!(H1error_velocity,sqrt(evaluate(H1VelocityErrorEvaluator,Solution[1])))
        append!(H1errorBestApproximation_velocity,sqrt(evaluate(H1VelocityErrorEvaluator,H1VelocityBestapproximation[1])))
        
        ## ouput errors
        if (level == nlevels)
            println("\n         |   L2ERROR      order   |   L2ERROR      order   ")
            println("   NDOF  | VELO-STOKES            | VELO-L2BEST            ");
            order = 0
            for j=1:nlevels
                if j > 1
                    order = log(L2error_velocity[j-1]/L2error_velocity[j]) / (log(NDofs[j]/NDofs[j-1])/2)
                end
                @printf("  %6d |",NDofs[j]);
                @printf(" %.5e ",L2error_velocity[j])
                @printf("   %.3f   |",order)
                if j > 1
                    order = log(L2errorBestApproximation_velocity[j-1]/L2errorBestApproximation_velocity[j]) / (log(NDofs[j]/NDofs[j-1])/2)
                end
                @printf(" %.5e ",L2errorBestApproximation_velocity[j])
                @printf("   %.3f\n",order)
            end
            println("\n         |   H1ERROR      order   |   H1ERROR      order   ")
            println("   NDOF  | VELO-STOKES            | VELO-H1BEST            ");
            order = 0
            for j=1:nlevels
                if j > 1
                    order = log(H1error_velocity[j-1]/H1error_velocity[j]) / (log(NDofs[j]/NDofs[j-1])/2)
                end
                @printf("  %6d |",NDofs[j]);
                @printf(" %.5e ",H1error_velocity[j])
                @printf("   %.3f   |",order)
                if j > 1
                    order = log(H1errorBestApproximation_velocity[j-1]/H1errorBestApproximation_velocity[j]) / (log(NDofs[j]/NDofs[j-1])/2)
                end
                @printf(" %.5e ",H1errorBestApproximation_velocity[j])
                @printf("   %.3f\n",order)
            end
            println("\n         |   L2ERROR      order   |   L2ERROR      order   ")
            println("   NDOF  | PRES-STOKES            | PRES-L2BEST            ");
            order = 0
            for j=1:nlevels
                if j > 1
                    order = log(L2error_pressure[j-1]/L2error_pressure[j]) / (log(NDofs[j]/NDofs[j-1])/2)
                end
                @printf("  %6d |",NDofs[j]);
                @printf(" %.5e ",L2error_pressure[j])
                @printf("   %.3f   |",order)
                if j > 1
                    order = log(L2errorBestApproximation_pressure[j-1]/L2errorBestApproximation_pressure[j]) / (log(NDofs[j]/NDofs[j-1])/2)
                end
                @printf(" %.5e ",L2errorBestApproximation_pressure[j])
                @printf("   %.3f\n",order)
            end
            println("\nLEGEND\n======")
            println("VELO-STOKES : discrete Stokes velocity solution ($(FESpaceVelocity.name))")
            println("VELO-L2BEST : L2-Bestapproximation of exact velocity (with boundary data)")
            println("VELO-H1BEST : H1-Bestapproximation of exact velocity (with boudnary data)")
            println("PRES-STOKES : discrete Stokes pressure solution ($(FESpacePressure.name))")
            println("PRES-L2BEST : L2-Bestapproximation of exact pressure (without boundary data)")

            GradientRobustMultiPhysics.plot(Solution, [0,1,2], [Identity, Identity]; Plotter = Plotter, verbosity = verbosity, use_subplots = true)
        end    
    end    
end

end
