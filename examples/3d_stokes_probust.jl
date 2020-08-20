push!(LOAD_PATH, "../src")
using GradientRobustMultiPhysics
using ExtendableGrids
ENV["MPLBACKEND"]="qt5agg"
using PyPlot
using Printf

# problem data
function HydrostaticTestProblem()
    # Stokes problem with f = grad(p)
    # u = 0
    # p = x^2+y^2+z^2 - 1
    function P1_pressure!(result,x)
        result[1] = x[1]^2 + x[2]^2 + x[3]^2 - 1
    end
    function P1_velo!(result,x)
        result[1] = 0.0;
        result[2] = 0.0;
    end
    function P1_velogradient!(result,x)
        result[1] = 0.0
        result[2] = 0.0;
        result[3] = 0.0;
        result[4] = 0.0;
    end
    function P1_rhs!(result,x)
        result[1] = 2*x[1]
        result[2] = 2*x[2]
        result[3] = 2*x[3]
    end
    return P1_pressure!, P1_velo!, P1_velogradient!, P1_rhs!, false
end


## everything is wrapped in a main function
function main()
    #####################################################################################
    #####################################################################################

    # meshing parameters
    xgrid_b = grid_unitcube(Parallelepiped3D); # initial grid
    nlevels = 4 # number of refinement levels
    split_into_tets = true

    # problem parameters
    viscosity = 1e-2
    exact_pressure!, exact_velocity!, exact_velocity_gradient!, rhs!, nonlinear = HydrostaticTestProblem()

    # choose finite element type
    FETypes = [H1BR{3}, L2P0{1}] # Bernardi--Raugel
    #FETypes = [H1CR{3}, L2P0{1}] # Crouzeix--Raviart

    # reconstruction operator
    ReconstructionOperator = ReconstructionIdentity{HDIVRT0{3}}

    # solver parameters
    maxIterations = 20  # termination criterion 1 for nonlinear mode
    maxResidual = 1e-12 # termination criterion 2 for nonlinear mode
    verbosity = 1 # deepness of messaging (the larger, the more)

    #####################################################################################    
    #####################################################################################

    # load Stokes problem prototype and assign data
    StokesProblem = IncompressibleNavierStokesProblem(3; viscosity = viscosity, nonlinear = nonlinear)
    add_boundarydata!(StokesProblem, 1, [1,2,3,4,5,6], BestapproxDirichletBoundary; data = exact_velocity!, bonus_quadorder = 1)
    add_rhsdata!(StokesProblem, 1, RhsOperator(ReconstructionIdentity, [0], rhs!, 3, 3; bonus_quadorder = 1))

    # define bestapproximation problems
    L2VelocityBestapproximationProblem = L2BestapproximationProblem(exact_velocity!, 3, 3; bestapprox_boundary_regions = [1,2,3,4,5,6], bonus_quadorder = 2)
    L2PressureBestapproximationProblem = L2BestapproximationProblem(exact_pressure!, 3, 1; bestapprox_boundary_regions = [], bonus_quadorder = 2)
    H1VelocityBestapproximationProblem = H1BestapproximationProblem(exact_velocity_gradient!, exact_velocity!, 3, 3; bestapprox_boundary_regions = [1,2,3,4,5,6], bonus_quadorder = 1, bonus_quadorder_boundary = 1)
 
    # define ItemIntegrators for L2/H1 error computation
    L2VelocityErrorEvaluator = L2ErrorIntegrator(exact_velocity!, Identity, 3, 3; bonus_quadorder = 0)
    L2PressureErrorEvaluator = L2ErrorIntegrator(exact_pressure!, Identity, 3, 1; bonus_quadorder = 2)
    H1VelocityErrorEvaluator = L2ErrorIntegrator(exact_velocity_gradient!, Gradient, 3, 9; bonus_quadorder = 0)
    L2error_velocity = []
    L2error_pressure = []
    L2error_velocity2 = []
    L2error_pressure2 = []
    L2errorInterpolation_velocity = []
    L2errorInterpolation_pressure = []
    L2errorBestApproximation_velocity = []
    L2errorBestApproximation_pressure = []
    H1error_velocity = []
    H1error_velocity2 = []
    H1errorBestApproximation_velocity = []
    NDofs = []
    
    # loop over levels
    for level = 1 : nlevels

        ## uniform mesh refinement
        if level > 1 
            xgrid_b = uniform_refine(xgrid_b)
        end
        if split_into_tets
            xgrid = split_grid_into(xgrid_b, Tetrahedron3D)
        else
            xgrid = xgrid_b
        end
        

        # generate FESpaces
        FESpaceVelocity = FESpace{FETypes[1]}(xgrid)
        FESpacePressure = FESpace{FETypes[2]}(xgrid)

        # solve Stokes problem with classical right-hand side/convection term
        StokesProblem.RHSOperators[1][1] = RhsOperator(Identity, [0], rhs!, 3, 3; bonus_quadorder = 1)
        Solution = FEVector{Float64}("Stokes velocity classical",FESpaceVelocity)
        append!(Solution,"Stokes pressure (classical)",FESpacePressure)
        solve!(Solution, StokesProblem; verbosity = verbosity, maxIterations = maxIterations, maxResidual = maxResidual)
        push!(NDofs,length(Solution.entries))

        # solve Stokes problem with pressure-robust right-hand side/convection term
        StokesProblem.RHSOperators[1][1] = RhsOperator(ReconstructionOperator, [0], rhs!, 3, 3; bonus_quadorder = 1)
        Solution2 = FEVector{Float64}("Stokes velocity p-robust",FESpaceVelocity)
        append!(Solution2,"Stokes pressure (p-robust)",FESpacePressure)
        solve!(Solution2, StokesProblem; verbosity = verbosity, maxIterations = maxIterations, maxResidual = maxResidual)

        # solve bestapproximation problems
        L2VelocityBestapproximation = FEVector{Float64}("L2-Bestapproximation velocity",FESpaceVelocity)
        L2PressureBestapproximation = FEVector{Float64}("L2-Bestapproximation pressure",FESpacePressure)
        H1VelocityBestapproximation = FEVector{Float64}("H1-Bestapproximation velocity",FESpaceVelocity)
        solve!(L2VelocityBestapproximation, L2VelocityBestapproximationProblem; verbosity = verbosity)
        solve!(L2PressureBestapproximation, L2PressureBestapproximationProblem; verbosity = verbosity)
        solve!(H1VelocityBestapproximation, H1VelocityBestapproximationProblem; verbosity = verbosity)

        # compute L2 and H1 error
        append!(L2error_velocity,sqrt(evaluate(L2VelocityErrorEvaluator,Solution[1])))
        append!(L2error_velocity2,sqrt(evaluate(L2VelocityErrorEvaluator,Solution2[1])))
        append!(L2errorBestApproximation_velocity,sqrt(evaluate(L2VelocityErrorEvaluator,L2VelocityBestapproximation[1])))
        append!(L2error_pressure,sqrt(evaluate(L2PressureErrorEvaluator,Solution[2])))
        append!(L2error_pressure2,sqrt(evaluate(L2PressureErrorEvaluator,Solution2[2])))
        append!(L2errorBestApproximation_pressure,sqrt(evaluate(L2PressureErrorEvaluator,L2PressureBestapproximation[1])))
        append!(H1error_velocity,sqrt(evaluate(H1VelocityErrorEvaluator,Solution[1])))
        append!(H1error_velocity2,sqrt(evaluate(H1VelocityErrorEvaluator,Solution2[1])))
        append!(H1errorBestApproximation_velocity,sqrt(evaluate(H1VelocityErrorEvaluator,H1VelocityBestapproximation[1])))
        
        # print results
        if (level == nlevels)
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
            println("VELO-CLASSIC : discrete Stokes velocity solution ($(FESpaceVelocity.name)) with classic right-hand side")
            println("VELO-PROBUST : discrete Stokes velocity solution ($(FESpaceVelocity.name)) with p-robust right-hand side")
            println("VELO-L2BEST : L2-Bestapproximation of exact velocity (with boundary data)")
            println("VELO-H1BEST : H1-Bestapproximation of exact velocity (with boudnary data)")
            println("PRES-CLASSIC : discrete Stokes pressure solution ($(FESpacePressure.name)) with classic right-hand sid")
            println("PRES-PROBUST : discrete Stokes pressure solution ($(FESpaceVelocity.name)) with p-robust right-hand side")
            println("PRES-L2BEST : L2-Bestapproximation of exact pressure (without boundary data)")
        end    
    end    


end


main()
