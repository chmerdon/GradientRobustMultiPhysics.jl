
using ExtendableGrids
ENV["MPLBACKEND"]="qt5agg"
using PyPlot
using Printf

push!(LOAD_PATH, "../src")
using GradientRobustMultiPhysics

include("../src/testgrids.jl")

# problem data
function HydrostaticTestProblem()
    # Stokes problem with f = grad(p)
    # u = 0
    # p = x^3+y^3 - 1//2
    function P1_pressure!(result,x)
        result[1] = x[1]^3 + x[2]^3 - 1//2
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
        result[1] = 3*x[1]^2
        result[2] = 3*x[2]^2
    end
    return P1_pressure!, P1_velo!, P1_velogradient!, P1_rhs!, false
end

function PotentialFlowTestProblem()
    # NavierStokes with f = 0
    # u = grad(h) with h = x^3 - 3xy^2
    # p = - |grad(h)|^2 + 14//5
    function P2_pressure!(result,x)
        result[1] = - 1//2 * (9*(x[1]^4 + x[2]^4) + 18*x[1]^2*x[2]^2) + 14//5
    end
    function P2_velo!(result,x)
        result[1] = 3*x[1]^2 - 3*x[2]^2;
        result[2] = -6*x[1]*x[2];
    end
    function P2_velogradient!(result,x)
        result[1] = 6*x[1]
        result[2] = -6*x[2];
        result[3] = -6*x[2];
        result[4] = -6*x[1];
    end
    function P2_rhs!(result,x)
        result[1] = 0
        result[2] = 0
    end
    return P2_pressure!, P2_velo!, P2_velogradient!, P2_rhs!, true
end


function main()
    #####################################################################################
    #####################################################################################

    # meshing parameters
    xgrid = uniform_refine(uniform_refine(grid_unitsquare_mixedgeometries())); # initial grid
    #xgrid = split_grid_into(xgrid,Triangle2D) # if you want just triangles
    nlevels = 3 # number of refinement levels

    # problem parameters
    viscosity = 1e-2
    #exact_pressure!, exact_velocity!, exact_velocity_gradient!, rhs!, nonlinear = HydrostaticTestProblem()
    exact_pressure!, exact_velocity!, exact_velocity_gradient!, rhs!, nonlinear = PotentialFlowTestProblem()

    # choose finite element type
    FETypes = [H1BR{2}, L2P0{1}] # Bernardi--Raugel
    #FETypes = [H1CR{2}, L2P0{1}] # Crouzeix--Raviart

    # reconstruction operator
    TestFunctionReconstruction = ReconstructionIdentity{HDIVRT0{2}} # do not change

    # solver parameters
    maxIterations = 20  # termination criterion 1 for nonlinear mode
    maxResidual = 1e-12 # termination criterion 2 for nonlinear mode
    verbosity = 1 # deepness of messaging (the larger, the more)

    #####################################################################################    
    #####################################################################################

    # load Stokes problem prototype and assign data
    StokesProblem = IncompressibleNavierStokesProblem(2; viscosity = viscosity, nonlinear = nonlinear)
    add_boundarydata!(StokesProblem, 1, [1,2,3,4], BestapproxDirichletBoundary; data = exact_velocity!, bonus_quadorder = 2)
    add_rhsdata!(StokesProblem, 1, RhsOperator(ReconstructionIdentity, [0], rhs!, 2, 2; bonus_quadorder = 2))

    # define bestapproximation problems
    L2VelocityBestapproximationProblem = L2BestapproximationProblem(exact_velocity!, 2, 2; bestapprox_boundary_regions = [1,2,3,4], bonus_quadorder = 2)
    L2PressureBestapproximationProblem = L2BestapproximationProblem(exact_pressure!, 2, 1; bestapprox_boundary_regions = [], bonus_quadorder = 4)
    H1VelocityBestapproximationProblem = H1BestapproximationProblem(exact_velocity_gradient!, exact_velocity!, 2, 2; bestapprox_boundary_regions = [1,2,3,4], bonus_quadorder = 1, bonus_quadorder_boundary = 2)
 
    # define ItemIntegrators for L2/H1 error computation
    L2VelocityErrorEvaluator = L2ErrorIntegrator(exact_velocity!, Identity, 2, 2; bonus_quadorder = 0)
    L2PressureErrorEvaluator = L2ErrorIntegrator(exact_pressure!, Identity, 2, 1; bonus_quadorder = 3)
    H1VelocityErrorEvaluator = L2ErrorIntegrator(exact_velocity_gradient!, Gradient, 2, 4; bonus_quadorder = 0)
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

        # uniform mesh refinement
        if (level > 1) 
            xgrid = uniform_refine(xgrid)
        end

        # generate FESpaces
        FESpaceVelocity = FESpace{FETypes[1]}(xgrid)
        FESpacePressure = FESpace{FETypes[2]}(xgrid)

        # solve Stokes problem with classical right-hand side/convection term
        StokesProblem.RHSOperators[1][1] = RhsOperator(Identity, [0], rhs!, 2, 2; bonus_quadorder = 2)
        if nonlinear
            StokesProblem.LHSOperators[1,1][1].store_operator = true # store matrix of Laplace operator
            StokesProblem.LHSOperators[1,1][2] = ConvectionOperator(1, 2, 2; testfunction_operator = Identity)
        end
        Solution = FEVector{Float64}("Stokes velocity classical",FESpaceVelocity)
        append!(Solution,"Stokes pressure (classical)",FESpacePressure)
        @time solve!(Solution, StokesProblem; verbosity = verbosity, maxIterations = maxIterations, maxResidual = maxResidual)
        push!(NDofs,length(Solution.entries))

        # solve Stokes problem with pressure-robust right-hand side/convection term
        StokesProblem.RHSOperators[1][1] = RhsOperator(TestFunctionReconstruction, [0], rhs!, 2, 2; bonus_quadorder = 2)
        if nonlinear
            StokesProblem.LHSOperators[1,1][2] = ConvectionOperator(1, 2, 2; testfunction_operator = TestFunctionReconstruction)
        end
        Solution2 = FEVector{Float64}("Stokes velocity p-robust",FESpaceVelocity)
        append!(Solution2,"Stokes pressure (p-robust)",FESpacePressure)
        @time solve!(Solution2, StokesProblem; verbosity = verbosity, maxIterations = maxIterations, maxResidual = maxResidual)

        
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
        
        # plot final solution
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
