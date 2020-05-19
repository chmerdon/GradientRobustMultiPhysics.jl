
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


function gridgen_mixedEG()

    NumberType = Float64
    xgrid=ExtendableGrid{NumberType,Int32}()
    xgrid[Coordinates]=Array{NumberType,2}([0 0; 4//10 0; 1 0; 0 6//10; 4//10 6//10; 1 6//10;0 1; 4//10 1; 1 1]')
    xCellNodes=VariableTargetAdjacency(Int32)
    xCellGeometries=[Triangle2D, Triangle2D, Parallelogram2D, Parallelogram2D, Triangle2D, Triangle2D];
    
    append!(xCellNodes,[1,5,4])
    append!(xCellNodes,[1,2,5])
    append!(xCellNodes,[2,3,6,5])
    append!(xCellNodes,[4,5,8,7]) 
    append!(xCellNodes,[5,6,9])
    append!(xCellNodes,[8,5,9])

    xgrid[CellNodes] = xCellNodes
    xgrid[CellGeometries] = xCellGeometries
    ncells = num_sources(xCellNodes)
    xgrid[CellRegions]=VectorOfConstants{Int32}(1,ncells)
    xgrid[BFaceRegions]=Array{Int32,1}([1,1,1,1,1,1,1,1])
    xBFaceNodes=Array{Int32,2}([1 2; 2 3; 3 6; 6 9; 9 8; 8 7; 7 4; 4 1]')
    xgrid[BFaceNodes]=xBFaceNodes
    nbfaces = num_sources(xBFaceNodes)
    xgrid[BFaceGeometries]=VectorOfConstants(Edge1D,nbfaces)
    xgrid[CoordinateSystem]=Cartesian2D

    return xgrid
end


function main()


    # initial grid
    xgrid = gridgen_mixedEG(); #xgrid = split_grid_into(xgrid,Triangle2D)
    nlevels = 5 # number of refinement levels
    verbosity = 3 # deepness of messaging (the larger, the more)

    # define expected solution, boundary data and volume data
    viscosity = 1.0
    function exact_pressure!(result,x)
        result[1] = x[1]^3 + x[2]^3 - 1//2
    end
    function exact_velocity!(result,x)
        result[1] = 0.0;
        result[2] = 0.0;
    end
    function exact_velocity_gradient!(result,x)
        result[1] = 0.0
        result[2] = 0.0;
        result[3] = 0.0;
        result[4] = 0.0;
    end
    function rhs!(result,x)
        result[1] = 3*x[1]^2
        result[2] = 3*x[2]^2
    end

    # PDE description
    MyLHS = Array{Array{AbstractPDEOperator,1},2}(undef,2,2)
    #MyLHS[1,1] = [LaplaceOperator(DoNotChangeAction(4))]
    MyLHS[1,1] = [LaplaceOperator(MultiplyScalarAction(viscosity,4))]
    MyLHS[1,2] = [LagrangeMultiplier(Divergence)] # automatically fills transposed block
    MyLHS[2,1] = []
    MyLHS[2,2] = []
    MyRHS = Array{Array{AbstractPDEOperator,1},1}(undef,2)
    MyRHS[1] = [RhsOperator(ReconstructionIdentity, [rhs!], 2, 2; bonus_quadorder = 2)]
    MyRHS[2] = []
    MyBoundaryVelocity = BoundaryOperator(2,2)
    append!(MyBoundaryVelocity, 1, HomogeneousDirichletBoundary)
    MyBoundaryPressure = BoundaryOperator(2,1) # empty, no pressure boundary conditions
    MyGlobalConstraints = Array{Array{AbstractGlobalConstraint,1},1}(undef,2)
    MyGlobalConstraints[1] = Array{AbstractGlobalConstraint,1}(undef,0)
    MyGlobalConstraints[2] = [FixedIntegralMean(0.0)]
    StokesProblem = PDEDescription("StokesProblem",MyLHS,MyRHS,[MyBoundaryVelocity,MyBoundaryPressure],MyGlobalConstraints)

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

        # generate Bernardi--Raugel
        FE_velocity = FiniteElements.getH1BRFiniteElement(xgrid)
        FE_pressure = FiniteElements.getP0FiniteElement(xgrid,1)
        if verbosity > 2
            FiniteElements.show(FE_velocity)
            FiniteElements.show(FE_pressure)
        end    

        # solve Stokes problem with classical right-hand side
        StokesProblem.RHSOperators[1][1] = RhsOperator(Identity, [rhs!], 2, 2; bonus_quadorder = 2)
        Solution = FEVector{Float64}("Stokes velocity classical",FE_velocity)
        append!(Solution,"Stokes pressure (classical)",FE_pressure)
        solve!(Solution, StokesProblem; verbosity = verbosity - 1)
        push!(NDofs,length(Solution.entries))

        # solve Stokes problem with pressure-robust right-hand side
        StokesProblem.RHSOperators[1][1] = RhsOperator(ReconstructionIdentity{FiniteElements.FEHdivRT0}, [rhs!], 2, 2; bonus_quadorder = 2)
        Solution2 = FEVector{Float64}("Stokes velocity p-robust",FE_velocity)
        append!(Solution2,"Stokes pressure (p-robust)",FE_pressure)
        solve!(Solution2, StokesProblem; verbosity = verbosity - 1)

        # L2 bestapproximation
        L2Bestapproximation = FEVector{Float64}("L2-Bestapproximation velocity",FE_velocity)
        append!(L2Bestapproximation,"L2-Bestapproximation pressure",FE_pressure)
        L2bestapproximate!(L2Bestapproximation[1], exact_velocity!; boundary_regions = [0], verbosity = verbosity - 1, bonus_quadorder = 0)
        L2bestapproximate!(L2Bestapproximation[2], exact_pressure!; boundary_regions = [], verbosity = verbosity - 1, bonus_quadorder = 3)

        # H1 bestapproximation (not working properly yet)
        H1Bestapproximation = FEVector{Float64}("H1-Bestapproximation velocity",FE_velocity)
        H1bestapproximate!(H1Bestapproximation[1], exact_velocity_gradient!, exact_velocity!; verbosity = verbosity - 1, bonus_quadorder = 0)
        
        # compute L2 and H1 error
        append!(L2error_velocity,sqrt(evaluate(L2VelocityErrorEvaluator,Solution[1])))
        append!(L2error_velocity2,sqrt(evaluate(L2VelocityErrorEvaluator,Solution2[1])))
        append!(L2errorBestApproximation_velocity,sqrt(evaluate(L2VelocityErrorEvaluator,L2Bestapproximation[1])))
        append!(L2error_pressure,sqrt(evaluate(L2PressureErrorEvaluator,Solution[2])))
        append!(L2error_pressure2,sqrt(evaluate(L2PressureErrorEvaluator,Solution2[2])))
        append!(L2errorBestApproximation_pressure,sqrt(evaluate(L2PressureErrorEvaluator,L2Bestapproximation[2])))
        append!(H1error_velocity,sqrt(evaluate(H1VelocityErrorEvaluator,Solution[1])))
        append!(H1error_velocity2,sqrt(evaluate(H1VelocityErrorEvaluator,Solution2[1])))
        append!(H1errorBestApproximation_velocity,sqrt(evaluate(H1VelocityErrorEvaluator,L2Bestapproximation[1])))
        
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
            println("VELO-CLASSIC : discrete Stokes velocity solution ($(FE_velocity.name)) with classic right-hand side")
            println("VELO-PROBUST : discrete Stokes velocity solution ($(FE_velocity.name)) with p-robust right-hand side")
            println("VELO-L2BEST : L2-Bestapproximation of exact velocity (with boundary data)")
            println("VELO-H1BEST : H1-Bestapproximation of exact velocity (with boudnary data)")
            println("PRES-CLASSIC : discrete Stokes pressure solution ($(FE_pressure.name)) with classic right-hand sid")
            println("PRES-PROBUST : discrete Stokes pressure solution ($(FE_velocity.name)) with p-robust right-hand side")
            println("PRES-L2BEST : L2-Bestapproximation of exact pressure (without boundary data)")
        end    
    end    


end


main()
