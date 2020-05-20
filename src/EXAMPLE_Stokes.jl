
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
    xgrid[BFaceRegions]=Array{Int32,1}([1,1,2,2,1,1,3,3])
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
    nlevels = 6 # number of refinement levels
    #fem = "BR" # Bernardi--Raugel
    #fem = "MINI" # MINI element
    fem = "CR" # MINI element
    #fem = "TH" # Taylor--Hood
    verbosity = 3 # deepness of messaging (the larger, the more)

    # define expected solution, boundary data and volume data
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
    function exact_solution_rhs!(result,x)
        result[1] = 0.0
        result[2] = 0.0
    end    
    function exact_divergence!(result,x)
        result[1] = 0.0
    end    

    # PDE description
    MyLHS = Array{Array{AbstractPDEOperator,1},2}(undef,2,2)
    #MyLHS[1,1] = [LaplaceOperator(DoNotChangeAction(4))]
    MyLHS[1,1] = [LaplaceOperator(MultiplyScalarAction(viscosity,4))]
    MyLHS[1,2] = [LagrangeMultiplier(Divergence)] # automatically fills transposed block
    MyLHS[2,1] = []
    MyLHS[2,2] = []
    MyRHS = Array{Array{AbstractPDEOperator,1},1}(undef,2)
    MyRHS[1] = [RhsOperator(Identity, [exact_solution_rhs!], 2, 2; bonus_quadorder = 3)]
    MyRHS[2] = []
    MyBoundaryVelocity = BoundaryOperator(2,2)
    append!(MyBoundaryVelocity, 1, HomogeneousDirichletBoundary; data = bnd_data_rest!)
    append!(MyBoundaryVelocity, 2, BestapproxDirichletBoundary; data = exact_velocity!, bonus_quadorder = 2)
    append!(MyBoundaryVelocity, 3, BestapproxDirichletBoundary; data = exact_velocity!, bonus_quadorder = 2)
    MyBoundaryPressure = BoundaryOperator(2,1) # empty, no pressure boundary conditions
    MyGlobalConstraints = Array{Array{AbstractGlobalConstraint,1},1}(undef,2)
    MyGlobalConstraints[1] = Array{AbstractGlobalConstraint,1}(undef,0)
    MyGlobalConstraints[2] = [FixedIntegralMean(0.0)]
    StokesProblem = PDEDescription("StokesProblem",MyLHS,MyRHS,[MyBoundaryVelocity,MyBoundaryPressure],MyGlobalConstraints)

    # define ItemIntegrators for L2/H1 error computation
    L2DivergenceErrorEvaluator = L2ErrorIntegrator(exact_divergence!, Divergence, 2, 1; bonus_quadorder = 0)
    L2VelocityErrorEvaluator = L2ErrorIntegrator(exact_velocity!, Identity, 2, 2; bonus_quadorder = 4)
    L2PressureErrorEvaluator = L2ErrorIntegrator(exact_pressure!, Identity, 2, 1; bonus_quadorder = 2)
    H1VelocityErrorEvaluator = L2ErrorIntegrator(exact_velocity_gradient!, Gradient, 2, 4; bonus_quadorder = 2)
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
        solve!(Solution, StokesProblem; verbosity = verbosity - 1)
        push!(NDofs,length(Solution.entries))

        # interpolate
        Interpolation = FEVector{Float64}("Interpolation velocity",FE_velocity)
        append!(Interpolation,"Interpolation pressure",FE_pressure)
        interpolate!(Interpolation[1], exact_velocity!; verbosity = verbosity - 1, bonus_quadorder = 2)
        interpolate!(Interpolation[2], exact_pressure!; verbosity = verbosity - 1, bonus_quadorder = 1)

        # L2 bestapproximation
        L2Bestapproximation = FEVector{Float64}("L2-Bestapproximation velocity",FE_velocity)
        append!(L2Bestapproximation,"L2-Bestapproximation pressure",FE_pressure)
        L2bestapproximate!(L2Bestapproximation[1], exact_velocity!; boundary_regions = [0], verbosity = verbosity - 1, bonus_quadorder = 2)
        L2bestapproximate!(L2Bestapproximation[2], exact_pressure!; boundary_regions = [], verbosity = verbosity - 1, bonus_quadorder = 1)

        # H1 bestapproximation
        H1Bestapproximation = FEVector{Float64}("H1-Bestapproximation velocity",FE_velocity)
        H1bestapproximate!(H1Bestapproximation[1], exact_velocity_gradient!, exact_velocity!; verbosity = verbosity - 1, bonus_quadorder = 1, bonus_quadorder_boundary = 2)
        

        # compute L2 and H1 error
        #println("\nL2divergence = $(sqrt(evaluate(L2DivergenceErrorEvaluator,Solution[1])))")
        #println("L2divergenceI = $(sqrt(evaluate(L2DivergenceErrorEvaluator,Interpolation[1])))")
        append!(L2error_velocity,sqrt(evaluate(L2VelocityErrorEvaluator,Solution[1])))
        append!(L2errorInterpolation_velocity,sqrt(evaluate(L2VelocityErrorEvaluator,Interpolation[1])))
        append!(L2errorBestApproximation_velocity,sqrt(evaluate(L2VelocityErrorEvaluator,L2Bestapproximation[1])))
        append!(L2error_pressure,sqrt(evaluate(L2PressureErrorEvaluator,Solution[2])))
        append!(L2errorInterpolation_pressure,sqrt(evaluate(L2PressureErrorEvaluator,Interpolation[2])))
        append!(L2errorBestApproximation_pressure,sqrt(evaluate(L2PressureErrorEvaluator,L2Bestapproximation[2])))
        append!(H1error_velocity,sqrt(evaluate(H1VelocityErrorEvaluator,Solution[1])))
        append!(H1errorInterpolation_velocity,sqrt(evaluate(H1VelocityErrorEvaluator,Interpolation[1])))
        append!(H1errorBestApproximation_velocity,sqrt(evaluate(H1VelocityErrorEvaluator,H1Bestapproximation[1])))
        
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
