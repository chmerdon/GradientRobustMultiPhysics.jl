
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
using BenchmarkTools

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
    FEorder = 1 # optimal convergence order of velocity finite element
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
    MyLHS[1,1] = [LaplaceOperator(MultiplyScalarAction(viscosity,4))]
    MyLHS[1,2] = [LagrangeMultiplier(Divergence)] # automatically fills transposed block
    MyLHS[2,1] = []
    MyLHS[2,2] = []
    MyRHS = Array{Array{AbstractPDEOperator,1},1}(undef,2)
    MyRHS[1] = [RhsOperator(Identity, [exact_solution_rhs!], 2, 2; bonus_quadorder = 3)]
    MyRHS[2] = []
    MyBoundary = BoundaryOperator(2,2)
    append!(MyBoundary, 1, HomogeneousDirichletBoundary; data = bnd_data_rest!)
    append!(MyBoundary, 2, BestapproxDirichletBoundary; data = exact_velocity!)
    append!(MyBoundary, 3, BestapproxDirichletBoundary; data = exact_velocity!)
    MyBoundaryPressure = BoundaryOperator(2,1) # empty, no pressure boundary conditions
    StokesProblem = PDEDescription("StokesProblem",MyLHS,MyRHS,[MyBoundary,MyBoundaryPressure])


    # define ItemIntegrators for L2/H1 error computation
    L2DivergenceErrorEvaluator = L2ErrorIntegrator(exact_divergence!, Divergence, 2, 1; bonus_quadorder = 0)
    L2VelocityErrorEvaluator = L2ErrorIntegrator(exact_velocity!, Identity, 2, 2; bonus_quadorder = 7)
    L2PressureErrorEvaluator = L2ErrorIntegrator(exact_pressure!, Identity, 2, 1; bonus_quadorder = 1)
    #H1ErrorEvaluator = L2ErrorIntegrator(exact_solution_gradient!, Gradient, 2; bonus_quadorder = 3)
    L2error_velocity = []
    L2error_pressure = []
    L2errorInterpolation_velocity = []
    L2errorInterpolation_pressure = []

    # loop over levels
    for level = 1 : nlevels

        # uniform mesh refinement
        if (level > 1) 
            xgrid = uniform_refine(xgrid)
        end

        # generate FE
        if FEorder == 1 # Bernardi--Raugel
            FE_velocity = FiniteElements.getH1BRFiniteElement(xgrid,2)
            FE_pressure = FiniteElements.getP0FiniteElement(xgrid,1)
        elseif FEorder == 2 # Taylor--Hood/Q2xP1
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

        # move integral mean
        pmeanIntegrator = ItemIntegrator(AbstractAssemblyTypeCELL, Identity, DoNotChangeAction(1))
        pressure_mean =  evaluate(pmeanIntegrator,Solution[2]; verbosity = verbosity - 1)
        total_area = sum(FE_velocity.xgrid[CellVolumes], dims=1)[1]
        pressure_mean /= total_area
        for j=1:FE_pressure.ndofs
            Solution[2][j] -= pressure_mean
        end    

        # interpolate
        Interpolation = FEVector{Float64}("Interpolation velocity",FE_velocity)
        append!(Interpolation,"Interpolation pressure",FE_pressure)
        interpolate!(Interpolation[1], exact_velocity!; verbosity = verbosity - 1, bonus_quadorder = 2)
        interpolate!(Interpolation[2], exact_pressure!; verbosity = verbosity - 1, bonus_quadorder = 1)

        # compute L2 and H1 error
        println("divergence = $(evaluate(L2DivergenceErrorEvaluator,Solution[1]))")
        println("divergenceI = $(evaluate(L2DivergenceErrorEvaluator,Interpolation[1]))")
        append!(L2error_velocity,sqrt(evaluate(L2VelocityErrorEvaluator,Solution[1])))
        append!(L2errorInterpolation_velocity,sqrt(evaluate(L2VelocityErrorEvaluator,Interpolation[1])))
        append!(L2error_pressure,sqrt(evaluate(L2PressureErrorEvaluator,Solution[2])))
        append!(L2errorInterpolation_pressure,sqrt(evaluate(L2PressureErrorEvaluator,Interpolation[2])))
        
        # plot final solution
        if (level == nlevels)
            println("\nL2error_velocity")
            Base.show(L2error_velocity)
            println("\nL2errorInterpolation_velocity")
            Base.show(L2errorInterpolation_velocity)
        
            println("\nL2error_pressure")
            Base.show(L2error_pressure)
            println("\nL2errorInterpolation_pressure")
            Base.show(L2errorInterpolation_pressure)

            # split grid into triangles for plotter
            xgrid = split_grid_into(xgrid,Triangle2D)

            # plot triangulation
            PyPlot.figure(1)
            ExtendableGrids.plot(xgrid, Plotter = PyPlot)

            # plot solution
            PyPlot.figure(2)
            nnodes = size(xgrid[Coordinates],2)
            ExtendableGrids.plot(xgrid, Solution[1][1:nnodes]; Plotter = PyPlot)
            PyPlot.figure(3)
            nnodes = size(xgrid[Coordinates],2)
            ExtendableGrids.plot(xgrid, Solution[2][1:nnodes]; Plotter = PyPlot)
        end    
    end    


end


main()
