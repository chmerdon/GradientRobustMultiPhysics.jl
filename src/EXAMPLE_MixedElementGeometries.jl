
using FEXGrid
using ExtendableGrids
using ExtendableSparse
using FiniteElements
using FEAssembly
using FESolvePoisson
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
    FEorder = 2 # optimal convergence order of finite element
    verbosity = 3 # deepness of messaging (the larger, the more)

    # define expected solution, boundary data and volume data
    diffusion = 1.0
    function exact_solution!(result,x)
        result[1] = x[1]*x[2]*(x[1]-1)*(x[2]-1) + x[1]
    end    
    function bnd_data_left!(result,x)
        result[1] = 0.0
    end    
    function bnd_data_right!(result,x)
        result[1] = 1.0
    end    
    function bnd_data_rest!(result,x)
        result[1] = x[1]
    end    
    function exact_solution_gradient!(result,x)
        result[1] = x[2]*(2*x[1]-1)*(x[2]-1) + 1.0
        result[2] = x[1]*(2*x[2]-1)*(x[1]-1)
    end    
    function exact_solution_rhs!(result,x)
        # diffusion part
        result[1] = -diffusion*(2*x[2]*(x[2]-1) + 2*x[1]*(x[1]-1))
        # convection part
        result[1] += x[2]*(2*x[1]-1)*(x[2]-1) + 1.0
    end    
    function convection!(result,x)
        result[1] = 1.0
        result[2] = 0.0
    end

    # fill problem description 
    PD = PoissonProblemDescription()
    PD.name = "Test problem"
    PD.diffusion = diffusion
    PD.quadorder4diffusion = 0
    PD.convection = convection!
    PD.quadorder4convection = 0
    PD.volumedata4region = [exact_solution_rhs!]
    PD.quadorder4region = [2]
    PD.boundarytype4bregion = [BestapproxDirichletBoundary, InterpolateDirichletBoundary, HomogeneousDirichletBoundary]
    PD.boundarydata4bregion = [bnd_data_rest!, bnd_data_right!, bnd_data_left!]
    PD.quadorder4bregion = [1,0]
    if verbosity > 0
        FESolvePoisson.show(PD)
    end    

    # define ItemIntegrators for L2/H1 error computation
    L2ErrorEvaluator = L2ErrorIntegrator(exact_solution!, Identity, 2; bonus_quadorder = 4)
    H1ErrorEvaluator = L2ErrorIntegrator(exact_solution_gradient!, Gradient, 2; bonus_quadorder = 3)
    L2error = []
    H1error = []
    L2errorInterpolation = []
    H1errorInterpolation = []

    # loop over levels
    for level = 1 : nlevels

        # uniform mesh refinement
        if (level > 1) 
            xgrid = uniform_refine(xgrid)
        end

        # generate FE
        if FEorder == 1
            FE = FiniteElements.getH1P1FiniteElement(xgrid,1)
        elseif FEorder == 2
            FE = FiniteElements.getH1P2FiniteElement(xgrid,1)
        end        
        if verbosity > 2
            FiniteElements.show(FE)
        end    

        # solve Poisson problem
        Solution = FEVector{Float64}("solution",FE)
        append!(Solution, "interpolation", FE)
        if verbosity > 2
            FiniteElements.show(Solution)
        end    

        solve!(Solution[1],PD; verbosity = verbosity - 1)
        interpolate!(Solution[2], exact_solution!; verbosity = verbosity - 1)

        # compute L2 and H1 error
        append!(L2error,sqrt(evaluate(L2ErrorEvaluator,Solution[1])))
        append!(L2errorInterpolation,sqrt(evaluate(L2ErrorEvaluator,Solution[2])))
        append!(H1error,sqrt(evaluate(H1ErrorEvaluator,Solution[1])))
        append!(H1errorInterpolation,sqrt(evaluate(H1ErrorEvaluator,Solution[2])))
        
        # plot final solution
        if (level == nlevels)
            println("\nL2error")
            Base.show(L2error)
            println("\nL2errorInterpolation")
            Base.show(L2errorInterpolation)
        
            println("\nH1error")
            Base.show(H1error)
            println("\nH1errorInterpolation")
            Base.show(H1errorInterpolation)

            # split grid into triangles for plotter
            xgrid = split_grid_into(xgrid,Triangle2D)

            # plot triangulation
            PyPlot.figure(1)
            ExtendableGrids.plot(xgrid, Plotter = PyPlot)

            # plot solution
            PyPlot.figure(2)
            nnodes = size(xgrid[Coordinates],2)
            ExtendableGrids.plot(xgrid, Solution[1][1:nnodes]; Plotter = PyPlot)
        end    
    end    


end


main()
