
using FEXGrid
using ExtendableGrids
using ExtendableSparse
using FiniteElements
using FEOperator
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
    xgrid[BFaceRegions]=Array{Int32,1}([1,1,1,1,1,1,2,2])
    xBFaceNodes=Array{Int32,2}([1 2; 2 3; 3 6; 6 9; 9 8; 8 7; 7 4; 4 1]')
    xgrid[BFaceNodes]=xBFaceNodes
    nbfaces = num_sources(xBFaceNodes)
    xgrid[BFaceGeometries]=VectorOfConstants(Edge1D,nbfaces)
    xgrid[CoordinateSystem]=Cartesian2D

    return xgrid
end


function main()


    # initial grid
    xgrid = gridgen_mixedEG()
    #xgrid = split_grid_into(xgrid,Triangle2D)
    nlevels = 5 # nrefinement levels
    diffusion = 1.0
    talkative = true


    function exact_solution!(result,x)
        result[1] = x[1]*x[2]*(x[1]-1)*(x[2]-1) + x[1]
    end    
    function bnd_data_left!(result,x)
        result[1] = 0.0
    end    
    function bnd_data_rest!(result,x)
        result[1] = x[1]
    end    
    function exact_solution_gradient!(result,x)
        result[1] = x[2]*(2*x[1]-1)*(x[2]-1) + 1.0
        result[2] = x[1]*(2*x[2]-1)*(x[1]-1)
    end    
    function exact_solution_laplacian!(result,x)
        result[1] = -diffusion*(2*x[2]*(x[2]-1) + 2*x[1]*(x[1]-1))
    end    

    # problem description 
    PD = PoissonProblemDescription()
    PD.name = "Test problem"
    PD.diffusion = diffusion
    PD.quadorder4diffusion = 0
    PD.volumedata4region = [exact_solution_laplacian!]
    PD.quadorder4region = [2]
    PD.boundarytype4bregion = [BestapproxDirichletBoundary, HomogeneousDirichletBoundary]
    PD.boundarydata4bregion = [bnd_data_rest!, bnd_data_left!]
    PD.quadorder4bregion = [1,0]

    FESolvePoisson.show(PD)

    L2error = []
    H1error = []
    L2errorInterpolation = []
    H1errorInterpolation = []
    for level = 1 : nlevels

        # uniform mesh refinement
        if (level > 1) 
            xgrid = uniform_refine(xgrid)
        end

        # generate FE
        FE = FiniteElements.getH1P1FiniteElement(xgrid,1)
        FiniteElements.show(FE)

        # solve Poisson problem
        Solution = FEVector{Float64}("solution",FE)
        FiniteElements.show(Solution)
        append!(Solution, "interpolation", FE)

        solve!(Solution[1],PD; talkative = talkative)
        interpolate!(Solution[2], exact_solution!)

        # compute L2 and H1 error
        append!(L2error,L2Error(Solution[1], exact_solution!, Identity; talkative = talkative, bonus_quadorder = 4))
        append!(H1error,L2Error(Solution[1], exact_solution_gradient!, Gradient; talkative = talkative, bonus_quadorder = 4))
        append!(L2errorInterpolation,L2Error(Solution[2], exact_solution!, Identity; talkative = talkative, bonus_quadorder = 4))
        append!(H1errorInterpolation,L2Error(Solution[2], exact_solution_gradient!, Gradient; talkative = talkative, bonus_quadorder = 4))
       
        # plot final solution
        if (level == nlevels)
            # split grid into triangles for plotter
            xgrid = split_grid_into(xgrid,Triangle2D)

            # plot triangulation
            PyPlot.figure(1)
            ExtendableGrids.plot(xgrid, Plotter = PyPlot)

            # plot solution
            PyPlot.figure(2)
            ExtendableGrids.plot(xgrid, Solution[1][:]; Plotter = PyPlot)
        end    
    end    


    println("\nL2error")
    Base.show(L2error)
    println("\nL2errorInterpolation")
    Base.show(L2errorInterpolation)

    println("\nH1error")
    Base.show(H1error)
    println("\nH1errorInterpolation")
    Base.show(H1errorInterpolation)


end


main()
