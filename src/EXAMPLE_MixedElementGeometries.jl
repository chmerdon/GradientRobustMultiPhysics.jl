
using FEXGrid
using ExtendableGrids
using ExtendableSparse
using FiniteElements
using FEOperator
using FESolvePoisson
using QuadratureRules
using VTKView
#ENV["MPLBACKEND"]="qt5agg"
#using PyPlot
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
    xgrid[BFaceRegions]=Array{Int32,1}([1,1,1,1,1,1,1,1])
    xBFaceNodes=Array{Int32,2}([1 2; 2 3; 3 6; 6 9; 9 8; 8 7; 7 4; 4 1]')
    xgrid[BFaceNodes]=xBFaceNodes
    nbfaces = num_sources(xBFaceNodes)
    xgrid[BFaceGeometries]=VectorOfConstants(Edge1D,nbfaces)
    xgrid[CoordinateSystem]=Cartesian2D

    return xgrid
end


function gridgen_triangles()

    NumberType = Float64
    xgrid=ExtendableGrid{NumberType,Int32}()
    xgrid[Coordinates]=Array{NumberType,2}([0 0; 4//10 0; 1 0; 0 6//10; 4//10 6//10; 1 6//10;0 1; 4//10 1; 1 1]')
    xCellNodes=VariableTargetAdjacency(Int32)
    xCellGeometries=VectorOfConstants(Triangle2D, 8);
    
    append!(xCellNodes,[1,5,4])
    append!(xCellNodes,[1,2,5])
    append!(xCellNodes,[4,5,8])
    append!(xCellNodes,[4,8,7])
    append!(xCellNodes,[2,3,5])
    append!(xCellNodes,[5,3,6])
    append!(xCellNodes,[5,6,8])
    append!(xCellNodes,[8,6,9])

    xgrid[CellNodes] = xCellNodes
    xgrid[CellGeometries] = xCellGeometries
    ncells = num_sources(xCellNodes)
    xgrid[CellRegions]=VectorOfConstants{Int32}(1,ncells)
    xgrid[BFaceRegions]=Array{Int32,1}([1,1,2,2,3,3,4,4])
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
    #xgrid = gridgen_triangles()
    nlevels = 6 # nrefinement levels
    diffusion = 1.0
    talkative = true


    function exact_solution!(result,x)
        result[1] = x[1]*x[2]*(x[1]-1)*(x[2]-1)+1
    end    
    function exact_solution_gradient!(result,x)
        result[1] = x[2]*(2*x[1]-1)*(x[2]-1)
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
    PD.boundarytype4bregion = [1]
    PD.boundarydata4bregion = [exact_solution!]
    PD.quadorder4bregion = [4]

    L2error = []
    H1error = []
    for level = 1 : nlevels

        # uniform mesh refinement
        if (level > 1) 
            xgrid = uniform_refine(xgrid)
        end

        # generate FE
        FE = FiniteElements.getH1P1FiniteElement(xgrid,1)
        FiniteElements.show_new(FE)

        # solve Poisson problem
        Solution = FEFunction{Float64}("solution",FE)
        solve!(Solution,PD; talkative = talkative)

        # compute L2 and H1 error
        append!(L2error,L2Error(Solution, exact_solution!, Identity; talkative = talkative, bonus_quadorder = 4))
        append!(H1error,L2Error(Solution, exact_solution_gradient!, Gradient; talkative = talkative, bonus_quadorder = 4))
       
    end    


    println("\nL2error")
    Base.show(L2error)

    println("\nH1error")
    Base.show(H1error)

end


main()
