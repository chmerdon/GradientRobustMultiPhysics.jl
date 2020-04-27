
using FEXGrid
using Grid
using FiniteElements
using QuadratureRules
using VTKView
using PyPlot
using BenchmarkTools

include("PROBLEMdefinitions/GRID_unitinterval.jl")
include("PROBLEMdefinitions/GRID_unitsquare.jl")

function main()

    NumberType = Float64
    mixed_geometries = true

    xgrid=ExtendableGrid{NumberType,Int32}()
    xgrid[Coordinates]=Array{NumberType,2}([0 0; 4//10 0; 1 0; 0 6//10; 4//10 6//10; 1 6//10;0 1; 4//10 1; 1 1]')
    xCellNodes=VariableTargetAdjacency(Int32)
    if mixed_geometries
        xCellTypes=[Triangle2D, Triangle2D, Parallelogram2D, Parallelogram2D, Triangle2D, Triangle2D];
    else
        xCellTypes=VectorOfConstants(Triangle2D,8)
    end    

    append!(xCellNodes,[1,5,4])
    append!(xCellNodes,[1,2,5])

    if mixed_geometries
        append!(xCellNodes,[2,3,6,5])
        append!(xCellNodes,[4,5,8,7])
    else    
        append!(xCellNodes,[4,5,8])
        append!(xCellNodes,[4,8,7])
        append!(xCellNodes,[2,3,5])
        append!(xCellNodes,[5,3,6])
    end    

    append!(xCellNodes,[5,6,8])
    append!(xCellNodes,[8,6,9])

    xgrid[CellNodes] = xCellNodes
    xgrid[CellTypes] = xCellTypes
    ncells = num_sources(xCellNodes)
    xgrid[CellRegions]=VectorOfConstants(1,ncells)
    xgrid[BFaceRegions]=Array{Int32,1}([1,1,2,2,3,3,4,4])
    xgrid[BFaceTypes]=VectorOfConstants(FEXGrid.Edge1D,ncells)
    xgrid[BFaceNodes]=Array{Int32,2}([1 2; 2 3; 3 6; 6 9; 9 8; 8 7; 7 4; 4 1]')
    xgrid[CoordinateSystem]=Cartesian2D

    function f(result,x)
        result[1] = x[1] - 1//2
    end    

    # integral of f over cells
    cellvalues = zeros(Real,num_sources(xgrid[CellNodes]),1)
    @time integrate!(cellvalues, xgrid, AbstractAssemblyTypeCELL, f, 1, 1, NumberType; talkative = true)
    show(sum(cellvalues, dims = 1))

    # integral of f over faces
    facevalues = zeros(Real,num_sources(xgrid[FaceNodes]),1)
    @time integrate!(facevalues, xgrid, AbstractAssemblyTypeFACE, f, 1, 1, NumberType; talkative = true)
    
    # integral of f over boundary faces
    show(sum(facevalues[xgrid[BFaces]], dims = 1))



    println("")
    FE = FiniteElements.registerP1FiniteElement(xgrid,2)
    FiniteElements.show_new(FE)

    Velocity = FEFunction{NumberType}("velocity",FE)
    Velocity[3] = 1

    #xgrid = split_grid_into(xgrid,Triangle2D)
    #XGrid.plot(xgrid; Plotter = VTKView)

end


main()
