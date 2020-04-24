
using FEXGrid
using Grid
using FiniteElements
using QuadratureRules

include("PROBLEMdefinitions/GRID_unitinterval.jl")
include("PROBLEMdefinitions/GRID_unitsquare.jl")

function main()

    NumberType = Rational

    xgrid=ExtendableGrid{NumberType,Int32}()
    xgrid[Coordinates]=Array{NumberType,2}([0 0; 4//10 0; 1 0; 0 6//10; 4//10 6//10; 1 6//10;0 1; 4//10 1; 1 1]')
    xCellNodes=VariableTargetAdjacency(Int32)
    xCellTypes=[Triangle2D, Triangle2D, Parallelogram2D, Parallelogram2D, Triangle2D, Triangle2D];
    #xCellTypes=VectorOfConstants(Triangle2D,8)

    append!(xCellNodes,[1,5,4])
    append!(xCellNodes,[1,2,5])

    append!(xCellNodes,[2,3,6,5])
    append!(xCellNodes,[4,5,8,7])
    #append!(xCellNodes,[4,5,8])
    #append!(xCellNodes,[4,8,7])
    #append!(xCellNodes,[2,3,5])
    #append!(xCellNodes,[5,3,6])

    append!(xCellNodes,[5,6,8])
    append!(xCellNodes,[8,6,9])

    xgrid[CellNodes] = xCellNodes
    xgrid[CellTypes] = xCellTypes
    ncells = num_sources(xCellNodes)
    xgrid[CellRegions]=VectorOfConstants(1,ncells)
    xgrid[BFaceRegions]=[1,1,2,2,3,3,4,4]
    xgrid[BFaceTypes]=VectorOfConstants(FEXGrid.Edge1D,ncells)
    xgrid[BFaceNodes]=Array{Int32,2}([1 2; 2 3; 3 6; 6 9; 9 8; 8 7; 7 4; 4 1]')
    xgrid[CoordinateSystem]=Cartesian2D


    show(xgrid[Coordinates])
    show(xgrid[CellVolumes])
    FE = FiniteElements.getP1FiniteElement(xgrid,2)
    FiniteElements.show_new(FE)

    cellvalues = zeros(Real,num_sources(xgrid[CellNodes]),1)
    integrate!(cellvalues, (x) -> x[1] - 1//2, xgrid, 1, 1; NumberType = NumberType, talkative = true)
    show(sum(cellvalues, dims = 1))

    #xgrid[FaceNodes] = XGrid.instantiate(xgrid,FaceNodes)
    #xgrid[CellFaces] = XGrid.instantiate(xgrid,CellFaces)
    #xgrid[CellSigns] = XGrid.instantiate(xgrid,CellSigns)
    #xgrid[CellVolumes] = XGrid.instantiate(xgrid,CellVolumes)
    #xgrid[FaceVolumes] = XGrid.instantiate(xgrid,FaceVolumes)
    #xgrid[BFaces] = XGrid.instantiate(xgrid,BFaces)
    #xgrid[FaceNormals] = XGrid.instantiate(xgrid,FaceNormals)

    #ensure_nodes4faces!(grid)
    #ensure_faces4cells!(grid)
    #ensure_signs4cells!(grid)
    #ensure_volume4cells!(grid)
    #ensure_length4faces!(grid)
    #ensure_normal4faces!(grid)
    #ensure_bfaces!(grid)

    #println("\nxgrid - FaceNodes")
    #show(xgrid[FaceNodes])

    #println("\ngrid - nodes4faces")
    #show(grid.nodes4faces)

    #println("\nxgrid - CellFaces")
    #show(xgrid[CellFaces])


    #println("\nxgrid - CellSigns")
    #show(xgrid[CellSigns])

    #println("\ngrid - signs4cells")
    #show(grid.signs4cells')

    #println("\ngrid - faces4cells")
    #show(grid.faces4cells')

    #println("\nxgrid - CellVolumes")
    #show(xgrid[CellVolumes])

    #println("\ngrid - volume4cells")
    #show(grid.volume4cells)

    #println("\nxgrid - FaceVolumes")
    #show(xgrid[FaceVolumes])

    #println("\ngrid - length4faces")
    #show(grid.length4faces')

    #println("\nxgrid - BFaces")
    #show(xgrid[BFaces])

    #println("\ngrid - bfaces")
    #show(grid.bfaces')

    #println("\nxgrid - FaceNormals")
    #show(xgrid[FaceNormals])

    #println("\ngrid - normal4faces")
    #show(grid.normal4faces')
end


main()
