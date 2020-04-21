
using FEXGrid
using Grid

include("PROBLEMdefinitions/GRID_unitsquare.jl")

function main()


    xgrid = xgridgen_unitsqure_triangle(0.05)
    xgrid[FaceNodes] = XGrid.instantiate(xgrid,FaceNodes)
    xgrid[CellFaces] = XGrid.instantiate(xgrid,CellFaces)
    xgrid[CellVolumes] = XGrid.instantiate(xgrid,CellVolumes)
    xgrid[FaceVolumes] = XGrid.instantiate(xgrid,FaceVolumes)
    xgrid[BFaces] = XGrid.instantiate(xgrid,BFaces)
    xgrid[FaceNormals] = XGrid.instantiate(xgrid,FaceNormals)

    grid = gridgen_unitsquare(0.05,false)
    ensure_nodes4faces!(grid)
    ensure_faces4cells!(grid)
    ensure_volume4cells!(grid)
    ensure_length4faces!(grid)
    ensure_normal4faces!(grid)
    ensure_bfaces!(grid)

    println("\nxgrid - FaceNodes")
    show(xgrid[FaceNodes])

    println("\ngrid - nodes4faces")
    show(grid.nodes4faces')

    println("\nxgrid - CellFaces")
    show(xgrid[CellFaces])

    println("\ngrid - nodes4faces")
    show(grid.faces4cells')

    println("\nxgrid - CellVolumes")
    show(xgrid[CellVolumes])

    println("\ngrid - volume4cells")
    show(grid.volume4cells')

    println("\nxgrid - FaceVolumes")
    show(xgrid[FaceVolumes])

    println("\ngrid - length4faces")
    show(grid.length4faces')

    println("\nxgrid - BFaces")
    show(xgrid[BFaces])

    println("\ngrid - bfaces")
    show(grid.bfaces')

    println("\nxgrid - FaceNormals")
    show(xgrid[FaceNormals])

    println("\ngrid - normal4faces")
    show(grid.normal4faces')
end


main()
