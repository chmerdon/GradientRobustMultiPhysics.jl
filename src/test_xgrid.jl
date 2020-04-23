
using FEXGrid
using Grid
using FiniteElements

include("PROBLEMdefinitions/GRID_unitinterval.jl")
include("PROBLEMdefinitions/GRID_unitsquare.jl")

function main()

    #xgrid = xgridgen_unitinterval(0.1); grid = gridgen_unitinterval(0.1)
    xgrid = xgridgen_unitsqure_triangle(0.25); grid = gridgen_unitsquare(0.25,false)
    #xgrid = xgridgen_unitsquare_quad(0.5); grid = gridgen_unitsquare_squares(0.5);

    xgrid[FaceNodes] = XGrid.instantiate(xgrid,FaceNodes)
    xgrid[CellFaces] = XGrid.instantiate(xgrid,CellFaces)
    xgrid[CellSigns] = XGrid.instantiate(xgrid,CellSigns)
    xgrid[CellVolumes] = XGrid.instantiate(xgrid,CellVolumes)
    xgrid[FaceVolumes] = XGrid.instantiate(xgrid,FaceVolumes)
    xgrid[BFaces] = XGrid.instantiate(xgrid,BFaces)
    xgrid[FaceNormals] = XGrid.instantiate(xgrid,FaceNormals)

    FE = FiniteElements.getP1FiniteElement(xgrid,2)
    FiniteElements.show_new(FE)


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
    #show(grid.nodes4faces')

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
    #show(grid.volume4cells')

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
