
using FEXGrid

include("PROBLEMdefinitions/GRID_unitsquare.jl")

function main()


    xgrid = xgridgen_unitsqure_triangle(0.1)
    FEXGrid.show(xgrid)
    xgrid[FaceNodes] = XGrid.instantiate(xgrid,FaceNodes)
    FEXGrid.show(xgrid)
    xgrid[CellFaces] = XGrid.instantiate(xgrid,CellFaces)
    FEXGrid.show(xgrid)
    
end


main()
