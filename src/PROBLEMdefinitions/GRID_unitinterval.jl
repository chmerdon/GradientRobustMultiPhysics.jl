function gridgen_unitinterval(maxarea)
    grid = Grid.Mesh{Float64}(Array{Float64,2}(Array{Float64,2}([0,0.5,1]')'),Array{Int64,2}([1 2;2 3]),Grid.ElemType1DInterval(),ceil(log2(1/maxarea)));
    Grid.ensure_bfaces!(grid)
    return grid;
end

function xgridgen_unitinterval(maxarea)
    grid = Grid.Mesh{Float64}(Array{Float64,2}(Array{Float64,2}([0,0.5,1]')'),Array{Int64,2}([1 2;2 3]),Grid.ElemType1DInterval(),ceil(log2(1/maxarea)));
    Grid.ensure_bfaces!(grid)

    ncells = size(grid.nodes4cells,2)
    xgrid=ExtendableGrid{Float64,Int32}()
    xgrid[Coordinates]=Array{Float64,2}(grid.coords4nodes')
    xgrid[CellRegions]=VectorOfConstants(1,ncells)
    xgrid[CellTypes]=VectorOfConstants(FEXGrid.Simplex1D_Cartesian1D,ncells)
    xgrid[BFaceRegions]=grid.bregions
    xgrid[BFaceTypes]=VectorOfConstants(FEXGrid.Point0D,ncells)
    xgrid[CellNodes]=Array{Int32,2}(grid.nodes4cells')
    xgrid[BFaceNodes]=Array{Int32,2}(grid.nodes4faces[grid.bfaces,:]')

    return xgrid
end