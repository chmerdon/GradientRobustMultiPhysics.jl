using Printf
using Triangulate
using XGrid

function xgridgen_unitsqure_triangle(maxarea)
    triin=Triangulate.TriangulateIO()
    triin.pointlist=Matrix{Cdouble}([0 0; 1 0; 1 1; 0 1]');
    triin.segmentlist=Matrix{Cint}([1 2 ; 2 3 ; 3 4 ; 4 1 ]')
    triin.segmentmarkerlist=Vector{Int32}([1, 2, 3, 4])
    flags = "pALVa$(@sprintf("%.16f", maxarea))"

    xgrid = XGrid.simplexgrid(flags, triin)
    return xgrid
end


function xgridgen_unitsquare_quad(maxarea, H = 4 // 10, L = 6 // 10; NumberType = Float64)
    coords4nodes = [0 0; L 0; 1 0; 0 H; L H; 1 H;0 1; L 1; 1 1]
    nodes4cells = zeros(Int64,4,4)
    nodes4cells[1,:] = [1, 2, 5, 4];
    nodes4cells[2,:] = [2, 3, 6, 5];
    nodes4cells[3,:] = [4, 5, 8, 7];
    nodes4cells[4,:] = [5, 6, 9, 8];
    nrefinements = ceil(1/log(1.0/maxarea,4))
    nodes4bfaces = [1 2; 2 3; 3 6; 6 9; 9 8; 8 7; 7 4; 4 1];
    bregions = [1,1,2,2,3,3,4,4]
    grid = Grid.Mesh{NumberType}(coords4nodes,nodes4cells,Array{Int32,2}(nodes4bfaces'),bregions,Grid.ElemType2DParallelogram(),nrefinements);

    ncells = size(grid.nodes4cells,2)
    xgrid=ExtendableGrid{NumberType,Int32}()
    xgrid[Coordinates]=Array{NumberType,2}(grid.coords4nodes')
    xgrid[CellRegions]=VectorOfConstants(1,ncells)
    xgrid[CellTypes]=VectorOfConstants(FEXGrid.Parallelogram2D,ncells)
    xgrid[BFaceRegions]=grid.bregions
    xgrid[BFaceTypes]=VectorOfConstants(FEXGrid.Edge1D,ncells)
    xgrid[CellNodes]=Array{Int32,2}(grid.nodes4cells')
    xgrid[BFaceNodes]=Array{Int32,2}(grid.nodes4faces[grid.bfaces,:]')
    xgrid[CoordinateSystem]=Cartesian2D

    return xgrid
end


function gridgen_unitsquare(maxarea, refine_barycentric = false)
    triin=Triangulate.TriangulateIO()
    triin.pointlist=Matrix{Cdouble}([0 0; 1 0; 1 1; 0 1]');
    triin.segmentlist=Matrix{Cint}([1 2 ; 2 3 ; 3 4 ; 4 1 ]')
    triin.segmentmarkerlist=Vector{Int32}([1, 2, 3, 4])
    (triout, vorout)=triangulate("pALVa$(@sprintf("%.16f", maxarea))", triin)
    coords4nodes = Array{Float64,2}(triout.pointlist');
    nodes4cells = Array{Int64,2}(triout.trianglelist');
    if refine_barycentric
        coords4nodes, nodes4cells = Grid.barycentric_refinement(Grid.ElemType2DTriangle(),coords4nodes,nodes4cells)
    end    
    grid = Grid.Mesh{Float64}(coords4nodes,nodes4cells,Grid.ElemType2DTriangle());
    Grid.assign_boundaryregions!(grid,triout.segmentlist,triout.segmentmarkerlist);
    return grid
end


function gridgen_unitsquare_uniform(maxarea, criss::Bool = true, cross::Bool = false)
    nodes4bfaces = [1 2; 2 3; 3 4; 4 1]
    bregions = [1,2,3,4]
        
    if criss && ~cross
        coords4nodes = [0 0; 1 0; 1 1; 0 1]
        nodes4cells = zeros(Int64,2,3)
        nodes4cells[1,:] = [1, 2, 3]
        nodes4cells[2,:] = [1, 3, 4]
        nrefinements = ceil(1/log(0.5/maxarea,4))
        grid = Grid.Mesh{Float64}(coords4nodes,nodes4cells,Array{Int64,2}(nodes4bfaces'),bregions,Grid.ElemType2DTriangle(),nrefinements);
    elseif cross && ~criss
        coords4nodes = [0 0; 1 0; 1 1; 0 1]
        nodes4cells = zeros(Int64,2,3)
        nodes4cells[1,:] = [1, 2, 4]
        nodes4cells[2,:] = [4, 2, 3]
        nrefinements = ceil(1/log(0.5/maxarea,4))
        grid = Grid.Mesh{Float64}(coords4nodes,nodes4cells,Array{Int64,2}(nodes4bfaces'),bregions,Grid.ElemType2DTriangle(),nrefinements);
    elseif criss && cross
        coords4nodes = [0 0; 1 0; 1 1; 0 1]
        nodes4cells = zeros(Int64,1,4)
        nodes4cells[1,:] = [1, 2, 3, 4];
        nrefinements = ceil(1/log(1.0/maxarea,4))
        grid = Grid.Mesh{Float64}(coords4nodes,nodes4cells,Array{Int64,2}(nodes4bfaces'),bregions,Grid.ElemType2DParallelogram(),nrefinements);
        nodes4cells = Grid.divide_into_triangles(Grid.ElemType2DParallelogram(),grid.nodes4cells)
        grid = Grid.Mesh{Float64}(grid.coords4nodes,nodes4cells,Grid.ElemType2DTriangle());
    end
    return grid
end



function gridgen_unitsquare_squares(maxarea, H = 0.4, L = 0.6)
    coords4nodes = [0 0; L 0; 1 0; 0 H; L H; 1 H;0 1; L 1; 1 1]
    nodes4cells = zeros(Int64,4,4)
    nodes4cells[1,:] = [1, 2, 5, 4];
    nodes4cells[2,:] = [2, 3, 6, 5];
    nodes4cells[3,:] = [4, 5, 8, 7];
    nodes4cells[4,:] = [5, 6, 9, 8];
    nrefinements = ceil(1/log(1.0/maxarea,4))
    nodes4bfaces = [1 2; 2 3; 3 6; 6 9; 9 8; 8 7; 7 4; 4 1];
    bregions = [1,1,2,2,3,3,4,4]
    grid = Grid.Mesh{Float64}(coords4nodes,nodes4cells,Array{Int64,2}(nodes4bfaces'),bregions,Grid.ElemType2DParallelogram(),nrefinements);
    return grid
end


function gridgen_unitsquare_single()
    coords4nodes = [0 0; 1 0; 1 1; 0 1]
    nodes4cells = zeros(Int64,1,4)
    nodes4cells[1,:] = [1, 2, 3, 4]
    nodes4bfaces = [1 2; 2 3; 3 4; 4 1];
    bregions = [1,2,3,4]
    grid = Grid.Mesh{Float64}(coords4nodes,nodes4cells,Grid.ElemType2DParallelogram());
    return grid
end