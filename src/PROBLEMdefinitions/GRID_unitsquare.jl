using Printf

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


function gridgen_unitsquare_squares(maxarea)
    coords4nodes = [0 0; 1 0; 1 1; 0 1]
    nodes4cells = zeros(Int64,1,4)
    nodes4cells[1,:] = [1, 2, 3, 4];
    nrefinements = ceil(1/log(1.0/maxarea,4))
    nodes4bfaces = [1 2;2 3;3 4;4 1]
    bregions = [1,2,3,4]
    grid = Grid.Mesh{Float64}(coords4nodes,nodes4cells,nodes4bfaces,bregions,Grid.ElemType2DParallelogram(),nrefinements);
    return grid
end