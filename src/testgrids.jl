#####################
# REFERENCE DOMAINS #
#####################

using ExtendableGrids

function reference_domain(::Type{<:Edge1D}; scale = [1])
    xgrid=ExtendableGrid{Float64,Int32}()
    xgrid[Coordinates]=Array{Float64,2}([0; scale]')
    xgrid[CellNodes] = Array{Int32,2}([1 2]')
    xgrid[CellGeometries] = VectorOfConstants(Edge1D,1);
    xgrid[CellRegions]=VectorOfConstants{Int32}(1,1)
    xgrid[BFaceRegions]=Array{Int32,1}([1,2])
    xgrid[BFaceNodes]=Array{Int32,2}([1; 2]')
    xgrid[BFaceGeometries]=VectorOfConstants(Vertex0D,2)
    xgrid[CoordinateSystem]=Cartesian1D
    return xgrid
end

function reference_domain(ET::Type{<:Triangle2D}; scale = [1,1])
    xgrid=ExtendableGrid{Float64,Int32}()
    xgrid[Coordinates]=Array{Float64,2}([0 0; scale[1] 0; 0 scale[2]]')
    xgrid[CellNodes] = Array{Int32,2}([1 2 3]')
    xgrid[CellGeometries] = VectorOfConstants(Triangle2D,1);
    xgrid[CellRegions]=VectorOfConstants{Int32}(1,1)
    xgrid[BFaceRegions]=Array{Int32,1}([1,2,3])
    xgrid[BFaceNodes]=Array{Int32,2}([1 2; 2 3; 3 1]')
    xgrid[BFaceGeometries]=VectorOfConstants(Edge1D,3)
    xgrid[CoordinateSystem]=Cartesian2D
    return xgrid
end

function reference_domain(ET::Type{<:Quadrilateral2D}; scale = [1,1])
    xgrid=ExtendableGrid{Float64,Int32}()
    xgrid[Coordinates]=Array{Float64,2}([0 0; scale[1] 0; scale[1] scale[2]; 0 scale[2]]')
    xgrid[CellNodes] = Array{Int32,2}([1 2 3 4]')
    xgrid[CellGeometries] = VectorOfConstants(ET,1);
    xgrid[CellRegions]=VectorOfConstants{Int32}(1,1)
    xgrid[BFaceRegions]=Array{Int32,1}([1,2,3,4])
    xgrid[BFaceNodes]=Array{Int32,2}([1 2; 2 3; 3 4; 4 1]')
    xgrid[BFaceGeometries]=VectorOfConstants(Edge1D,4)
    xgrid[CoordinateSystem]=Cartesian2D
    return xgrid
end

function reference_domain(ET::Type{<:Tetrahedron3D}; scale = [1,1,1])
    xgrid=ExtendableGrid{Float64,Int32}()
    xgrid[Coordinates]=Array{Float64,2}([0 0 0; scale[1] 0 0; 0 scale[2] 0; 0 0 scale[3]]')
    xgrid[CellNodes] = Array{Int32,2}([1 2 3 4]')
    xgrid[CellGeometries] = VectorOfConstants(ET,1);
    xgrid[CellRegions]=VectorOfConstants{Int32}(1,1)
    xgrid[BFaceRegions]=Array{Int32,1}([1,2,3,4])
    xgrid[BFaceNodes]=Array{Int32,2}([1 3 2; 1 2 4; 2 3 4; 3 1 4]')
    xgrid[BFaceGeometries]=VectorOfConstants(Triangle2D,4)
    xgrid[CoordinateSystem]=Cartesian3D
    return xgrid
end

function reference_domain(ET::Type{<:Hexahedron3D}; scale = [1,1,1])
    xgrid=ExtendableGrid{Float64,Int32}()
    xgrid[Coordinates]=Array{Float64,2}([0 0 0; scale[1] 0 0; 0 scale[2] 0; 0 0 scale[3]; scale[1] scale[2] 0; scale[1] 0 scale[3]; 0 scale[2] scale[3]; scale[1] scale[2] scale[3]]')
    xgrid[CellNodes] = Array{Int32,2}([1 2 3 4 5 6 7 8]')
    xgrid[CellGeometries] = VectorOfConstants(ET,1);
    xgrid[CellRegions]=VectorOfConstants{Int32}(1,1)
    xgrid[BFaceRegions]=Array{Int32,1}([1,2,3,4,5,6])
    xgrid[BFaceNodes]=Array{Int32,2}([1 3 5 2; 1 2 6 4; 2 5 8 6;5 3 7 8;3 1 4 7;4 6 8 7]')
    xgrid[BFaceGeometries]=VectorOfConstants(Parallelogram2D,6)
    xgrid[CoordinateSystem]=Cartesian3D
    return xgrid
end

# unit cube as one cell with six boundary regions (bottom, front, right, back, left, top)
function grid_unitcube(EG::Type{<:Hexahedron3D})
    return reference_domain(EG)
end

# unit cube as six tets with six boundary regions (bottom, front, right, back, left, top)
function grid_unitcube(::Type{Tetrahedron3D})
    xgrid=ExtendableGrid{Float64,Int32}()
    xgrid[Coordinates]=Array{Float64,2}([0 0 0; 1 0 0; 1 1 0; 0 1 0; 0 0 1; 1 0 1; 1 1 1; 0 1 1]')

    xCellNodes=Array{Int32,2}([1 2 3 7; 1 3 4 7; 1 5 6 7; 1 8 5 7; 1 6 2 7;1 4 8 7]')
    xgrid[CellNodes] = xCellNodes
    xgrid[CellGeometries] = VectorOfConstants(Tetrahedron3D,6);
    ncells = num_sources(xCellNodes)
    xgrid[CellRegions]=VectorOfConstants{Int32}(1,ncells)
    xgrid[BFaceRegions]=Array{Int32,1}([1,1,2,2,3,3,4,4,5,5,6,6])
    xBFaceNodes=Array{Int32,2}([1 3 2; 1 4 3; 1 2 6;1 6 5;2 3 7;2 7 6;3 4 7;7 4 8;8 4 1;1 5 8; 5 6 7; 5 7 8]')
    xgrid[BFaceNodes]=xBFaceNodes
    nbfaces = num_sources(xBFaceNodes)
    xgrid[BFaceGeometries]=VectorOfConstants(Triangle2D,nbfaces)
    xgrid[CoordinateSystem]=Cartesian3D
    return xgrid
end


# unit square as one cell with four boundary regions (bottom, right, top, left)
function grid_unitsquare(EG::Type{<:Quadrilateral2D})
    return reference_domain(EG)
end

# unit square as two triangles with four boundary regions (bottom, right, top, left)
function grid_unitsquare(::Type{<:Triangle2D})
    xgrid=ExtendableGrid{Float64,Int32}()
    xgrid[Coordinates]=Array{Float64,2}([0 0; 1 0; 1 1; 0 1]')
    xgrid[CellNodes]=Array{Int32,2}([1 2 3; 1 3 4]')
    xgrid[CellGeometries]=VectorOfConstants(Triangle2D,2)
    xgrid[CellRegions]=VectorOfConstants{Int32}(1,2)
    xgrid[BFaceRegions]=Array{Int32,1}([1,1,2,2,3,3,4,4])
    xgrid[BFaceNodes]=Array{Int32,2}([1 2; 2 3; 3 4; 4 1]')
    xgrid[BFaceGeometries]=VectorOfConstants(Edge1D,4)
    xgrid[CoordinateSystem]=Cartesian2D
    return xgrid
end

# unit suqare as mixed triangles and squares with four boundary regions (bottom, right, top, left)
function grid_unitsquare_mixedgeometries()

    xgrid=ExtendableGrid{Float64,Int32}()
    xgrid[Coordinates]=Array{Float64,2}([0 0; 4//10 0; 1 0; 0 6//10; 4//10 6//10; 1 6//10;0 1; 4//10 1; 1 1]')
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
    xgrid[BFaceRegions]=Array{Int32,1}([1,1,2,2,3,3,4,4])
    xBFaceNodes=Array{Int32,2}([1 2; 2 3; 3 6; 6 9; 9 8; 8 7; 7 4; 4 1]')
    xgrid[BFaceNodes]=xBFaceNodes
    nbfaces = num_sources(xBFaceNodes)
    xgrid[BFaceGeometries]=VectorOfConstants(Edge1D,nbfaces)
    xgrid[CoordinateSystem]=Cartesian2D

    return xgrid
end