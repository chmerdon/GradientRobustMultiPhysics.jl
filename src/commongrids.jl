#####################
# REFERENCE DOMAINS #
#####################

# reference domain generated from data in shape_specs
function reference_domain(EG::Type{<:AbstractElementGeometry}; scale = [1,1,1])
    xgrid=ExtendableGrid{Float64,Int32}()
    xCoordinates=Array{Float64,2}(refcoords_for_geometry(EG)')
    for j = 1 : size(xCoordinates,1)
        xCoordinates[j,:] .*= scale[j]
    end
    xgrid[Coordinates] = xCoordinates
    xCellNodes = zeros(Int32,nnodes_for_geometry(EG),1)
    xCellNodes[:] = 1:nnodes_for_geometry(EG)
    xgrid[CellNodes] = xCellNodes
    xgrid[CellGeometries] = VectorOfConstants(EG,1);
    xgrid[CellRegions]=VectorOfConstants{Int32}(1,1)
    xgrid[BFaceRegions]=Array{Int32,1}(1:nfaces_for_geometry(EG))
    xgrid[BFaceNodes]=Array{Int32,2}(face_enum_rule(EG)')
    xgrid[BFaceGeometries]=VectorOfConstants(facetype_of_cellface(EG, 1), nfaces_for_geometry(EG))
    if dim_element(EG) == 0
        xgrid[CoordinateSystem]=Cartesian0D
    elseif dim_element(EG) == 1
        xgrid[CoordinateSystem]=Cartesian1D
    elseif dim_element(EG) == 2
        xgrid[CoordinateSystem]=Cartesian2D
    elseif dim_element(EG) == 3
        xgrid[CoordinateSystem]=Cartesian3D
    end
    return xgrid
end

# unit cube as one cell with six boundary regions (bottom, front, right, back, left, top)
function grid_unitcube(EG::Type{<:Hexahedron3D}; scale = [1,1,1])
    return reference_domain(EG; scale = scale)
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