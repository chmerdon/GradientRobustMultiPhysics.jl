module FEXGrid

using ExtendableGrids

export FaceNodes, FaceGeometries, CellFaces, CellSigns, CellVolumes, FaceVolumes, FaceRegions, FaceCells, FaceNormals, BFaces, BFaceCellPos
export nfaces_per_cell, facetype_of_cellface

include("XGridAssemblyJunctions.jl");
export AbstractAssemblyType, AbstractAssemblyTypeCELL, AbstractAssemblyTypeFACE, AbstractAssemblyTypeBFACE, AbstractAssemblyTypeBFACECELL
export xrefFACE2xrefCELL
export GridComponentNodes4AssemblyType, GridComponentTypes4AssemblyType, GridComponentVolumes4AssemblyType, GridComponentRegions4AssemblyType

include("L2GTransformer.jl");
export L2GTransformer, update!, eval!, mapderiv!


export split_grid_into


# additional ExtendableGrids adjacency types 
abstract type FaceNodes <: AbstractGridAdjacency end
abstract type CellFaces <: AbstractGridAdjacency end
abstract type CellSigns <: AbstractGridAdjacency end
abstract type CellVolumes <: AbstractGridFloatArray1D end
abstract type FaceVolumes <: AbstractGridFloatArray1D end
abstract type FaceCells <: AbstractGridAdjacency end
abstract type FaceNormals <: AbstractGridFloatArray2D end
abstract type FaceGeometries <: AbstractElementGeometries end
abstract type FaceRegions <: AbstractElementRegions end
abstract type BFaces <: AbstractGridIntegerArray1D end
abstract type BFaceCellPos <: AbstractGridIntegerArray1D end # position of bface in adjacent cell
abstract type BFaceVolumes <: AbstractGridFloatArray1D end

# additional ExtendableGrids adjacency types for finite elements
abstract type CellDofs <: AbstractGridAdjacency end
abstract type FaceDofs <: AbstractGridAdjacency end
abstract type Coefficients <: AbstractGridComponent end

# functions that specify the number of faces of a celltype
nfaces_per_cell(::Type{<:Edge1D}) = 2
nfaces_per_cell(::Type{<:Triangle2D}) = 3
nfaces_per_cell(::Type{<:Tetrahedron3D}) = 4
nfaces_per_cell(::Type{<:Quadrilateral2D}) = 4

# functions that specify the local enumeration of faces
face_enum_rule(::Type{<:Edge1D}) = [1; 2]
face_enum_rule(::Type{<:Triangle2D}) = [1 2; 2 3; 3 1]
face_enum_rule(::Type{<:Quadrilateral2D}) = [1 2; 2 3; 3 4; 4 1]

# functions that specify number of nodes on the k-th cell face
# why k?: think about ElementTypes that have faces of different nature
# e.g. a pyramid with rectangular basis and triangular sides
# this maybe requires a ordering rule for the nodes in the element
# (e.g. for the pyramid first four nodes for the basis come first)
nnodes_per_cellface(::Type{<:Edge1D}, k) = 1
nnodes_per_cellface(::Type{<:Triangle2D}, k) = 2
nnodes_per_cellface(::Type{<:Tetrahedron3D}, k) = 3
nnodes_per_cellface(::Type{<:Quadrilateral2D}, k) = 2

# functions that specify the facetype of the k-th cellface
facetype_of_cellface(::Type{<:Edge1D}, k) = Vertex0D
facetype_of_cellface(::Type{<:Triangle2D}, k) = Edge1D
facetype_of_cellface(::Type{<:Tetrahedron3D}, k) = Triangle2D
facetype_of_cellface(::Type{<:Quadrilateral2D}, k) = Edge1D


# functions that tell how to split one ElementGeometry into another
split_rule(::Type{Triangle2D}, ::Type{Triangle2D}) = reshape([1,2,3],1,3)
split_rule(::Type{<:Quadrilateral2D}, ::Type{Triangle2D}) = [1 2 3;1 3 4]

# function that generates a new simplexgrid from a mixed grid
function split_grid_into(source_grid::ExtendableGrid{T,K}, targetgeometry::Type{Triangle2D}) where {T,K}
    xgrid=ExtendableGrid{T,K}()
    xgrid[Coordinates]=source_grid[Coordinates]
    oldCellTypes = source_grid[CellGeometries]
    EG = unique(oldCellTypes)
    
    split_rules = Array{Array{Int,2},1}(undef,length(EG))
    for j = 1 : length(EG)
        split_rules[j] = split_rule(EG[j],targetgeometry)
    end
    xCellNodes=[]
    oldCellNodes=source_grid[CellNodes]
    nnodes4cell = 0
    ncells = 0
    cellET = Triangle2D
    iEG = 1
    for cell = 1 : num_sources(oldCellNodes)
        nnodes4cell = num_targets(oldCellNodes,cell)
        cellET = oldCellTypes[cell]
        iEG = findfirst(isequal(cellET), EG)
        for j = 1 : size(split_rules[iEG],1), k = 1 : size(split_rules[iEG],2)
            append!(xCellNodes,oldCellNodes[split_rules[iEG][j,k],cell])
        end    
        ncells += size(split_rules[iEG],1)
    end
    xCellNodes = reshape(xCellNodes,3,ncells)
    xgrid[CellNodes] = Array{Int32,2}(xCellNodes)
    xgrid[CellGeometries] = VectorOfConstants(Triangle2D,8)
    xgrid[CellRegions]=ones(Int32,ncells)
    xgrid[BFaceNodes]=source_grid[BFaceNodes]
    xgrid[BFaceRegions]=source_grid[BFaceRegions]
    xgrid[BFaceGeometries]=VectorOfConstants(Edge1D,num_sources(source_grid[BFaceNodes]))
    xgrid[CoordinateSystem]=source_grid[CoordinateSystem]
    return xgrid
end

# show function for ExtendableGrids and defined Components in its Dict
function show(xgrid::ExtendableGrid)

    dim = size(xgrid[Coordinates],1)
    nnodes = num_sources(xgrid[Coordinates])
    ncells = num_sources(xgrid[CellNodes])
    
	println("ExtendableGrids");
    println("======");
	println("dim: $(dim)")
	println("nnodes: $(nnodes)")
    println("ncells: $(ncells)")
    if haskey(ExtendableGrids.components,FaceNodes)
        nfaces = num_sources(xgrid[FaceNodes])
        println("nfaces: $(nfaces)")
    else
        println("nfaces: (FaceNodes not instantiated)")
    end
    println("")
    println("Components");
    println("===========");
    for tuple in ExtendableGrids.components
        println("> $(tuple[1])")
    end
end

# FaceNodes = nodes for each face (implicitly defines the enumerations of faces)
function ExtendableGrids.instantiate(xgrid::ExtendableGrid, ::Type{FaceNodes})

    # each edge consists of dim nodes (beware: has to be replaced later if triangulation of submanifolds are included)
    dim = size(xgrid[Coordinates],1) 
    xCellNodes = xgrid[CellNodes]
    ncells = num_sources(xCellNodes)
    xCellGeometries = xgrid[CellGeometries]

    # helper field to store all face nodes (including reversed duplicates)
    xFaceNodesAll = VariableTargetAdjacency(Int32)
    
    types4allfaces = []
    nfaces = 0
    swap = 0
    current_face = zeros(Int32,max_num_targets_per_source(xCellNodes)) # should be large enough to store largest nnodes_per_cellface
    faces_per_cell = 0
    nodes_per_cellface = 0
    cellEG = Triangle2D
    for cell = 1 : ncells
        cellEG = xCellGeometries[cell]
        faces_per_cell = nfaces_per_cell(cellEG)
        face_rule = face_enum_rule(cellEG)

        for k = 1 : faces_per_cell
            nodes_per_cellface = nnodes_per_cellface(cellEG, k)
            
            # get face nodes
            for j = 1 : nodes_per_cellface
                current_face[j] = xCellNodes[face_rule[k,j],cell]; 
            end

            # bubble_sort face nodes
            for j = nodes_per_cellface:-1:2
                for j2 = 1 : j-1
                    if current_face[j2] < current_face[j2+1]
                        swap = current_face[j2+1]
                        current_face[j2+1] = current_face[j2]
                        current_face[j2] = swap
                    end    
                end
            end

            append!(xFaceNodesAll,current_face[1:nodes_per_cellface])
            types4allfaces = [types4allfaces; facetype_of_cellface(xCellGeometries[cell], k)]
            nfaces += 1
        end    
    end

    # remove duplicates and assign FaceGeometries
    xFaceNodes = VariableTargetAdjacency(Int32)
    xFaceGeometries = []
    idx = 0
    for face=1:nfaces
        idx = findfirst(j->all(i->xFaceNodesAll[i,face] == xFaceNodesAll[i,j],1:dim),1:nfaces)
        if idx == face
            append!(xFaceNodes,xFaceNodesAll[:,idx])
            xFaceGeometries = [xFaceGeometries; types4allfaces[idx]]
        end
    end
    xgrid[FaceGeometries] = Array{DataType,1}(xFaceGeometries)
    xFaceNodes
end

# CellFaces = faces for each cell
function ExtendableGrids.instantiate(xgrid::ExtendableGrid, ::Type{CellFaces})

    # init CellFaces
    xFaceNodes = xgrid[FaceNodes]
    xCellFaces = VariableTargetAdjacency(Int32)

    # get links to other stuff
    dim = size(xgrid[Coordinates],1) 
    xCellNodes = xgrid[CellNodes]
    ncells = num_sources(xCellNodes)
    nfaces = num_sources(xFaceNodes)
    xCellGeometries = xgrid[CellGeometries]

    # loop over all cells and all cell faces
    current_face = zeros(Int32,max_num_targets_per_source(xCellNodes)) # should be large enough to store largest nnodes_per_cellface
    faces = zeros(Int32,max_num_targets_per_source(xCellNodes)) # should be large enough to store largest nfaces_per_cell
    faces_per_cell = 0
    nodes_per_cellface = 0
    match = false
    cellEG = Triangle2D
    for cell = 1 : ncells
        cellEG = xCellGeometries[cell]
        faces_per_cell = nfaces_per_cell(cellEG)
        face_rule = face_enum_rule(cellEG)

        for k = 1 : faces_per_cell
            nodes_per_cellface = nnodes_per_cellface(cellEG, k)
            
            # get face nodes
            for j = 1 : nodes_per_cellface
                current_face[j] = xCellNodes[face_rule[k,j],cell]; 
            end

            # bubble_sort face nodes
            for j = nodes_per_cellface:-1:2
                for j2 = 1 : j-1
                    if current_face[j2] < current_face[j2+1]
                        swap = current_face[j2+1]
                        current_face[j2+1] = current_face[j2]
                        current_face[j2] = swap
                    end    
                end
            end

            # find facenr
            for face = 1 : nfaces
                match = true
                for k = 1 : nodes_per_cellface
                    if current_face[k] != xFaceNodes[k,face]
                        match = false
                        break;
                    end
                end        
                if match == true
                    faces[k] = face
                end
            end            
        end    
        append!(xCellFaces,faces[1:faces_per_cell])    
    end
    xCellFaces
end

# CellSigns = orientation signs for each face on each cell
function ExtendableGrids.instantiate(xgrid::ExtendableGrid, ::Type{CellSigns})

    # init CellSigns
    xCellSigns = VariableTargetAdjacency(Int32) # +1/-1 would be enough

    # get links to other stuff
    dim = size(xgrid[Coordinates],1) 
    xFaceNodes = xgrid[FaceNodes]
    xCellFaces = xgrid[CellFaces]
    xCellNodes = xgrid[CellNodes]
    ncells = num_sources(xCellNodes)
    nfaces = num_sources(xFaceNodes)
    xCellGeometries = xgrid[CellGeometries]

    # loop over all cells and all cell faces
    signs = zeros(Int32,max_num_targets_per_source(xCellFaces))
    faces_per_cell = 0
    temp = 0
    for cell = 1 : ncells
        faces_per_cell = nfaces_per_cell(xCellGeometries[cell])
        fill!(signs,-1)
        if xCellGeometries[cell] == Edge1D
            append!(xCellFaces,[-1,1])
        elseif xCellGeometries[cell] == Triangle2D || xCellGeometries[cell] == Quadrilateral2D
            for k = 1 : faces_per_cell
                if xCellNodes[k,cell] == xFaceNodes[1,xCellFaces[k,cell]]
                    signs[k] = 1
                end       
            end  
            append!(xCellSigns,signs[1:faces_per_cell])
        elseif xCellGeometries[cell] == Tetrahedron3D # no experience with 3D yet, might be wrong !!!
            for k = 1 : 4
                # find matching node in face nodes and look if the next one matches, too
                temp = 1
                for j = 1 : 3
                    if xCellNodes[k,cell] == xFaceNodes[j,xCellFaces[k,cell]]    
                        temp = j
                        break
                    end    
                end
                if xCellNodes[k,cell] == xFaceNodes[temp,xCellFaces[k,cell]] && xCellNodes[mod(k,4)+1,cell] == xFaceNodes[mod(temp,3)+1,xCellFaces[k,cell]]
                    signs[k] = 1
                end       
            end  
            append!(xCellSigns,signs[1:faces_per_cell])
        end
    end
    xCellSigns
end



# some methods to compute volume of different ElemTypes (beware: on submanifolds formulas get different)

function Volume4ElemType(Coords, Nodes, item, ::Type{Vertex0D}, ::Type{ExtendableGrids.AbstractCoordinateSystem})
    return 0.0
end

function Volume4ElemType(Coords, Nodes, item, ::Type{Edge1D}, ::Type{Cartesian1D})
    return abs(Coords[1, Nodes[2,item]] - Coords[1, Nodes[1,item]])
end

function Volume4ElemType(Coords, Nodes, item, ::Type{Edge1D}, ::Type{Cartesian2D})
    return sqrt((Coords[1, Nodes[2,item]] - Coords[1, Nodes[1,item]]).^2 + (Coords[2, Nodes[2,item]] - Coords[2, Nodes[1,item]]).^2)
end

function Volume4ElemType(Coords, Nodes, item, ::Type{Triangle2D}, ::Type{Cartesian2D})
    return 1 // 2 * ( Coords[1, Nodes[1, item]] * (Coords[2, Nodes[2,item]] -  Coords[2, Nodes[3, item]])
                  +   Coords[1, Nodes[2, item]] * (Coords[2, Nodes[3,item]] -  Coords[2, Nodes[1, item]])
                  +   Coords[1, Nodes[3, item]] * (Coords[2, Nodes[1,item]] -  Coords[2, Nodes[2, item]]) )
end

#function Volume4ElemType(Coords, Nodes, item, ::Type{Parallelogram2D}, ::Type{Cartesian2D})
#    return ( Coords[1, Nodes[1, item]] * (Coords[2, Nodes[2,item]] -  Coords[2, Nodes[3, item]])
#           + Coords[1, Nodes[2, item]] * (Coords[2, Nodes[3,item]] -  Coords[2, Nodes[1, item]])
#           + Coords[1, Nodes[3, item]] * (Coords[2, Nodes[1,item]] -  Coords[2, Nodes[2, item]]) )
#end

function Volume4ElemType(Coords, Nodes, item, ::Type{<:Quadrilateral2D}, ::Type{Cartesian2D})
    return 1//2 * (   (Coords[1, Nodes[1, item]] - Coords[1, Nodes[3, item]]) * (Coords[2, Nodes[2, item]] - Coords[2, Nodes[4, item]])
                    + (Coords[1, Nodes[4, item]] - Coords[1, Nodes[2, item]]) * (Coords[2, Nodes[1, item]] - Coords[2, Nodes[3, item]]) );
end
  


function ExtendableGrids.instantiate(xgrid::ExtendableGrid, ::Type{CellVolumes})

    # get links to other stuff
    xCoordinates = xgrid[Coordinates]
    xCellNodes = xgrid[CellNodes]
    ncells = num_sources(xCellNodes)
    xCellGeometries = xgrid[CellGeometries]
    xCoordinateSystem = xgrid[CoordinateSystem]

    # init CellVolumes
    xCellVolumes = zeros(Real,ncells)

    for cell = 1 : ncells
        xCellVolumes[cell] = Volume4ElemType(xCoordinates,xCellNodes,cell,xCellGeometries[cell],xCoordinateSystem)
    end

    xCellVolumes
end


function ExtendableGrids.instantiate(xgrid::ExtendableGrid, ::Type{FaceVolumes})

    # get links to other stuff
    xCoordinates = xgrid[Coordinates]
    xFaceNodes = xgrid[FaceNodes]
    nfaces = num_sources(xFaceNodes)
    xFaceGeometries = xgrid[FaceGeometries]
    xCoordinateSystem = xgrid[CoordinateSystem]

    # init FaceVolumes
    xFaceVolumes = zeros(Real,nfaces)

    for face = 1 : nfaces
        xFaceVolumes[face] = Volume4ElemType(xCoordinates,xFaceNodes,face,xFaceGeometries[face],xCoordinateSystem)
    end

    xFaceVolumes
end


function ExtendableGrids.instantiate(xgrid::ExtendableGrid, ::Type{BFaceVolumes})
    xgrid[FaceVolumes][xgrid[BFaces]]
end


function ExtendableGrids.instantiate(xgrid::ExtendableGrid, ::Type{BFaces})

    # get links to other stuff
    xCoordinates = xgrid[Coordinates]
    xFaceNodes = xgrid[FaceNodes]
    xBFaceNodes = xgrid[BFaceNodes]
    nbfaces = num_sources(xBFaceNodes)
    nfaces = num_sources(xFaceNodes)

    # init BFaces
    xBFaces = zeros(Int32,nbfaces)

    for bface = 1 : nbfaces
        # lazy sorting, can be improved later
        current_bface = sort(xBFaceNodes[:,bface]);
        current_bface = current_bface[end:-1:1]
        for face = 1 : nfaces
            if current_bface == xFaceNodes[:,face]
                xBFaces[bface] = face
                break
            end
        end
    end

    xBFaces
end


function ExtendableGrids.instantiate(xgrid::ExtendableGrid, ::Type{FaceCells})
    # get links to other stuff
    xCoordinates = xgrid[Coordinates]
    xCellFaces = xgrid[CellFaces]
    xFaceNodes = xgrid[FaceNodes]
    ncells = num_sources(xCellFaces)
    nfaces = num_sources(xFaceNodes)

    # init BFaces
    xFaceCells = zeros(Int32,2,nfaces)

    nfaces4cell = 0
    facenr = 0
    for cell = 1 : ncells
        nfaces4cell = num_targets(xCellFaces,cell)
        for f = 1 : nfaces4cell
            face = xCellFaces[f,cell]
            if xFaceCells[1,face] == 0
                xFaceCells[1,face] = cell
            else
                xFaceCells[2,face] = cell
            end    
        end
    end

    return xFaceCells
end

# This assigns Regions to faces by looking at neighbouring cells
# don't know yet if this is a good idea

function ExtendableGrids.instantiate(xgrid::ExtendableGrid, ::Type{FaceRegions})
    return VectorOfConstants(Int32(0),num_sources(xgrid[FaceNodes]))
end


function ExtendableGrids.instantiate(xgrid::ExtendableGrid, ::Type{BFaceCellPos})

    # get links to other stuff
    xCoordinates = xgrid[Coordinates]
    xCellFaces = xgrid[CellFaces]
    xFaceCells = xgrid[FaceCells]
    xBFaces = xgrid[BFaces]
    nbfaces = length(xBFaces)

    # init BFaces
    xBFaceCellPos = zeros(Int32,nbfaces)

    cface = 0
    cell = 0
    nfaces4cell = 0
    for bface = 1 : nbfaces
        cface = xBFaces[bface]
        cell = xFaceCells[1,cface]
        nfaces4cell = num_targets(xCellFaces,cell)
        for face = 1 : nfaces4cell
            if cface == xCellFaces[face,cell]
                xBFaceCellPos[bface] = face
                break
            end
        end
    end

    xBFaceCellPos
end



function Normal4ElemType!(normal, Coords, Nodes, item, ::Type{Vertex0D}, ::Type{Cartesian1D})
    # rotate tangent
    normal[1] = 1.0
end

function Normal4ElemType!(normal, Coords, Nodes, item, ::Type{Edge1D}, ::Type{Cartesian2D})
    # rotate tangent
    normal[1] = Coords[2, Nodes[2,item]] - Coords[2,Nodes[1,item]]
    normal[2] = Coords[1,Nodes[1,item]] - Coords[1, Nodes[2,item]]
    # divide by length
    normal ./= sqrt(normal[1]^2+normal[2]^2)
end

function ExtendableGrids.instantiate(xgrid::ExtendableGrid, ::Type{FaceNormals})

    # get links to other stuff
    dim = size(xgrid[Coordinates],1) 
    xCoordinates = xgrid[Coordinates]
    xFaceNodes = xgrid[FaceNodes]
    nfaces = num_sources(xFaceNodes)
    xFaceGeometries = xgrid[FaceGeometries]
    xCoordinateSystem = xgrid[CoordinateSystem]

    # init FaceNormals
    xFaceNormals = zeros(Real,dim,nfaces)
    normal = zeros(Real,dim)
    for face = 1 : nfaces
        Normal4ElemType!(normal,xCoordinates,xFaceNodes,face,xFaceGeometries[face],xCoordinateSystem)
        for k = 1 : dim
            xFaceNormals[k, face] = normal[k]
        end    
    end

    xFaceNormals
end

end # module
