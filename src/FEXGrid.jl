module FEXGrid

using ExtendableGrids

export FaceNodes, FaceGeometries, FaceVolumes, FaceRegions, FaceCells, FaceNormals
export CellFaces, CellSigns, CellVolumes
export BFaces, BFaceCellPos, BFaceVolumes
export nfaces_per_cell, facetype_of_cellface

include("FEXGrid_AssemblyJunctions.jl");
export AbstractAssemblyType
export AbstractAssemblyTypeCELL, AbstractAssemblyTypeFACE, AbstractAssemblyTypeBFACE, AbstractAssemblyTypeBFACECELL
export GridComponentNodes4AssemblyType
export GridComponentVolumes4AssemblyType
export GridComponentGeometries4AssemblyType
export GridComponentRegions4AssemblyType

include("FEXGrid_L2GTransformer.jl");
export L2GTransformer, update!, eval!, mapderiv!, piola!


export uniqueEG,split_grid_into, uniform_refine


# additional ElementGeometryTypes with parent information
abstract type Edge1DWithParent{Parent <: AbstractElementGeometry} <: Edge1D end
abstract type Edge1DWithParents{Parent1 <: AbstractElementGeometry, Parent2 <: AbstractElementGeometry} <: Edge1D end
export Edge1DWithParent, Edge1DWithParents

function AddParent(FEG::Type{<:Edge1D}, CEG::Type{<:AbstractElementGeometry})
    return Edge1DWithParent{CEG}
end


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


# unique functions that only selects uniques in specified regions
function uniqueEG(xItemGeometries, xItemRegions, xItemDofs, regions)
    nitems = 0
    try
        nitems = num_sources(xItemGeometries)
    catch
        nitems = length(xItemGeometries)
    end      
    EG::Array{DataType,1} = []
    ndofs4EG = Array{Array{Int,1},1}(undef,length(xItemDofs))
    for e = 1 : length(xItemDofs)
        ndofs4EG[e] = []
    end
    iEG = 0
    cellEG = Triangle2D
    for item = 1 : nitems
        for j = 1 : length(regions)
            if xItemRegions[item] == regions[j]
                cellEG = xItemGeometries[item]
                iEG = 0
                for k = 1 : length(EG)
                    if cellEG == EG[k]
                        iEG = k
                        break;
                    end
                end
                if iEG == 0
                    append!(EG, [xItemGeometries[item]])
                    for e = 1 : length(xItemDofs)
                        append!(ndofs4EG[e], num_targets(xItemDofs[e],item))
                    end
                end  
                break; # rest of for loop can be skipped
            end    
        end
    end    
    return EG, ndofs4EG
end


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
facetype_of_cellface(::Type{<:Triangle2D}, k) = Edge1DWithParent{Triangle2D}
facetype_of_cellface(::Type{<:Quadrilateral2D}, k) = Edge1DWithParent{Quadrilateral2D}
facetype_of_cellface(::Type{<:Tetrahedron3D}, k) = Triangle2D

facetype_of_cellface(P1::Type{<:AbstractElementGeometry2D},P2::Type{<:AbstractElementGeometry2D}, k) = Edge1DWithParents{P1,P2}


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
    cellEG = Triangle2D
    iEG = 1
    for cell = 1 : num_sources(oldCellNodes)
        nnodes4cell = num_targets(oldCellNodes,cell)
        cellEG = oldCellTypes[cell]
        iEG = findfirst(isequal(cellEG), EG)
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

# uniform uniform_refine
# first k nodes are the CellNodes
# next m nodes are the CellFaces (midpoints)
# next node is the CellMidpoint (if needed)

uniform_refine_rule(::Type{<:Triangle2D}) = [1 4 6; 2 5 4; 3 6 5; 4 5 6]
uniform_refine_rule(::Type{<:Quadrilateral2D}) = [1 5 9 8; 2 6 9 5; 3 7 9 6; 4 8 9 7]

uniform_refine_needcellmidpoints(::Type{<:AbstractElementGeometry}) = false
uniform_refine_needcellmidpoints(::Type{<:Quadrilateral2D}) = true

# function that generates a new simplexgrid from a mixed grid
function uniform_refine(source_grid::ExtendableGrid{T,K}) where {T,K}
    xgrid = ExtendableGrid{T,K}()
    oldCoordinates = source_grid[Coordinates]
    oldCellTypes = source_grid[CellGeometries]
    EG = unique(oldCellTypes)
    
    refine_rules = Array{Array{Int,2},1}(undef,length(EG))
    for j = 1 : length(EG)
        refine_rules[j] = uniform_refine_rule(EG[j])
    end
    xCellNodes = VariableTargetAdjacency(Int32)
    xCellGeometries = []
    oldCellNodes = source_grid[CellNodes]
    oldFaceNodes = source_grid[FaceNodes]
    oldCellFaces = source_grid[CellFaces]
    nfaces = num_sources(oldFaceNodes)

    # determine number of new vertices
    cellEG = Triangle2D
    newvertices = nfaces
    for cell = 1 : num_sources(oldCellNodes)
        cellEG = oldCellTypes[cell]
        if uniform_refine_needcellmidpoints(cellEG) == true
            newvertices += 1
        end    
    end
    oldvertices = size(oldCoordinates,2)
    xCoordinates = zeros(Float64,size(oldCoordinates,1),oldvertices+newvertices)
    @views xCoordinates[:,1:oldvertices] = oldCoordinates

    # add face midpoints to Coordinates
    nnodes4face = 0
    newvertex = zeros(Float64,size(xCoordinates,1))
    for face = 1 : nfaces
        nnodes4face = num_targets(oldFaceNodes,face)
        fill!(newvertex,0.0)
        for k = 1 : nnodes4face, d = 1 : size(xCoordinates,1)
            newvertex[d] += xCoordinates[d,oldFaceNodes[k,face]] 
        end    
        newvertex ./= nnodes4face
        for d = 1 : size(xCoordinates,1)
            xCoordinates[d,oldvertices+face] = newvertex[d]
        end
    end    
    
    # determine new cells
    nnodes4cell = 0
    ncells = 0
    iEG = 1
    subcellnodes = zeros(Int32,max_num_targets_per_source(oldCellNodes)+max_num_targets_per_source(oldCellFaces)+1)
    newnode = oldvertices + nfaces
    m = 0
    for cell = 1 : num_sources(oldCellNodes)
        nnodes4cell = num_targets(oldCellNodes,cell)
        nfaces4cell = num_targets(oldCellFaces,cell)
        cellEG = oldCellTypes[cell]
        iEG = findfirst(isequal(cellEG), EG)
        if uniform_refine_needcellmidpoints(cellEG) == true
            # add cell midpoint to Coordinates
            newnode += 1
            fill!(newvertex,0.0)
            for k = 1 : nnodes4cell, d = 1 : size(xCoordinates,1)
                newvertex[d] += xCoordinates[d,oldCellNodes[k,cell]] 
            end    
            newvertex ./= nnodes4cell
            for d = 1 : size(xCoordinates,1)
                xCoordinates[d,newnode] = newvertex[d]
            end
        end
        for j = 1 : size(refine_rules[iEG],1)
            for k = 1 : size(refine_rules[iEG],2)
                m = refine_rules[iEG][j,k]
                if m <= nnodes4cell 
                    subcellnodes[k] = oldCellNodes[m,cell]
                elseif m <= nnodes4cell + nfaces4cell
                    subcellnodes[k] = oldvertices + oldCellFaces[m-nnodes4cell,cell]
                else
                    subcellnodes[k] = newnode
                end        
            end    
            append!(xCellNodes,subcellnodes[1:size(refine_rules[iEG],2)])
            push!(xCellGeometries,cellEG)
        end    
        ncells += size(refine_rules[iEG],1)
    end

    # update BFaces (will only work for 2D faces atm)
    oldBFaceNodes = source_grid[BFaceNodes]
    oldBFaces = source_grid[BFaces]
    oldBFaceRegions = source_grid[BFaceRegions]
    xBFaceNodes = zeros(Int32,size(oldBFaceNodes,1),2*size(oldBFaceNodes,2))
    xBFaceRegions = zeros(Int32,2*size(oldBFaceNodes,2))
    nbfaces = num_sources(oldBFaceNodes)
    for bface = 1 : nbfaces
        face = oldBFaces[bface]
        xBFaceNodes[:,2*bface-1] = [oldBFaceNodes[1,bface], oldvertices+face]
        xBFaceNodes[:,2*bface] = [oldvertices+face, oldBFaceNodes[2,bface]]
        xBFaceRegions[2*bface-1] = oldBFaceRegions[bface]
        xBFaceRegions[2*bface] = oldBFaceRegions[bface]
    end    

    xgrid[Coordinates] = xCoordinates
    if typeof(oldCellNodes) == Array{Int32,2}
        nnodes4cell = size(oldCellNodes,1)
        xgrid[CellNodes] = reshape(xCellNodes.colentries,nnodes4cell,num_sources(xCellNodes))
    else
        xgrid[CellNodes] = xCellNodes
    end
    xgrid[CellRegions]=VectorOfConstants{Int32}(1,ncells)
    xgrid[CellGeometries] = Array{DataType,1}(xCellGeometries)
    xgrid[BFaceNodes]=xBFaceNodes
    xgrid[BFaceRegions]=xBFaceRegions
    xgrid[BFaceGeometries]=VectorOfConstants(Edge1D,2*nbfaces)
    xgrid[CoordinateSystem]=source_grid[CoordinateSystem]
    Base.show()
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
    nnodes = num_sources(xgrid[Coordinates])
    xCellGeometries = xgrid[CellGeometries]

    #transpose CellNodes to get NodeCells
    xNodeCells = atranspose(xCellNodes)
    max_ncell4node = max_num_targets_per_source(xNodeCells)

    xFaceNodes = VariableTargetAdjacency(Int32)
    xCellFaces = VariableTargetAdjacency(Int32)
    xFaceCells = zeros(Int32,0) # cells are appended and at the end rewritten into 2,nfaces array
    xCellSigns = VariableTargetAdjacency(Int32)
    xFaceGeometries::Array{DataType,1} = []
    xBFaces::Array{Int32,1} = []

    current_face = zeros(Int32,max_num_targets_per_source(xCellNodes)) # should be large enough to store largest nnodes_per_cellface
    flag4face = zeros(Bool,nnodes)
    cellEG = Triangle2D
    cell2EG = Triangle2D
    node = 0
    node_cells = zeros(Int32,max_ncell4node) # should be large enough to store largest nnodes_per_cellface
    face = 0
    cell2 = 0
    nneighbours = 0
    faces_per_cell = 0
    faces_per_cell2 = 0
    nodes_per_cellface = 0
    nodes_per_cellface2 = 0
    common_nodes = 0
    # pre-allocate xCellFaces
    for cell = 1 : ncells
        cellEG = xCellGeometries[cell]
        append!(xCellFaces,zeros(Int32,nfaces_per_cell(cellEG)))
        append!(xCellSigns,zeros(Int32,nfaces_per_cell(cellEG)))
    end   

    # loop over cells
    for cell = 1 : ncells
        cellEG = xCellGeometries[cell]
        faces_per_cell = nfaces_per_cell(cellEG)
        face_rule = face_enum_rule(cellEG)

        # loop over cell faces
        for k = 1 : faces_per_cell

            # check if face is already known to cell
            if xCellFaces[k,cell] > 0
                continue;
            end    

            nodes_per_cellface = nnodes_per_cellface(cellEG, k)

            # flag face nodes and commons4cells
            for j = 1 : nodes_per_cellface
                node = xCellNodes[face_rule[k,j],cell]
                current_face[j] = node
                flag4face[node] = true; 
            end

            # get first node and its neighbours
            node = xCellNodes[face_rule[k,1],cell]
            nneighbours = num_targets(xNodeCells,node)
            node_cells[1:nneighbours] = xNodeCells[:,node]

            # loop over neighbours
            no_neighbours_found = true
            for n = 1 : nneighbours
                cell2 = node_cells[n]

                # skip if cell2 is the same as cell
                if (cell == cell2) 
                    continue; 
                end

                # loop over faces of cell2
                cell2EG = xCellGeometries[cell2]
                faces_per_cell2 = nfaces_per_cell(cell2EG)
                face_rule2 = face_enum_rule(cell2EG)
                for f2 = 1 : faces_per_cell2
                    # check if face is already known to cell2
                    if xCellFaces[f2,cell2] > 0
                        continue;
                    end    

                    #otherwise compare nodes of face and face2
                    nodes_per_cellface2 = nnodes_per_cellface(cell2EG, f2)
                    common_nodes = 0
                    if nodes_per_cellface == nodes_per_cellface2
                        for j = 1 : nodes_per_cellface2
                            if flag4face[xCellNodes[face_rule2[f2,j],cell2]]
                                common_nodes += 1
                            else
                                continue;    
                            end    
                        end
                    end

                    # if all nodes are the same, register face
                    if (common_nodes == nodes_per_cellface2)
                        no_neighbours_found = false
                        face += 1
                        # set index for adjacencies missing
                        #xCellFaces[k,cell] = face
                        #xCellFaces[f2,cell2] = face
                        push!(xFaceCells,cell)
                        push!(xFaceCells,cell2)
                        xCellFaces.colentries[xCellFaces.colstart[cell]+k-1] = face
                        xCellFaces.colentries[xCellFaces.colstart[cell2]+f2-1] = face
                        xCellSigns.colentries[xCellSigns.colstart[cell]+k-1] = 1
                        xCellSigns.colentries[xCellSigns.colstart[cell2]+f2-1] = -1
                        append!(xFaceNodes,current_face[1:nodes_per_cellface])
                        push!(xFaceGeometries,facetype_of_cellface(cellEG,cell2EG,k))
                        break;
                    end

                end
            end

            # if no common neighbour cell is found, register face (boundary faces)
            if no_neighbours_found == true
                face += 1
                # set index for adjacencies missing
                #xCellFaces[k,cell] = face
                push!(xFaceCells,cell)
                push!(xFaceCells,0)
                xCellFaces.colentries[xCellFaces.colstart[cell]+k-1] = face
                xCellSigns.colentries[xCellSigns.colstart[cell]+k-1] = 1
                append!(xFaceNodes,current_face[1:nodes_per_cellface])
                push!(xFaceGeometries,facetype_of_cellface(cellEG,k))
            end

            #reset flag4face
            for j = 1 : nnodes
                flag4face[j] = 0
            end    
        end    
    end
    xgrid[FaceGeometries] = xFaceGeometries
    xgrid[CellFaces] = xCellFaces
    xgrid[CellSigns] = xCellSigns
    xgrid[FaceCells] = reshape(xFaceCells,2,face)
    xFaceNodes
end

# CellFaces = faces for each cell
function ExtendableGrids.instantiate(xgrid::ExtendableGrid, ::Type{CellFaces})
    ExtendableGrids.instantiate(xgrid, FaceNodes)
    xgrid[CellFaces]
end

# CellSigns = orientation signs for each face on each cell
function ExtendableGrids.instantiate(xgrid::ExtendableGrid, ::Type{CellSigns})
    ExtendableGrids.instantiate(xgrid, FaceNodes)
    xgrid[CellSigns]
end



# some methods to compute volume of different ElemTypes (beware: on submanifolds formulas get different)

function Volume4ElemType(Coords, Nodes, item, ::Type{<:Vertex0D}, ::Type{ExtendableGrids.AbstractCoordinateSystem})
    return 0.0
end

function Volume4ElemType(Coords, Nodes, item, ::Type{<:Edge1D}, ::Type{Cartesian1D})
    return abs(Coords[1, Nodes[2,item]] - Coords[1, Nodes[1,item]])
end

function Volume4ElemType(Coords, Nodes, item, ::Type{<:Edge1D}, ::Type{Cartesian2D})
    return sqrt((Coords[1, Nodes[2,item]] - Coords[1, Nodes[1,item]]).^2 + (Coords[2, Nodes[2,item]] - Coords[2, Nodes[1,item]]).^2)
end

function Volume4ElemType(Coords, Nodes, item, ::Type{<:Triangle2D}, ::Type{Cartesian2D})
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
    xFaceCells = xgrid[FaceCells]
    xCellGeometries = xgrid[CellGeometries]
    nbfaces = num_sources(xBFaceNodes)
    nfaces = num_sources(xFaceNodes)

    # init BFaces
    xBFaces = zeros(Int32,nbfaces)
    xBFaceGeometries = xgrid[BFaceGeometries]
    if typeof(xBFaceGeometries) == VectorOfConstants{DataType}
        EG = xBFaceGeometries[1]
        xBFaceGeometries = Array{DataType,1}(undef,nbfaces)
        for j = 1 : nbfaces
            xBFaceGeometries[j] = EG
        end
    end

    current_bface = zeros(Int32,max_num_targets_per_source(xFaceNodes))
    nodes_per_bface::Int32 = 0
    swap::Int32 = 0
    match::Bool = false
    for bface = 1 : nbfaces
        nodes_per_bface = num_targets(xBFaceNodes,bface)
        for j = 1 : nodes_per_bface
            current_bface[j] = xBFaceNodes[j,bface]
        end    
        
        # find matching face
        for face = 1 : nfaces
            match = true
            for k = 1 : nodes_per_bface
                if current_bface[k] != xFaceNodes[k,face]
                    match = false
                    break
                end
            end        
            if match == true
                xBFaces[bface] = face
                xBFaceGeometries[bface] = AddParent(xBFaceGeometries[bface],xCellGeometries[xFaceCells[1,face]])
                break
            end
            if face == nfaces
                println("! WARNING ! did not find maching face for bface $bface")
                println("(Maybe BFaceNodes are not in anti-clockwise order?)")
            end
        end
    end

    xgrid[BFaceGeometries] = xBFaceGeometries
    xBFaces
end


function ExtendableGrids.instantiate(xgrid::ExtendableGrid, ::Type{FaceCells})
    ExtendableGrids.instantiate(xgrid, FaceNodes)
    xgrid[FaceCells]
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



function Normal4ElemType!(normal, Coords, Nodes, item, ::Type{<:Vertex0D}, ::Type{Cartesian1D})
    # rotate tangent
    normal[1] = 1.0
end

function Normal4ElemType!(normal, Coords, Nodes, item, ::Type{<:Edge1D}, ::Type{Cartesian2D})
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
