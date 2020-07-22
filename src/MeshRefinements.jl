
# functions that tell how to split one ElementGeometry into another
split_rule(::Type{Triangle2D}, ::Type{Triangle2D}) = reshape([1,2,3],1,3)
split_rule(::Type{<:Quadrilateral2D}, ::Type{Triangle2D}) = [1 2 3;1 3 4]
split_rule(::Type{Edge1D}, ::Type{Triangle2D}) = reshape([1,2,2],1,3)

# function that generates a new simplexgrid from a mixed grid
function split_grid_into(source_grid::ExtendableGrid{T,K}, targetgeometry::Type{Triangle2D}) where {T,K}
    xgrid=ExtendableGrid{T,K}()
    xgrid[Coordinates]=source_grid[Coordinates]
    oldCellTypes = source_grid[CellGeometries]
    EG = Base.unique(oldCellTypes)
    
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
    xgrid[CellGeometries] = VectorOfConstants(Triangle2D,ncells)
    xgrid[CellRegions]=ones(Int32,ncells)
    xgrid[BFaceNodes]=source_grid[BFaceNodes]
    xgrid[BFaceRegions]=source_grid[BFaceRegions]
    xgrid[BFaceGeometries]=VectorOfConstants(Edge1D,num_sources(source_grid[BFaceNodes]))
    xgrid[CoordinateSystem]=source_grid[CoordinateSystem]
    return xgrid
end

# uniform refinement rules
# first k nodes are the CellNodes
# next m nodes are the CellFaces (midpoints)
# next node is the CellMidpoint (if needed)

uniform_refine_rule(::Type{<:Edge1D}) = [1 3; 3 2]
uniform_refine_rule(::Type{<:Triangle2D}) = [1 4 6; 2 5 4; 3 6 5; 4 5 6]
uniform_refine_rule(::Type{<:Quadrilateral2D}) = [1 5 9 8; 2 6 9 5; 3 7 9 6; 4 8 9 7]

uniform_refine_needcellmidpoints(::Type{<:AbstractElementGeometry}) = false
uniform_refine_needcellmidpoints(::Type{<:Edge1D}) = true
uniform_refine_needcellmidpoints(::Type{<:Quadrilateral2D}) = true


# barycentric refinement rules
# first k nodes are the CellNodes, k+1-th  node is cell midpoint
barycentric_refine_rule(::Type{<:Triangle2D}) = [1 2 4; 2 3 4; 3 1 4]


# function that generates a new simplexgrid from a mixed grid
function uniform_refine(source_grid::ExtendableGrid{T,K}) where {T,K}
    xgrid = ExtendableGrid{T,K}()
    oldCoordinates = source_grid[Coordinates]
    oldCellTypes = source_grid[CellGeometries]
    EG = Base.unique(oldCellTypes)

    # get dimension of CellGeometries
    # currently it is assumed to be the same for all cells
    dim = dim_element(EG[1]) 
    
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
    newvertices = 0 # in 1D no additional vertices on the faces are needed
    if dim == 2 # in 2D each face id halved
        newvertices = nfaces
    end
    oldvertices = size(oldCoordinates,2)
    newnode = oldvertices + newvertices
    # additionally cell midpoints are needed for some refinements
    for cell = 1 : num_sources(oldCellNodes)
        cellEG = oldCellTypes[cell]
        if uniform_refine_needcellmidpoints(cellEG) == true
            newvertices += 1
        end    
    end
    xCoordinates = zeros(Float64,size(oldCoordinates,1),oldvertices+newvertices)
    @views xCoordinates[:,1:oldvertices] = oldCoordinates

    
    newvertex = zeros(Float64,size(xCoordinates,1))
    if dim == 2 # add face midpoints to Coordinates
        nnodes4face = 0
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
    end
    
    # determine new cells
    nnodes4cell = 0
    ncells = 0
    iEG = 1
    subcellnodes = zeros(Int32,max_num_targets_per_source(oldCellNodes)+max_num_targets_per_source(oldCellFaces)+1)
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
    
    if dim == 1
        xgrid[BFaceNodes] = oldBFaceNodes
        xgrid[BFaceRegions] = oldBFaceRegions
        xgrid[BFaceGeometries] = source_grid[BFaceGeometries]
    elseif dim == 2 # refine boundary faces
        # todo: use uniform_refine for Edge1D here
        for bface = 1 : nbfaces
            face = oldBFaces[bface]
            xBFaceNodes[:,2*bface-1] = [oldBFaceNodes[1,bface], oldvertices+face]
            xBFaceNodes[:,2*bface] = [oldvertices+face, oldBFaceNodes[2,bface]]
            xBFaceRegions[2*bface-1] = oldBFaceRegions[bface]
            xBFaceRegions[2*bface] = oldBFaceRegions[bface]
        end
        xgrid[BFaceNodes] = xBFaceNodes
        xgrid[BFaceRegions] = xBFaceRegions
        xgrid[BFaceGeometries] = VectorOfConstants(Edge1D,2*nbfaces)
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
    xgrid[CoordinateSystem]=source_grid[CoordinateSystem]
    return xgrid
end

# function that generates a new barycentrically refined simplexgrid from a mixed grid
# (first grid is plit into triangles)
function barycentric_refine(source_grid::ExtendableGrid{T,K}) where {T,K}
    # split first into triangles
    source_grid = split_grid_into(source_grid,Triangle2D)

    xgrid = ExtendableGrid{T,K}()
    oldCoordinates = source_grid[Coordinates]
    oldCellTypes = source_grid[CellGeometries]
    EG = Base.unique(oldCellTypes)
    
    refine_rules = Array{Array{Int,2},1}(undef,length(EG))
    for j = 1 : length(EG)
        refine_rules[j] = barycentric_refine_rule(EG[j])
    end
    xCellNodes = VariableTargetAdjacency(Int32)
    xCellGeometries = []

    oldCellNodes = source_grid[CellNodes]
    oldCellFaces = source_grid[CellFaces]

    # determine number of new vertices
    cellEG = Triangle2D
    newvertices = 0
    for cell = 1 : num_sources(oldCellNodes)
        newvertices += 1
    end
    oldvertices = size(oldCoordinates,2)
    xCoordinates = zeros(Float64,size(oldCoordinates,1),oldvertices+newvertices)
    @views xCoordinates[:,1:oldvertices] = oldCoordinates

    # determine new cells
    nnodes4cell = 0
    ncells = 0
    iEG = 1
    subcellnodes = zeros(Int32,max_num_targets_per_source(oldCellNodes)+max_num_targets_per_source(oldCellFaces)+1)
    newnode = oldvertices
    m = 0
    newvertex = zeros(Float64,size(xCoordinates,1))
    for cell = 1 : num_sources(oldCellNodes)
        nnodes4cell = num_targets(oldCellNodes,cell)
        nfaces4cell = num_targets(oldCellFaces,cell)
        cellEG = oldCellTypes[cell]
        iEG = findfirst(isequal(cellEG), EG)
        
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

        for j = 1 : size(refine_rules[iEG],1)
            for k = 1 : size(refine_rules[iEG],2)
                m = refine_rules[iEG][j,k]
                if m <= nnodes4cell 
                    subcellnodes[k] = oldCellNodes[m,cell]
                else
                    subcellnodes[k] = newnode
                end        
            end    
            append!(xCellNodes,subcellnodes[1:size(refine_rules[iEG],2)])
            push!(xCellGeometries,cellEG)
        end    
        ncells += size(refine_rules[iEG],1)
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
    xgrid[BFaceNodes]=source_grid[BFaceNodes]
    xgrid[BFaceRegions]=source_grid[BFaceRegions]
    xgrid[BFaceGeometries]=source_grid[BFaceGeometries]
    xgrid[CoordinateSystem]=source_grid[CoordinateSystem]
    Base.show()
    return xgrid
end