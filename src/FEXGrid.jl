module FEXGrid

export FaceNodes
export CellFaces

using XGrid

abstract type FaceNodes <: AbstractGridAdjacency end
abstract type CellFaces <: AbstractGridAdjacency end


# show function for Grid
function show(xgrid::ExtendableGrid)

    dim = size(xgrid[Coordinates],1)
    nnodes = nsources(xgrid[Coordinates])
    ncells = nsources(xgrid[CellNodes])
    
	println("XGrid");
    println("======");
	println("dim: $(dim)")
	println("nnodes: $(nnodes)")
    println("ncells: $(ncells)")
    if haskey(xgrid.components,FaceNodes)
        nfaces = nsources(xgrid[FaceNodes])
        println("nfaces: $(nfaces)")
    else
        println("nfaces: (FaceNodes not instantiated)")
    end
    println("")
    println("Components");
    println("===========");
    for tuple in xgrid.components
        println("> $(tuple[1])")
    end
end


function XGrid.instantiate(xgrid::ExtendableGrid, ::Type{FaceNodes})

    # each edge consists of dim nodes (beware: has to be replaced later if triangulation of submanifolds are included)
    dim = size(xgrid[Coordinates],1) 
    xCellNodes = xgrid[CellNodes]
    ncells = nsources(xCellNodes)
    xCellTypes = xgrid[CellTypes]

    # helper field to store all face nodes (including reversed duplicates)
    nodes4allfaces = zeros(Int32,0)
    nfaces = 0
    swap = 0
    current_face = zeros(Int32,dim)
    for cell = 1 : ncells
        if xCellTypes[cell] == Simplex1D
            append!(nodes4allfaces,xCellNodes[1,cell])
            nfaces += 1
        elseif xCellTypes[cell] == Simplex2D
            for k = 1 : 3
                current_face[1] = xCellNodes[k,cell]
                current_face[2] = xCellNodes[mod(k,3)+1,cell];

                # sort face numbers
                if current_face[1] > current_face[2]
                    swap = current_face[1]
                    current_face[1] = current_face[2]
                    current_face[2] = swap
                end

                append!(nodes4allfaces,current_face)
                nfaces += 1
            end  
        elseif xCellTypes[cell] == Simplex3D
            for k = 1 : 4
                current_face[1] = xCellNodes[k,cell]
                current_face[2] = xCellNodes[mod(k,4)+1,cell];
                current_face[3] = xCellNodes[mod(k,4)+2,cell];

                # sort face numbers (beware: bad idea for quads!!!)
                if current_face[1] > current_face[2]
                    swap = current_face[1]
                    current_face[1] = current_face[2]
                    current_face[2] = swap
                end
                if current_face[2] > current_face[3]
                    swap = current_face[2]
                    current_face[2] = current_face[3]
                    current_face[3] = swap
                end
                if current_face[1] > current_face[2]
                    swap = current_face[1]
                    current_face[1] = current_face[2]
                    current_face[2] = swap
                end

                append!(nodes4allfaces,current_face)
                nfaces += 1
            end  
        end
    end    

    # remove duplicates and assign as fixed adjacency
    nodes4allfaces = reshape(nodes4allfaces,(dim,nfaces))
    nodes4allfaces = unique(nodes4allfaces, dims = 2);
    nodes4allfaces
end

function XGrid.instantiate(xgrid::ExtendableGrid, ::Type{CellFaces})

    # todo
    # idea : use A = ExtendableSparseMatrix{Int64,Int64} and fill it with
    #        A(n1,..,nk) = j
    # for each face from FaceNodes[:,j] = (n1,..,nk) and all its permutations
    #
    # then go through cell faces and check via the SparseMatrix which number it has

    # use a transpose to get adjacency nodes2face
    Nodes2Faces = atranspose(xgrid[FaceNodes])

    # init CellFaces
    xCellFaces = VariableTargetAdjacency(Int32)

    # get links to other stuff
    dim = size(xgrid[Coordinates],1) 
    xCellNodes = xgrid[CellNodes]
    ncells = nsources(xCellNodes)
    xCellTypes = xgrid[CellTypes]

    # loop over all cells and all cell faces
    current_face = zeros(Int64,dim)
    faces = zeros(Int64,ntargets(xCellNodes))
    for cell = 1 : ncells
        if xCellTypes[cell] == Simplex1D
            append!(xCellFaces,xCellNodes[:,cell])
        elseif xCellTypes[cell] == Simplex2D
            for k = 1 : 3
                current_face[1] = xCellNodes[k,cell]
                current_face[2] = xCellNodes[mod(k,3)+1,cell];

                # sort face numbers
                if current_face[1] > current_face[2]
                    swap = current_face[1]
                    current_face[1] = current_face[2]
                    current_face[2] = swap
                end

                # find facenr
                faces[k] = Nodes2Faces[current_face[1],current_face[2]]
            end  
            append!(xCellFaces,faces[1:3])
        elseif xCellTypes[cell] == Simplex3D
            for k = 1 : 4
                current_face[1] = xCellNodes[k,cell]
                current_face[2] = xCellNodes[mod(k,4)+1,cell];
                current_face[3] = xCellNodes[mod(k,4)+2,cell];

                # sort face numbers (beware: bad idea for quads!!!)
                if current_face[1] > current_face[2]
                    swap = current_face[1]
                    current_face[1] = current_face[2]
                    current_face[2] = swap
                end
                if current_face[2] > current_face[3]
                    swap = current_face[2]
                    current_face[2] = current_face[3]
                    current_face[3] = swap
                end
                if current_face[1] > current_face[2]
                    swap = current_face[1]
                    current_face[1] = current_face[2]
                    current_face[2] = swap
                end

                # find facenr
                faces[k] = Nodes2Faces[current_face[1],current_face[2],current_face[3]]
            end  
            append!(xCellFaces,faces[1:4])
        end
    end
    xCellFaces
end


# function ensure_length4faces!(Grid::Mesh)
#     ensure_nodes4faces!(Grid)
#     celldim = size(Grid.nodes4faces,2) - 1;
#     nfaces::Int = size(Grid.nodes4faces,1);
#     if size(Grid.length4faces,1) != size(nfaces,1)
#         if celldim == 1 # also allow d-dimensional points on a line!
#             Grid.length4faces= zeros(eltype(Grid.coords4nodes),nfaces);
#             xdim::Int = size(Grid.coords4nodes,2)
#             for face = 1 : nfaces
#                 for d = 1 : xdim
#                     Grid.length4faces[face] += (Grid.coords4nodes[Grid.nodes4faces[face,2],d] - Grid.coords4nodes[Grid.nodes4faces[face,1],d]).^2
#                 end
#                  Grid.length4faces[face] = sqrt(Grid.length4faces[face]);    
#             end   
#         elseif celldim == 0 # points of length 1
#             Grid.length4faces = ones(eltype(Grid.coords4nodes),nfaces);
#         end
#     end        
# end  

# function ensure_volume4cells!(Grid::Mesh)
#     ncells::Int = size(Grid.nodes4cells,1);
#     if size(Grid.volume4cells,1) != size(ncells,1)
#         Grid.volume4cells = zeros(eltype(Grid.coords4nodes),ncells);
#         if typeof(Grid.elemtypes[1]) <: ElemType1DInterval
#             for cell = 1 : ncells
#                 Grid.volume4cells[cell] = abs(Grid.coords4nodes[Grid.nodes4cells[cell,2],1] - Grid.coords4nodes[Grid.nodes4cells[cell,1],1])
#             end    
#         elseif typeof(Grid.elemtypes[1]) <: Abstract1DElemType
#             xdim::Int = size(Grid.coords4nodes,2)
#             for cell = 1 : ncells
#                 for d = 1 : xdim
#                     Grid.volume4cells[cell] += (Grid.coords4nodes[Grid.nodes4cells[cell,2],d] - Grid.coords4nodes[Grid.nodes4cells[cell,1],d]).^2
#                 end
#                 Grid.volume4cells[cell] = sqrt(Grid.volume4cells[cell]);    
#             end    
#         elseif typeof(Grid.elemtypes[1]) <: ElemType2DTriangle
#             for cell = 1 : ncells 
#                 Grid.volume4cells[cell] = 1 // 2 * (
#                Grid.coords4nodes[Grid.nodes4cells[cell,1],1] * (Grid.coords4nodes[Grid.nodes4cells[cell,2],2] -  Grid.coords4nodes[Grid.nodes4cells[cell,3],2])
#             + Grid.coords4nodes[Grid.nodes4cells[cell,2],1] * (Grid.coords4nodes[Grid.nodes4cells[cell,3],2] - Grid.coords4nodes[Grid.nodes4cells[cell,1],2])
#             + Grid.coords4nodes[Grid.nodes4cells[cell,3],1] * (Grid.coords4nodes[Grid.nodes4cells[cell,1],2] - Grid.coords4nodes[Grid.nodes4cells[cell,2],2]));
#             end        
#         elseif typeof(Grid.elemtypes[1]) <: ElemType2DParallelogram
#             for cell = 1 : ncells 
#                 Grid.volume4cells[cell] = (
#                Grid.coords4nodes[Grid.nodes4cells[cell,1],1] * (Grid.coords4nodes[Grid.nodes4cells[cell,2],2] -  Grid.coords4nodes[Grid.nodes4cells[cell,3],2])
#             + Grid.coords4nodes[Grid.nodes4cells[cell,2],1] * (Grid.coords4nodes[Grid.nodes4cells[cell,3],2] - Grid.coords4nodes[Grid.nodes4cells[cell,1],2])
#             + Grid.coords4nodes[Grid.nodes4cells[cell,3],1] * (Grid.coords4nodes[Grid.nodes4cells[cell,1],2] - Grid.coords4nodes[Grid.nodes4cells[cell,2],2]));
#             end      
#         elseif typeof(Grid.elemtypes[1]) <: ElemType2DQuadrilateral
#             for cell = 1 : ncells 
#                 Grid.volume4cells[cell] = 0.5 * ((Grid.coords4nodes[Grid.nodes4cells[cell,1],1] - Grid.coords4nodes[Grid.nodes4cells[cell,3],1]) * (Grid.coords4nodes[Grid.nodes4cells[cell,2],2] - Grid.coords4nodes[Grid.nodes4cells[cell,4],2]) + (Grid.coords4nodes[Grid.nodes4cells[cell,4],1] - Grid.coords4nodes[Grid.nodes4cells[cell,2],1]) * (Grid.coords4nodes[Grid.nodes4cells[cell,1],2] - Grid.coords4nodes[Grid.nodes4cells[cell,3],2]));
#             end        
#        # elseif typeof(Grid.elemtypes[1]) <: ElemType3DTetraeder
#        #    A = ones(eltype(Grid.coords4nodes),4,4);
#        #    for cell = 1 : ncells 
#        #     A[1,[2,3,4]] = Grid.coords4nodes[Grid.nodes4cells[cell,1],:];
#        #     A[2,[2,3,4]] = Grid.coords4nodes[Grid.nodes4cells[cell,2],:];
#        #     A[3,[2,3,4]] = Grid.coords4nodes[Grid.nodes4cells[cell,3],:];
#        #     A[4,[2,3,4]] = Grid.coords4nodes[Grid.nodes4cells[cell,4],:];
#        #     Grid.volume4cells[cell] = 1 // 6 * abs(det(A));
#        #    end
#         end
#     end        
# end   

# # determine the face numbers of the boundary faces
# # (they appear only once in faces4cells)
# function ensure_bfaces!(Grid::Mesh)
#     dim = get_dimension(Grid.elemtypes[1])
#     @assert dim <= 3
#     if size(Grid.bfaces,1) <= 0
#         ensure_faces4cells!(Grid::Mesh)
#         ncells = size(Grid.faces4cells,1);    
#         nfaces = size(Grid.nodes4faces,1);
#         takeface = zeros(Bool,nfaces);
#         for cell = 1 : ncells
#             for j = 1 : size(Grid.faces4cells,2)
#                 @inbounds takeface[Grid.faces4cells[cell,j]] = .!takeface[Grid.faces4cells[cell,j]];
#             end    
#         end
#         Grid.bfaces = findall(takeface);
#         Grid.bregions = ones(Int64,length(Grid.bfaces));
#     end
# end

# function assign_boundaryregions!(grid,nodes4bfaces,bregions)
#     ensure_bfaces!(grid)
#     ensure_nodes4faces!(grid)
#     nodes1 = [0, 0]
#     nodes2 = [0, 0]
#     for j = 1 : length(bregions)
#         nodes1[:] = sort(nodes4bfaces[:,j])
#         for k = 1 : length(grid.bfaces)
#             nodes2[:] = sort(grid.nodes4faces[grid.bfaces[k],:])
#             if (nodes1 == nodes2)
#                 grid.bregions[k] = bregions[j]
#                 break;
#             end
#         end        
#     end
# end

# # compute nodes4faces (implicating an enumeration of the faces)
# function ensure_nodes4faces!(Grid::Mesh)
#     dim = get_dimension(Grid.elemtypes[1])
#     @assert dim <= 3
#     ncells::Int = size(Grid.nodes4cells,1);
#     index::Int = 0;
#     temp::Int64 = 0;
#     if (size(Grid.nodes4faces,1) <= 0)
#         if dim == 1
#             # in 1D nodes are faces
#             nnodes::Int = size(Grid.coords4nodes,1);
#             Grid.nodes4faces = zeros(Int64,nnodes,1);
#             Grid.nodes4faces[:] = 1:nnodes;
#         elseif dim == 2
#             nodes_per_cell = size(Grid.nodes4cells,2)
#             helperfield = zeros(Int64,nodes_per_cell,1)
#             helperfield[:] = 0:nodes_per_cell-1
#             helperfield[1] = nodes_per_cell
#             # compute nodes4faces with duplicates
#             Grid.nodes4faces = zeros(Int64,nodes_per_cell*ncells,2);
#             index = 1
#             for cell = 1 : ncells
#                 for k = 1 : nodes_per_cell
#                     Grid.nodes4faces[index,1] = Grid.nodes4cells[cell,k];
#                     Grid.nodes4faces[index,2] = Grid.nodes4cells[cell,helperfield[k]];
#                     index += 1;
#                 end    
#             end    
    
#             # sort each row ( faster than: sort!(Grid.nodes4faces, dims = 2);)
#             for j = 1 : nodes_per_cell*ncells
#                 if Grid.nodes4faces[j,2] > Grid.nodes4faces[j,1]
#                     temp = Grid.nodes4faces[j,1];
#                     Grid.nodes4faces[j,1] = Grid.nodes4faces[j,2];
#                     Grid.nodes4faces[j,2] = temp;
#                 end
#             end
        
#             # find unique rows -> this fixes the enumeration of the faces!
#             Grid.nodes4faces = unique(Grid.nodes4faces, dims = 1);
#         elseif dim == 3 # might work for tets
#             # compute nodes4faces with duplicates
#             Grid.nodes4faces = zeros(Int64,4*ncells,3);
#             for cell = 1 : ncells
#                 Grid.nodes4faces[index+1,1] = Grid.nodes4cells[cell,1];
#                 Grid.nodes4faces[index+1,2] = Grid.nodes4cells[cell,2];
#                 Grid.nodes4faces[index+1,3] = Grid.nodes4cells[cell,3];
#                 Grid.nodes4faces[index+2,1] = Grid.nodes4cells[cell,2]; 
#                 Grid.nodes4faces[index+2,2] = Grid.nodes4cells[cell,3]; 
#                 Grid.nodes4faces[index+2,3] = Grid.nodes4cells[cell,4];
#                 Grid.nodes4faces[index+3,1] = Grid.nodes4cells[cell,3];
#                 Grid.nodes4faces[index+3,2] = Grid.nodes4cells[cell,4];
#                 Grid.nodes4faces[index+3,3] = Grid.nodes4cells[cell,1];
#                 Grid.nodes4faces[index+4,1] = Grid.nodes4cells[cell,4];
#                 Grid.nodes4faces[index+4,2] = Grid.nodes4cells[cell,1];
#                 Grid.nodes4faces[index+4,3] = Grid.nodes4cells[cell,2];
#                 index += 4;
#             end    
    
#             # sort each row ( faster than: sort!(Grid.nodes4faces, dims = 2);)
#             for j = 1 : 4*ncells
#                 if Grid.nodes4faces[j,2] > Grid.nodes4faces[j,1]
#                     temp = Grid.nodes4faces[j,1];
#                     Grid.nodes4faces[j,1] = Grid.nodes4faces[j,2];
#                     Grid.nodes4faces[j,2] = temp;
#                 end
#                 if Grid.nodes4faces[j,3] > Grid.nodes4faces[j,2]
#                     temp = Grid.nodes4faces[j,2];
#                     Grid.nodes4faces[j,2] = Grid.nodes4faces[j,3];
#                     Grid.nodes4faces[j,3] = temp;
#                 end
#                 if Grid.nodes4faces[j,2] > Grid.nodes4faces[j,1]
#                     temp = Grid.nodes4faces[j,1];
#                     Grid.nodes4faces[j,1] = Grid.nodes4faces[j,2];
#                     Grid.nodes4faces[j,2] = temp;
#                 end
#             end
        
#             # find unique rows -> this fixes the enumeration of the faces!
#             Grid.nodes4faces = unique(Grid.nodes4faces, dims = 1);    
#         end
#     end    
# end

# # compute faces4cells
# function ensure_faces4cells!(Grid::Mesh)
#     dim = get_dimension(Grid.elemtypes[1])
#     @assert dim <= 3
#     if size(Grid.faces4cells,1) != size(Grid.nodes4cells,1)
#         ensure_nodes4faces!(Grid)
#         if dim == 1
#             # in 1D nodes are faces
#             Grid.faces4cells = Grid.nodes4cells;        
#         elseif dim == 2
#             nodes_per_cell = size(Grid.nodes4cells,2)
#             nnodes = size(Grid.coords4nodes,1);
#             nfaces = size(Grid.nodes4faces,1);
#             ncells = size(Grid.nodes4cells,1);
    
#             face4nodes = sparse(view(Grid.nodes4faces,:,1),view(Grid.nodes4faces,:,2),1:nfaces,nnodes,nnodes);
#             face4nodes = face4nodes + face4nodes';
    
#             Grid.faces4cells = zeros(Int,size(Grid.nodes4cells,1),nodes_per_cell);
#             helperfield = zeros(Int64,nodes_per_cell,1)
#             helperfield[:] = 2:nodes_per_cell+1
#             helperfield[end] = 1
#             for cell = 1 : ncells
#                 for k = 1 : nodes_per_cell
#                     Grid.faces4cells[cell,k] = face4nodes[Grid.nodes4cells[cell,k],Grid.nodes4cells[cell,helperfield[k]]];
#                 end    
#             end
#         elseif dim == 3    
#             # todo
#             println("faces4cells for tets not yet implemented!")
#             @assert dim <= 3
#         end
#     end    
# end


# # compute signs4cells
# function ensure_signs4cells!(Grid::Mesh)
#     dim = get_dimension(Grid.elemtypes[1])
#     @assert dim <= 3
#     if size(Grid.signs4cells,1) != size(Grid.nodes4cells,1)
#         ensure_faces4cells!(Grid)
#         ncells::Int64 = size(Grid.nodes4cells,1)
#         Grid.signs4cells = ones(ncells,size(Grid.faces4cells,2))
#         for cell = 1 : ncells
#             for f = 1 : size(Grid.faces4cells,2)
#                 if Grid.nodes4faces[Grid.faces4cells[cell,f],1] != Grid.nodes4cells[cell,f]
#                     Grid.signs4cells[cell,f] = -1
#                 end       
#             end
#         end
#     end   
# end


# # compute cells4faces
# function ensure_cells4faces!(Grid::Mesh)
#     dim = get_dimension(Grid.elemtypes[1])
#     @assert dim <= 3
#     ensure_nodes4faces!(Grid)
#     nfaces::Int = size(Grid.nodes4faces,1);
#     ensure_faces4cells!(Grid)
#     if size(Grid.cells4faces,1) != nfaces
#         Grid.cells4faces = zeros(Int,nfaces,2);
#         for j = 1:size(Grid.faces4cells,1) 
#             for k = 1:size(Grid.faces4cells,2)
#                 if Grid.cells4faces[Grid.faces4cells[j,k],1] == 0
#                     Grid.cells4faces[Grid.faces4cells[j,k],1] = j
#                 else    
#                     Grid.cells4faces[Grid.faces4cells[j,k],2] = j
#                 end    
#             end
#         end
#     end
# end


# # compute normal4faces
# function ensure_normal4faces!(Grid::Mesh)
#     dim::Int = size(Grid.nodes4cells,2)
#     ensure_nodes4faces!(Grid)
#     xdim::Int = size(Grid.coords4nodes,2)
#     nfaces::Int = size(Grid.nodes4faces,1);
#     if size(Grid.normal4faces,1) != nfaces
#         Grid.normal4faces = zeros(Int,nfaces,xdim);
#         for j = 1:nfaces 
#             # rotate tangent
#             Grid.normal4faces[j,:] = Grid.coords4nodes[Grid.nodes4faces[j,1],[2,1]] - Grid.coords4nodes[Grid.nodes4faces[j,2],[2,1]];
#             Grid.normal4faces[j,1] *= -1
#             # divide by length
#             Grid.normal4faces[j,:] ./= sqrt(dot(Grid.normal4faces[j,:],Grid.normal4faces[j,:]))
            
#         end
#     end
# end


# function get_boundary_grid(Grid::Mesh);
#     ensure_nodes4faces!(Grid)
#     ensure_bfaces!(Grid)
#     return Mesh(Grid.coords4nodes,Grid.nodes4faces[Grid.bfaces,:]);
# end

end # module
