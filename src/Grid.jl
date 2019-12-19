module Grid

using SparseArrays
using LinearAlgebra

export Mesh, ensure_volume4cells!, ensure_bfaces!, ensure_faces4cells!, ensure_nodes4faces!, ensure_cells4faces!, ensure_normal4faces!, get_boundary_grid

mutable struct Mesh{T <: Real}
    coords4nodes::Array{T,2}
    nodes4cells::Array{Int,2}
    
    volume4cells::Array{T,1}
    nodes4faces::Array{Int,2}
    faces4cells::Array{Int,2}
    bfaces::Array{Int,1}
    cells4faces::Array{Int,2}
    normal4faces::Array{T,2}
    
    function Mesh{T}(coords,nodes) where {T<:Real}
        # only 2d triangulations allowed yet
        new(coords,nodes,[],[[] []],[[] []],[],[[] []],[[] []]);
    end
end


function Mesh{T}(coords,nodes,nrefinements) where {T<:Real}
    for j=1:nrefinements
        @assert size(nodes,2) <= 3
        coords, nodes = uniform_refinement(coords,nodes)
    end
    return Mesh{T}(coords,nodes);
end

# default constructor for Float64-typed triangulations
function Mesh(coords,nodes,nrefinements = 0)
    for j=1:nrefinements
        coords, nodes = uniform_refinement(coords,nodes)
    end
    return Mesh{Float64}(coords,nodes);
end


# perform a uniform (red) refinement of the triangulation
function uniform_refinement(coords4nodes::Array,nodes4cells::Array)
    
  nnodes = size(coords4nodes,1);
  ncells = size(nodes4cells,1);
    
  if size(coords4nodes,2) == 1
    coords4nodes = @views [coords4nodes; 1 // 2 * (coords4nodes[nodes4cells[:,1],1] + coords4nodes[nodes4cells[:,2],1])];
    
    nodes4cells_new = zeros(Int,2*ncells,2);
    for cell = 1 : ncells
        nodes4cells_new[(1:2) .+ (cell-1)*2,:] = 
            [nodes4cells[cell,1] nnodes+cell;
            nnodes+cell nodes4cells[cell,2]];
    end
    
  elseif size(coords4nodes,2) == 2
    # compute nodes4faces
    nodes4faces = @views [nodes4cells[:,1] nodes4cells[:,2]; nodes4cells[:,2] nodes4cells[:,3]; nodes4cells[:,3] nodes4cells[:,1]];
    
    # sort each row ( faster than: sort!(Grid.nodes4faces, dims = 2);)
    temp::Int64 = 0;
    for j = 1 : 3*ncells
        if nodes4faces[j,2] > nodes4faces[j,1]
            temp = nodes4faces[j,1];
            nodes4faces[j,1] = nodes4faces[j,2];
            nodes4faces[j,2] = temp;
        end
    end
        
    # find unique rows -> this fixes the enumeration of the faces!
    nodes4faces = unique(nodes4faces, dims = 1);
    nfaces = size(nodes4faces,1);
    
    # compute and append face midpoints
    coords4nodes = @views [coords4nodes; 1 // 2 * (coords4nodes[nodes4faces[:,1],:] + coords4nodes[nodes4faces[:,2],:])];
    
    # mapping to get number of new mipoint between two old nodes
    newnode4nodes = @views sparse(nodes4faces[:,1],nodes4faces[:,2],(1:nfaces) .+ nnodes,nnodes,nnodes);
    newnode4nodes = newnode4nodes + newnode4nodes';
    
    # build up new nodes4cells of uniform refinements
    nodes4cells_new = zeros(Int,4*ncells,3);
    newnodes = zeros(Int,3);
    for cell = 1 : ncells
        newnodes = map(j->(newnode4nodes[nodes4cells[cell,j],nodes4cells[cell,mod(j,3)+1]]),1:3);
        nodes4cells_new[(1:4) .+ (cell-1)*4,:] = 
            [nodes4cells[cell,1] newnodes[1] newnodes[3];
            newnodes[1] nodes4cells[cell,2] newnodes[2];
            newnodes[2] newnodes[3] newnodes[1];
            newnodes[3] newnodes[2] nodes4cells[cell,3]];
    end
  end  
  return coords4nodes, nodes4cells_new;
end


# perform a uniform (red) refinement of the triangulation
function uniform_refinement_old(coords4nodes::Array,nodes4cells::Array)
    
  nnodes = size(coords4nodes,1);
  ncells = size(nodes4cells,1);
    
  if size(coords4nodes,2) == 1
    coords4nodes = [coords4nodes; 1 // 2 * (coords4nodes[nodes4cells[:,1],1] + coords4nodes[nodes4cells[:,2],1])];
    
    nodes4cells_new = zeros(Int,2*ncells,2);
    for cell = 1 : ncells
        nodes4cells_new[(1:2) .+ (cell-1)*2,:] = 
            [nodes4cells[cell,1] nnodes+cell;
            nnodes+cell nodes4cells[cell,2]];
    end
    
  elseif size(coords4nodes,2) == 2
    # compute nodes4faces
    nodes4faces = [nodes4cells[:,1] nodes4cells[:,2]; nodes4cells[:,2] nodes4cells[:,3]; nodes4cells[:,3] nodes4cells[:,1]];
    # find unique rows -> this fixes the enumeration of the faces!
    sort!(nodes4faces, dims = 2); # sort each row
    nodes4faces = unique(nodes4faces, dims = 1);
    nfaces = size(nodes4faces,1);
    
    # compute and append face midpoints
    coords4nodes = [coords4nodes; 1 // 2 * (coords4nodes[nodes4faces[:,1],:] + coords4nodes[nodes4faces[:,2],:])];
    
    # mapping to get number of new mipoint between two old nodes
    newnode4nodes = sparse(nodes4faces[:,1],nodes4faces[:,2],(1:nfaces) .+       nnodes,nnodes,nnodes);
    newnode4nodes = newnode4nodes + newnode4nodes';
    
    # build up new nodes4cells of uniform refinements
    nodes4cells_new = zeros(Int,4*ncells,3);
    newnodes = zeros(Int,3);
    for cell = 1 : ncells
        newnodes = map(j->(newnode4nodes[nodes4cells[cell,j],nodes4cells[cell,mod(j,3)+1]]),1:3);
        nodes4cells_new[(1:4) .+ (cell-1)*4,:] = 
            [nodes4cells[cell,1] newnodes[1] newnodes[3];
            newnodes[1] nodes4cells[cell,2] newnodes[2];
            newnodes[2] newnodes[3] newnodes[1];
            newnodes[3] newnodes[2] nodes4cells[cell,3]];
    end
  end  
  return coords4nodes, nodes4cells_new;
end

function ensure_volume4cells!(Grid::Mesh)
    celldim = size(Grid.nodes4cells,2) - 1;
    ncells::Int = size(Grid.nodes4cells,1);
    if size(Grid.volume4cells,1) != size(ncells,1)
        Grid.volume4cells = zeros(eltype(Grid.coords4nodes),ncells);
        if celldim == 1 # also allow d-dimensional points on a line!
            Grid.volume4cells = zeros(eltype(Grid.coords4nodes),ncells);
            xdim::Int = size(Grid.coords4nodes,2)
            for cell = 1 : ncells
                for d = 1 : xdim
                    Grid.volume4cells[cell] += (Grid.coords4nodes[Grid.nodes4cells[cell,2],d] - Grid.coords4nodes[Grid.nodes4cells[cell,1],d]).^2
                end
                 Grid.volume4cells[cell] = sqrt(Grid.volume4cells[cell]);    
            end    
        elseif celldim == 2
            for cell = 1 : ncells 
                Grid.volume4cells[cell] = 1 // 2 * (
               Grid.coords4nodes[Grid.nodes4cells[cell,1],1] * (Grid.coords4nodes[Grid.nodes4cells[cell,2],2] -  Grid.coords4nodes[Grid.nodes4cells[cell,3],2])
            + Grid.coords4nodes[Grid.nodes4cells[cell,2],1] * (Grid.coords4nodes[Grid.nodes4cells[cell,3],2] - Grid.coords4nodes[Grid.nodes4cells[cell,1],2])
            + Grid.coords4nodes[Grid.nodes4cells[cell,3],1] * (Grid.coords4nodes[Grid.nodes4cells[cell,1],2] - Grid.coords4nodes[Grid.nodes4cells[cell,2],2]));
            end    
            
        elseif celldim == 3
           A = ones(eltype(Grid.coords4nodes),4,4);
           for cell = 1 : ncells 
            A[1,[2,3,4]] = Grid.coords4nodes[Grid.nodes4cells[cell,1],:];
            A[2,[2,3,4]] = Grid.coords4nodes[Grid.nodes4cells[cell,2],:];
            A[3,[2,3,4]] = Grid.coords4nodes[Grid.nodes4cells[cell,3],:];
            A[4,[2,3,4]] = Grid.coords4nodes[Grid.nodes4cells[cell,4],:];
            Grid.volume4cells[cell] = 1 // 6 * abs(det(A));
           end
        end
    end        
end    


function ensure_volume4cells_old!(Grid::Mesh)
    celldim = size(Grid.nodes4cells,2) - 1;
    ncells::Int = size(Grid.nodes4cells,1);
    @assert celldim <= 2
    if size(Grid.volume4cells,1) != size(ncells,1)
        if celldim == 1 # also allow d-dimensional points on a line!
            Grid.volume4cells = zeros(eltype(Grid.coords4nodes),ncells);
            xdim::Int = size(Grid.coords4nodes,2)
            for cell = 1 : ncells
                for d = 1 : xdim
                    Grid.volume4cells[cell] += (Grid.coords4nodes[Grid.nodes4cells[cell,2],d] - Grid.coords4nodes[Grid.nodes4cells[cell,1],d]).^2
                end
                 Grid.volume4cells[cell] = sqrt(Grid.volume4cells[cell]);    
            end    
        elseif celldim == 2
            Grid.volume4cells = 1 // 2 *(
               Grid.coords4nodes[Grid.nodes4cells[:,1],1] .* (Grid.coords4nodes[Grid.nodes4cells[:,2],2] -  Grid.coords4nodes[Grid.nodes4cells[:,3],2])
            .+ Grid.coords4nodes[Grid.nodes4cells[:,2],1] .* (Grid.coords4nodes[Grid.nodes4cells[:,3],2] - Grid.coords4nodes[Grid.nodes4cells[:,1],2])
            .+ Grid.coords4nodes[Grid.nodes4cells[:,3],1] .* (Grid.coords4nodes[Grid.nodes4cells[:,1],2] - Grid.coords4nodes[Grid.nodes4cells[:,2],2]));
        elseif celldim == 3
            # todo
        end
    end        
end  

# determine the face numbers of the boundary faces
# (they appear only once in faces4cells)
function ensure_bfaces!(Grid::Mesh)
    dim::Int = size(Grid.nodes4cells,2)
    @assert dim <= 3
    if size(Grid.bfaces,1) <= 0
        ensure_faces4cells!(Grid::Mesh)
        ncells = size(Grid.faces4cells,1);    
        nfaces = size(Grid.nodes4faces,1);
        takeface = zeros(Bool,nfaces);
        for cell = 1 : ncells
            for j = 1 : dim
                @inbounds takeface[Grid.faces4cells[cell,j]] = .!takeface[Grid.faces4cells[cell,j]];
            end    
        end
        Grid.bfaces = findall(takeface);
    end
end

# compute nodes4faces (implicating an enumeration of the faces)
function ensure_nodes4faces!(Grid::Mesh)
    dim::Int = size(Grid.nodes4cells,2)
    @assert dim <= 4
    ncells::Int = size(Grid.nodes4cells,1);
    index::Int = 0;
    temp::Int64 = 0;
    if (size(Grid.nodes4faces,1) <= 0)
        if dim == 2
            # in 1D nodes are faces
            nnodes::Int = size(Grid.coords4nodes,1);
            Grid.nodes4faces = zeros(Int64,nnodes,1);
            Grid.nodes4faces[:] = 1:nnodes;
        elseif dim == 3
            # compute nodes4faces with duplicates
            Grid.nodes4faces = zeros(Int64,3*ncells,2);
            for cell = 1 : ncells
                Grid.nodes4faces[index+1,1] = Grid.nodes4cells[cell,1];
                Grid.nodes4faces[index+1,2] = Grid.nodes4cells[cell,2];
                Grid.nodes4faces[index+2,1] = Grid.nodes4cells[cell,2]; 
                Grid.nodes4faces[index+2,2] = Grid.nodes4cells[cell,3];
                Grid.nodes4faces[index+3,1] = Grid.nodes4cells[cell,3];
                Grid.nodes4faces[index+3,2] = Grid.nodes4cells[cell,1];
                index += 3;
            end    
    
            # sort each row ( faster than: sort!(Grid.nodes4faces, dims = 2);)
            for j = 1 : 3*ncells
                if Grid.nodes4faces[j,2] > Grid.nodes4faces[j,1]
                    temp = Grid.nodes4faces[j,1];
                    Grid.nodes4faces[j,1] = Grid.nodes4faces[j,2];
                    Grid.nodes4faces[j,2] = temp;
                end
            end
        
            # find unique rows -> this fixes the enumeration of the faces!
            Grid.nodes4faces = unique(Grid.nodes4faces, dims = 1);
        elseif dim == 4
            # compute nodes4faces with duplicates
            Grid.nodes4faces = zeros(Int64,4*ncells,3);
            for cell = 1 : ncells
                Grid.nodes4faces[index+1,1] = Grid.nodes4cells[cell,1];
                Grid.nodes4faces[index+1,2] = Grid.nodes4cells[cell,2];
                Grid.nodes4faces[index+1,3] = Grid.nodes4cells[cell,3];
                Grid.nodes4faces[index+2,1] = Grid.nodes4cells[cell,2]; 
                Grid.nodes4faces[index+2,2] = Grid.nodes4cells[cell,3]; 
                Grid.nodes4faces[index+2,3] = Grid.nodes4cells[cell,4];
                Grid.nodes4faces[index+3,1] = Grid.nodes4cells[cell,3];
                Grid.nodes4faces[index+3,2] = Grid.nodes4cells[cell,4];
                Grid.nodes4faces[index+3,3] = Grid.nodes4cells[cell,1];
                Grid.nodes4faces[index+4,1] = Grid.nodes4cells[cell,4];
                Grid.nodes4faces[index+4,2] = Grid.nodes4cells[cell,1];
                Grid.nodes4faces[index+4,3] = Grid.nodes4cells[cell,2];
                index += 4;
            end    
    
            # sort each row ( faster than: sort!(Grid.nodes4faces, dims = 2);)
            for j = 1 : 4*ncells
                if Grid.nodes4faces[j,2] > Grid.nodes4faces[j,1]
                    temp = Grid.nodes4faces[j,1];
                    Grid.nodes4faces[j,1] = Grid.nodes4faces[j,2];
                    Grid.nodes4faces[j,2] = temp;
                end
                if Grid.nodes4faces[j,3] > Grid.nodes4faces[j,2]
                    temp = Grid.nodes4faces[j,2];
                    Grid.nodes4faces[j,2] = Grid.nodes4faces[j,3];
                    Grid.nodes4faces[j,3] = temp;
                end
                if Grid.nodes4faces[j,2] > Grid.nodes4faces[j,1]
                    temp = Grid.nodes4faces[j,1];
                    Grid.nodes4faces[j,1] = Grid.nodes4faces[j,2];
                    Grid.nodes4faces[j,2] = temp;
                end
            end
        
            # find unique rows -> this fixes the enumeration of the faces!
            Grid.nodes4faces = unique(Grid.nodes4faces, dims = 1);    
        end
    end    
end

# compute faces4cells
function ensure_faces4cells!(Grid::Mesh)
    dim::Int = size(Grid.nodes4cells,2)
    if size(Grid.faces4cells,1) != size(Grid.nodes4cells,1)
        ensure_nodes4faces!(Grid)
        if dim == 2
            # in 1D nodes are faces
            Grid.faces4cells = Grid.nodes4cells;        
        elseif dim == 3
            nnodes = size(Grid.coords4nodes,1);
            nfaces = size(Grid.nodes4faces,1);
            ncells = size(Grid.nodes4cells,1);
    
            face4nodes = sparse(view(Grid.nodes4faces,:,1),view(Grid.nodes4faces,:,2),1:nfaces,nnodes,nnodes);
            face4nodes = face4nodes + face4nodes';
    
            Grid.faces4cells = zeros(Int,size(Grid.nodes4cells,1),3);
            for cell = 1 : ncells
                Grid.faces4cells[cell,1] = face4nodes[Grid.nodes4cells[cell,1],Grid.nodes4cells[cell,2]];
                Grid.faces4cells[cell,2] = face4nodes[Grid.nodes4cells[cell,2],Grid.nodes4cells[cell,3]];
                Grid.faces4cells[cell,3] = face4nodes[Grid.nodes4cells[cell,3],Grid.nodes4cells[cell,1]];
            end
        elseif dim == 4    
            # todo
            println("faces4cells for tets not yet implemented!")
            @assert dim <= 3
        end
    end    
end


# compute cells4faces
function ensure_cells4faces!(Grid::Mesh)
    dim::Int = size(Grid.nodes4cells,2)
    ensure_nodes4faces!(Grid)
    nfaces::Int = size(Grid.nodes4faces,1);
    ensure_faces4cells!(Grid)
    if size(Grid.cells4faces,1) != nfaces
        Grid.cells4faces = zeros(Int,nfaces,2);
        for j = 1:size(Grid.faces4cells,1) 
            for k = 1:size(Grid.faces4cells,2)
                if Grid.cells4faces[Grid.faces4cells[j,k],1] == 0
                    Grid.cells4faces[Grid.faces4cells[j,k],1] = j
                else    
                    Grid.cells4faces[Grid.faces4cells[j,k],2] = j
                end    
            end
        end
    end
end


# compute normal4faces
function ensure_normal4faces!(Grid::Mesh)
    dim::Int = size(Grid.nodes4cells,2)
    ensure_nodes4faces!(Grid)
    xdim::Int = size(Grid.coords4nodes,2)
    nfaces::Int = size(Grid.nodes4faces,1);
    if size(Grid.normal4faces,1) != nfaces
        Grid.normal4faces = zeros(Int,nfaces,xdim);
        for j = 1:nfaces 
            # rotate tangent
            Grid.normal4faces[j,:] = Grid.coords4nodes[Grid.nodes4faces[j,1],[2,1]] - Grid.coords4nodes[Grid.nodes4faces[j,2],[2,1]];
            Grid.normal4faces[j,1] *= -1
            # divide by length
            Grid.normal4faces[j,:] ./= sqrt(dot(Grid.normal4faces[j,:],Grid.normal4faces[j,:]))
            
        end
    end
end


function get_boundary_grid(Grid::Mesh);
    ensure_nodes4faces!(Grid)
    ensure_bfaces!(Grid)
    return Mesh(Grid.coords4nodes,Grid.nodes4faces[Grid.bfaces,:]);
end

end # module
