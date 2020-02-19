struct ElemType2DTriangle <: Abstract2DElemType end


function get_reference_cordinates(::ElemType2DTriangle)
    return [0 0;
            1 0;
            0 1]
end

function get_face_elemtype(::ElemType2DTriangle)
    return ElemType1DInterval();
end


# perform a barycentric refinement of 2D triangles
function barycentric_refinement(::ElemType2DTriangle,coords4nodes::Array,nodes4cells::Array)
    
    nnodes = size(coords4nodes,1);
    ncells = size(nodes4cells,1);

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
    
    # compute and append face midpoints and cell midpoints
    coords4nodes = @views [coords4nodes;
                           1 // 2 * (coords4nodes[nodes4faces[:,1],:] + coords4nodes[nodes4faces[:,2],:])
                           1 // 3 * (coords4nodes[nodes4cells[:,1],:] + coords4nodes[nodes4cells[:,2],:] + coords4nodes[nodes4cells[:,3],:])];
    
    # mapping to get number of new mipoint between two old nodes
    newnode4nodes = @views sparse(nodes4faces[:,1],nodes4faces[:,2],(1:nfaces) .+ nnodes,nnodes,nnodes);
    newnode4nodes = newnode4nodes + newnode4nodes';
    
    # build up new nodes4cells of uniform refinements
    nodes4cells_new = zeros(Int,6*ncells,3);
    newnodes = zeros(Int,3);
    for cell = 1 : ncells
        newnodes = map(j->(newnode4nodes[nodes4cells[cell,j],nodes4cells[cell,mod(j,3)+1]]),1:3);
        newcenternode = nfaces + nnodes + cell
        nodes4cells_new[(1:6) .+ (cell-1)*6,:] = 
            [nodes4cells[cell,1] newnodes[1] newcenternode;
             newnodes[1] nodes4cells[cell,2] newcenternode;
             nodes4cells[cell,2] newnodes[2] newcenternode;
             newnodes[2] nodes4cells[cell,3] newcenternode;
             nodes4cells[cell,3] newnodes[3] newcenternode;
             newnodes[3] nodes4cells[cell,1] newcenternode]
    end  
  return coords4nodes, nodes4cells_new;
end



# perform a uniform (red) refinement of 2D triangles
function uniform_refinement(::ElemType2DTriangle,coords4nodes::Array,nodes4cells::Array)
    
    nnodes = size(coords4nodes,1);
    ncells = size(nodes4cells,1);

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
  return coords4nodes, nodes4cells_new;
end


# transformation for H1 elements on a triangle
function local2global(grid::Mesh, ::ElemType2DTriangle)
    A = Matrix{Float64}(undef,2,2)
    b = Vector{Float64}(undef,2)
    x = Vector{Real}(undef,2)
    return function fix_cell(cell)
        b[1] = grid.coords4nodes[grid.nodes4cells[cell,1],1]
        b[2] = grid.coords4nodes[grid.nodes4cells[cell,1],2]
        A[1,1] = grid.coords4nodes[grid.nodes4cells[cell,2],1] - b[1]
        A[1,2] = grid.coords4nodes[grid.nodes4cells[cell,3],1] - b[1]
        A[2,1] = grid.coords4nodes[grid.nodes4cells[cell,2],2] - b[2]
        A[2,2] = grid.coords4nodes[grid.nodes4cells[cell,3],2] - b[2]
        return function closure(xref)
            x[1] = A[1,1]*xref[1] + A[1,2]*xref[2] + b[1]
            x[2] = A[2,1]*xref[1] + A[2,2]*xref[2] + b[2]
            # faster than: x = A*xref + b
            return x
        end
    end    
end

# transformation for H1 elements on a face
function local2global_face(grid::Mesh, ::ElemType2DTriangle)
    A = Matrix{Float64}(undef,2,1)
    b = Vector{Float64}(undef,2)
    x = Vector{Real}(undef,2)
    return function fix_face(face)
        b[1] = grid.coords4nodes[grid.nodes4faces[face,1],1]
        b[2] = grid.coords4nodes[grid.nodes4faces[face,1],2]
        A[1,1] = grid.coords4nodes[grid.nodes4faces[face,2],1] - b[1]
        A[2,1] = grid.coords4nodes[grid.nodes4faces[face,2],2] - b[2]
        return function closure(xref)
            x[1] = A[1,1]*xref[1] + b[1]
            x[2] = A[2,1]*xref[1] + b[2]
            return x
        end
    end    
end

# exact tinversion of transformation for H1 elements on a line
function local2global_tinv_jacobian(grid::Mesh, ::ElemType2DTriangle)
    det = 0.0
    function closure(A!::Array{T,2},cell::Int64) where T <: Real
        # transposed inverse of A
        A![2,2] = grid.coords4nodes[grid.nodes4cells[cell,2],1] - grid.coords4nodes[grid.nodes4cells[cell,1],1]
        A![2,1] = -grid.coords4nodes[grid.nodes4cells[cell,3],1] + grid.coords4nodes[grid.nodes4cells[cell,1],1]
        A![1,2] = -grid.coords4nodes[grid.nodes4cells[cell,2],2] + grid.coords4nodes[grid.nodes4cells[cell,1],2]
        A![1,1] = grid.coords4nodes[grid.nodes4cells[cell,3],2] - grid.coords4nodes[grid.nodes4cells[cell,1],2]
        
        # divide by  determinant
        det = A![1,1]*A![2,2] - A![1,2]*A![2,1]
    
        A![1] = A![1]/det
        A![2] = A![2]/det
        A![3] = A![3]/det
        A![4] = A![4]/det
        return det
    end
end


# Piola transformation for Hdiv elements on a triangle
function local2global_Piola(grid::Mesh, ::ElemType2DTriangle)
    det = 0.0;
    return function closure(A!::Array{T,2},cell::Int64) where T <: Real
        A![1,1] = grid.coords4nodes[grid.nodes4cells[cell,2],1] - grid.coords4nodes[grid.nodes4cells[cell,1],1]
        A![1,2] = grid.coords4nodes[grid.nodes4cells[cell,3],1] - grid.coords4nodes[grid.nodes4cells[cell,1],1]
        A![2,1] = grid.coords4nodes[grid.nodes4cells[cell,2],2] - grid.coords4nodes[grid.nodes4cells[cell,1],2]
        A![2,2] = grid.coords4nodes[grid.nodes4cells[cell,3],2] - grid.coords4nodes[grid.nodes4cells[cell,1],2]
        det = A![1,1]*A![2,2] - A![1,2]*A![2,1]
        return det
    end    
end