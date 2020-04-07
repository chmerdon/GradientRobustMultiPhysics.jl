struct ElemType2DParallelogram <: Abstract2DElemType end


function get_reference_cordinates(::ElemType2DParallelogram)
    return [0 0;
            1 0;
            1 1;
            0 1]
end

function get_face_elemtype(::ElemType2DParallelogram)
    return ElemType1DInterval();
end


# perform a barycentric refinement of 2D parallelograms
function uniform_refinement(::ElemType2DParallelogram,coords4nodes::Array,nodes4cells::Array,nodes4bfaces::Array = [[] []],bregions::Array = [])
    
    nnodes = size(coords4nodes,1);
    ncells = size(nodes4cells,1);

    # compute nodes4faces
    nodes4faces = @views [nodes4cells[:,1] nodes4cells[:,2]; nodes4cells[:,2] nodes4cells[:,3]; nodes4cells[:,3] nodes4cells[:,4]; nodes4cells[:,4] nodes4cells[:,1]];
    
    # sort each row ( faster than: sort!(Grid.nodes4faces, dims = 2);)
    temp::Int64 = 0;
    for j = 1 : 4*ncells
        if nodes4faces[j,2] > nodes4faces[j,1]
            temp = nodes4faces[j,1];
            nodes4faces[j,1] = nodes4faces[j,2];
            nodes4faces[j,2] = temp;
        end
    end
        
    # find unique rows -> this fixes the enumeration of the faces!
    nodes4faces = unique(nodes4faces, dims = 1);
    nfaces = size(nodes4faces,1);

    # mapping to get number of new mipoint between two old nodes
    newnode4nodes = @views sparse(nodes4faces[:,1],nodes4faces[:,2],(1:nfaces) .+ nnodes,nnodes,nnodes);
    newnode4nodes = newnode4nodes + newnode4nodes';

    # compute and append face midpoints and cell midpoints
    coords4nodes = @views [coords4nodes;
                           1 // 2 * (coords4nodes[nodes4faces[:,1],:] + coords4nodes[nodes4faces[:,2],:]);
                           1 // 4 * (coords4nodes[nodes4cells[:,1],:] + coords4nodes[nodes4cells[:,2],:] + coords4nodes[nodes4cells[:,3],:] + coords4nodes[nodes4cells[:,4],:])];
    


    # build up new nodes4cells of uniform refinements
    nfaces = size(nodes4faces,1)
    nodes4cells_new = zeros(Int,4*ncells,4);
    newfacenodes = zeros(Int,4);
    for cell = 1 : ncells
        newfacenodes = map(j->(newnode4nodes[nodes4cells[cell,j],nodes4cells[cell,mod(j,4)+1]]),1:4);        
        newcenternode = nnodes + nfaces + cell
        nodes4cells_new[(1:4) .+ (cell-1)*4,:] = 
            [nodes4cells[cell,1] newfacenodes[1] newcenternode newfacenodes[4];
             nodes4cells[cell,2] newfacenodes[2] newcenternode newfacenodes[1];
             nodes4cells[cell,3] newfacenodes[3] newcenternode newfacenodes[2];
             nodes4cells[cell,4] newfacenodes[4] newcenternode newfacenodes[3]]
    end  

    # build up new nodes4bfaces and bregions
    nodes4bfaces_new = zeros(Int,2*size(nodes4bfaces,1),2)
    bregions_new = zeros(Int,2*size(nodes4bfaces,1),1)
    for face = 1 : size(nodes4bfaces,1) 
        newfacenode = newnode4nodes[nodes4bfaces[face,1],nodes4bfaces[face,2]]
        nodes4bfaces_new[(1:2) .+ (face-1)*2,:] = 
            [nodes4bfaces[face,1] newfacenode;
            newfacenode nodes4bfaces[face,2]]
        bregions_new[2*face-1] = bregions[face]
        bregions_new[2*face] = bregions[face]
    end

    return coords4nodes, nodes4cells_new, nodes4bfaces_new, bregions_new;
end




# transformation for H1 elements on a parallelogram
function local2global(grid::Mesh, ::ElemType2DParallelogram)
    A = Matrix{Float64}(undef,2,2)
    b = Vector{Float64}(undef,2)
    x = Vector{Real}(undef,2)
    return function fix_cell(cell)
        b[1] = grid.coords4nodes[grid.nodes4cells[cell,1],1]
        b[2] = grid.coords4nodes[grid.nodes4cells[cell,1],2]
        A[1,1] = grid.coords4nodes[grid.nodes4cells[cell,2],1] - b[1]
        A[1,2] = grid.coords4nodes[grid.nodes4cells[cell,4],1] - b[1]
        A[2,1] = grid.coords4nodes[grid.nodes4cells[cell,2],2] - b[2]
        A[2,2] = grid.coords4nodes[grid.nodes4cells[cell,4],2] - b[2]
        return function closure(xref)
            x[1] = A[1,1]*xref[1] + A[1,2]*xref[2] + b[1]
            x[2] = A[2,1]*xref[1] + A[2,2]*xref[2] + b[2]
            # faster than: x = A*xref + b
            return x
        end
    end    
end

# transformation for H1 elements on a face
function local2global_face(grid::Mesh, ::ElemType2DParallelogram)
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

# exact tinversion of transformation for H1 elements on a parallelogram
function local2global_tinv_jacobian(grid::Mesh, ::ElemType2DParallelogram)
    det = 0.0
    function closure(A!::Array{T,2},cell::Int64) where T <: Real
        # transposed inverse of A
        A![2,2] = grid.coords4nodes[grid.nodes4cells[cell,2],1] - grid.coords4nodes[grid.nodes4cells[cell,1],1]
        A![2,1] = -grid.coords4nodes[grid.nodes4cells[cell,4],1] + grid.coords4nodes[grid.nodes4cells[cell,1],1]
        A![1,2] = -grid.coords4nodes[grid.nodes4cells[cell,2],2] + grid.coords4nodes[grid.nodes4cells[cell,1],2]
        A![1,1] = grid.coords4nodes[grid.nodes4cells[cell,4],2] - grid.coords4nodes[grid.nodes4cells[cell,1],2]
        
        # divide by  determinant
        det = A![1,1]*A![2,2] - A![1,2]*A![2,1]
    
        A![1] = A![1]/det
        A![2] = A![2]/det
        A![3] = A![3]/det
        A![4] = A![4]/det
        return det
    end
end


# Piola transformation for Hdiv elements on a parallelogram
function local2global_Piola(grid::Mesh, ::ElemType2DParallelogram)
    det = 0.0;
    return function closure(A!::Array{T,2},cell::Int64) where T <: Real
        A![1,1] = grid.coords4nodes[grid.nodes4cells[cell,2],1] - grid.coords4nodes[grid.nodes4cells[cell,1],1]
        A![1,2] = grid.coords4nodes[grid.nodes4cells[cell,4],1] - grid.coords4nodes[grid.nodes4cells[cell,1],1]
        A![2,1] = grid.coords4nodes[grid.nodes4cells[cell,2],2] - grid.coords4nodes[grid.nodes4cells[cell,1],2]
        A![2,2] = grid.coords4nodes[grid.nodes4cells[cell,4],2] - grid.coords4nodes[grid.nodes4cells[cell,1],2]
        det = A![1,1]*A![2,2] - A![1,2]*A![2,1]
        return det
    end    
end