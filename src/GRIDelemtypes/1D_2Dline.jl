struct ElemType1D2DLine <: Abstract1DElemType end

function get_reference_cordinates(::ElemType1D2DLine)
    return [0, 1];
end

function get_face_elemtype(::ElemType1D2DLine)
    return ElemType0DPoint();
end

# perform a uniform (red) refinement of 1D intervals
function uniform_refinement(::ElemType1D2DLine,coords4nodes::Array,nodes4cells::Array)
    
    nnodes = size(coords4nodes,1);
    ncells = size(nodes4cells,1);
    coords4nodes = @views [coords4nodes; 1 // 2 * (coords4nodes[nodes4cells[:,1],:] + coords4nodes[nodes4cells[:,2],:])];
      
    nodes4cells_new = zeros(Int,2*ncells,2);
    for cell = 1 : ncells
        nodes4cells_new[(1:2) .+ (cell-1)*2,:] = [nodes4cells[cell,1] nnodes+cell;
                                                  nnodes+cell nodes4cells[cell,2]];
    end
    return coords4nodes, nodes4cells_new;
end

# transformation for H1 elements on a line
function local2global(grid::Mesh, ::ElemType1D2DLine)
    A = Matrix{Float64}(undef,2,1)
    b = Vector{Float64}(undef,2)
    x = Vector{Real}(undef,2)
    return function fix_cell(cell::Int64)
        b[1] = grid.coords4nodes[grid.nodes4cells[cell,1],1]
        b[2] = grid.coords4nodes[grid.nodes4cells[cell,1],2]
        A[1,1] = grid.coords4nodes[grid.nodes4cells[cell,2],1] - b[1]
        A[2,1] = grid.coords4nodes[grid.nodes4cells[cell,2],2] - b[2]
        return function closure(xref)
            x[1] = A[1,1]*xref[1] + b[1]
            x[2] = A[2,1]*xref[1] + b[2]
            return x
        end
    end    
end

# exact tinversion of transformation for H1 elements on a line
function local2global_tinv_jacobian(grid::Mesh, ::ElemType1D2DLine)
    det = 0.0
    function closure(A!::Array{T,2},cell::Int64) where T <: Real
    end
end