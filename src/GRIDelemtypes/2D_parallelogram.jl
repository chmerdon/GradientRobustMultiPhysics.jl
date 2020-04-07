struct ElemType2DParallelogram <: Abstract2DQuadrilateral end

# Note that the parallelogram is a special kind of quadrilateral that allows
# for a linear transformation map (actually the same as the triangle up to renumbering)
# therefore only the functions for the mapping are redefined

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