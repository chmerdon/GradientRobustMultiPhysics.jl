struct H1BRFiniteElement{T, dim, ncomponents} <: AbstractH1FiniteElement
    name::String;                 # full name of finite element (used in messages)
    grid::Grid.Mesh{T};           # link to grid
    xref4dofs4cell::Array{T,2};   # coordinates for degrees of freedom in reference domain
end

function getBRFiniteElement(grid,ncomponents)
    ensure_nodes4faces!(grid);
    ensure_faces4cells!(grid);
    ensure_volume4cells!(grid);
    ensure_normal4faces!(grid);
    T = eltype(grid.coords4nodes);
    dim = size(grid.nodes4cells,2) - 1;
    @assert dim == 2
    @assert ncomponents == 2
    xref4dofs4cell = repeat([0 0; 1 0; 0 1; 0.5 0; 0.5 0.5; 0 0.5],ncomponents); 
    return H1BRFiniteElement{T,dim,ncomponents}("BR (H1FiniteElement, dim=$dim, ncomponents=$ncomponents)",grid, Array{T,2}(xref4dofs4cell))
end 

get_xref4dof(FE::H1BRFiniteElement, k) = xref4dofs4cell[k,:]

# POLYNOMIAL ORDER
get_polynomial_order(FE::H1BRFiniteElement) = 2;

# TOTAL NUMBER OF DOFS
get_ndofs(FE::H1BRFiniteElement{T,2,2} where T <: Real) = 2*size(FE.grid.coords4nodes,1) + size(FE.grid.nodes4faces,1);

# MAXIMAL DOFS ON CELL
get_maxndofs4cell(FE::H1BRFiniteElement{T,2,2} where T <: Real) = 9

# MAXIMAL DOFS ON FACE
get_maxndofs4face(FE::H1BRFiniteElement{T,2,2} where T <: Real) = 5

# NUMBER OF COMPONENTS
get_ncomponents(FE::H1BRFiniteElement{T,2,2} where T <: Real) = 2

# LOCAL DOF TO GLOBAL DOF ON CELL
get_globaldof4cell(FE::H1BRFiniteElement{T,2,2} where T <: Real, cell, ::Val{1}) = FE.grid.nodes4cells[cell,1]
get_globaldof4cell(FE::H1BRFiniteElement{T,2,2} where T <: Real, cell, ::Val{2}) = FE.grid.nodes4cells[cell,2]
get_globaldof4cell(FE::H1BRFiniteElement{T,2,2} where T <: Real, cell, ::Val{3}) = FE.grid.nodes4cells[cell,3]
get_globaldof4cell(FE::H1BRFiniteElement{T,2,2} where T <: Real, cell, ::Val{4}) = size(FE.grid.coords4nodes,1) + FE.grid.nodes4cells[cell,1]
get_globaldof4cell(FE::H1BRFiniteElement{T,2,2} where T <: Real, cell, ::Val{5}) = size(FE.grid.coords4nodes,1) + FE.grid.nodes4cells[cell,2]
get_globaldof4cell(FE::H1BRFiniteElement{T,2,2} where T <: Real, cell, ::Val{6}) = size(FE.grid.coords4nodes,1) +FE.grid.nodes4cells[cell,3]
get_globaldof4cell(FE::H1BRFiniteElement{T,2,2} where T <: Real, cell, ::Val{7}) = 2*size(FE.grid.coords4nodes,1) +FE.grid.faces4cells[cell,1]
get_globaldof4cell(FE::H1BRFiniteElement{T,2,2} where T <: Real, cell, ::Val{8}) = 2*size(FE.grid.coords4nodes,1) +FE.grid.faces4cells[cell,2]
get_globaldof4cell(FE::H1BRFiniteElement{T,2,2} where T <: Real, cell, ::Val{9}) = 2*size(FE.grid.coords4nodes,1) +FE.grid.faces4cells[cell,3]

# LOCAL DOF TO GLOBAL DOF ON FACE
get_globaldof4face(FE::H1BRFiniteElement{T,2,2} where T <: Real, face, ::Val{1}) = FE.grid.nodes4faces[face,1]
get_globaldof4face(FE::H1BRFiniteElement{T,2,2} where T <: Real, face, ::Val{2}) = FE.grid.nodes4faces[face,2]
get_globaldof4face(FE::H1BRFiniteElement{T,2,2} where T <: Real, face, ::Val{3}) = size(FE.grid.coords4nodes,1) + FE.grid.nodes4faces[face,1]
get_globaldof4face(FE::H1BRFiniteElement{T,2,2} where T <: Real, face, ::Val{4}) = size(FE.grid.coords4nodes,1) + FE.grid.nodes4faces[face,2]
get_globaldof4face(FE::H1BRFiniteElement{T,2,2} where T <: Real, face, ::Val{5}) = 2*size(FE.grid.coords4nodes,1) + face


# BASIS FUNCTIONS
function get_all_basis_functions_on_cell(FE::H1BRFiniteElement{T,2,2} where T <: Real)
    temp = 0.0;
    fb1 = 0.0;
    fb2 = 0.0;
    fb3 = 0.0;
    function closure(xref)
        temp = 1 - xref[1] - xref[2];
        fb1 = 4*temp*xref[1];
        fb2 = 4*xref[1]*xref[2]
        fb3 = 4*temp*xref[2];
        return [temp 0.0;
                xref[1] 0.0;
                xref[2] 0.0;
                0.0 temp;
                0.0 xref[1];
                0.0 xref[2];
                fb1 fb1;
                fb2 fb2;
                fb3 fb3]
    end
end

function set_basis_coefficients_on_cell!(coefficients, FE::H1BRFiniteElement{T,2,2} where T <: Real, cell::Int64)
    # multiplication with normal vectors
    fill!(coefficients,1.0)
    coefficients[7,1] = FE.grid.normal4faces[FE.grid.faces4cells[cell,1],1];
    coefficients[7,2] = FE.grid.normal4faces[FE.grid.faces4cells[cell,1],2];
    coefficients[8,1] = FE.grid.normal4faces[FE.grid.faces4cells[cell,2],1];
    coefficients[8,2] = FE.grid.normal4faces[FE.grid.faces4cells[cell,2],2];
    coefficients[9,1] = FE.grid.normal4faces[FE.grid.faces4cells[cell,3],1];
    coefficients[9,2] = FE.grid.normal4faces[FE.grid.faces4cells[cell,3],2];
end    

function get_all_basis_functions_on_face(FE::H1BRFiniteElement{T,2,2} where T <: Real)
    temp = 0.0;
    fb = 0.0;
    function closure(xref)
        temp = 1 - xref[1];
        fb = 4*temp*xref[1];
        return [temp 0.0;
                xref[1] 0.0;
                0.0 temp;
                0.0 xref[1];
                fb fb]
    end
end

function set_basis_coefficients_on_face!(coefficients, FE::H1BRFiniteElement{T,2,2} where T <: Real, face::Int64)
    # multiplication with normal vectors
    fill!(coefficients,1.0)
    coefficients[5,1] = FE.grid.normal4faces[face,1];
    coefficients[5,2] = FE.grid.normal4faces[face,2];
end    

function Hdivreconstruction_available(FE::H1BRFiniteElement{T,2,2} where T <: Real)
    return true
end

function get_Hdivreconstruction_space(FE::H1BRFiniteElement{T,2,2} where T <: Real)
    return getRT0FiniteElement(FE.grid)
end

function get_Hdivreconstruction_trafo!(T,FE)
    ensure_length4faces!(FE.grid);
    nfaces = size(FE.grid.nodes4faces,1)
    nnodes = size(FE.grid.coords4nodes,1)
    for face = 1 : nfaces
        # reconstruction coefficients for P1 basis functions
        for k = 1 : 2
            node = FE.grid.nodes4faces[face,k]
            T[node,face] = 1 // 2 * FE.grid.length4faces[face] * FE.grid.normal4faces[face,1]
            T[nnodes+node,face] = 1 // 2 * FE.grid.length4faces[face] * FE.grid.normal4faces[face,2]
        end
        # reconstruction coefficient for quadratic face bubbles
        T[2*nnodes+face,face] = 2 // 3 * FE.grid.length4faces[face]
    end
    return T
end
