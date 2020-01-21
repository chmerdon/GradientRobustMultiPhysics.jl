struct H1CRFiniteElement{T, dim, ncomponents} <: AbstractH1FiniteElement
    name::String;                 # full name of finite element (used in messages)
    grid::Grid.Mesh{T};           # link to grid
    xref4dofs4cell::Array{T,2};   # coordinates for degrees of freedom in reference domain
end

function getCRFiniteElement(grid,ncomponents)
    ensure_faces4cells!(grid);
    ensure_nodes4faces!(grid);
    ensure_volume4cells!(grid);
    T = eltype(grid.coords4nodes);
    dim = size(grid.nodes4cells,2) - 1;
    @assert dim == 2
    xref4dofs4cell = repeat([0.5 0; 0.5 0.5; 0 0.5],ncomponents);
    return H1CRFiniteElement{T,dim,ncomponents}("CR (H1FiniteElement, dim=$dim, ncomponents=$ncomponents)",grid, Array{T,2}(xref4dofs4cell))
end   


get_xref4dof(FE::H1CRFiniteElement, k) = xref4dofs4cell[k,:]

# POLYNOMIAL ORDER
get_polynomial_order(FE::H1CRFiniteElement) = 1;

# TOTAL NUMBER OF DOFS
get_ndofs(FE::H1CRFiniteElement{T,2,1} where T <: Real) = size(FE.grid.nodes4faces,1);
get_ndofs(FE::H1CRFiniteElement{T,2,2} where T <: Real) = 2*size(FE.grid.nodes4faces,1);

# MAXIMAL DOFS ON CELL
get_maxndofs4cell(FE::H1CRFiniteElement{T,2,1} where T <: Real) = 3
get_maxndofs4cell(FE::H1CRFiniteElement{T,2,2} where T <: Real) = 6

# MAXIMAL DOFS ON FACE
get_maxndofs4face(FE::H1CRFiniteElement{T,2,1} where T <: Real) = 1
get_maxndofs4face(FE::H1CRFiniteElement{T,2,2} where T <: Real) = 2

# NUMBER OF COMPONENTS
get_ncomponents(FE::H1CRFiniteElement{T,2,1} where T <: Real) = 1
get_ncomponents(FE::H1CRFiniteElement{T,2,2} where T <: Real) = 2

# LOCAL DOF TO GLOBAL DOF ON CELL
get_globaldof4cell(FE::H1CRFiniteElement{T,2,1} where T <: Real, cell, ::Val{1}) = FE.grid.faces4cells[cell,1]
get_globaldof4cell(FE::H1CRFiniteElement{T,2,1} where T <: Real, cell, ::Val{2}) = FE.grid.faces4cells[cell,2]
get_globaldof4cell(FE::H1CRFiniteElement{T,2,1} where T <: Real, cell, ::Val{3}) = FE.grid.faces4cells[cell,3]

get_globaldof4cell(FE::H1CRFiniteElement{T,2,2} where T <: Real, cell, ::Val{1}) = FE.grid.faces4cells[cell,1]
get_globaldof4cell(FE::H1CRFiniteElement{T,2,2} where T <: Real, cell, ::Val{2}) = FE.grid.faces4cells[cell,2]
get_globaldof4cell(FE::H1CRFiniteElement{T,2,2} where T <: Real, cell, ::Val{3}) = FE.grid.faces4cells[cell,3]
get_globaldof4cell(FE::H1CRFiniteElement{T,2,2} where T <: Real, cell, ::Val{4}) = size(FE.grid.nodes4faces,1) + FE.grid.faces4cells[cell,1]
get_globaldof4cell(FE::H1CRFiniteElement{T,2,2} where T <: Real, cell, ::Val{5}) = size(FE.grid.nodes4faces,1) + FE.grid.faces4cells[cell,2]
get_globaldof4cell(FE::H1CRFiniteElement{T,2,2} where T <: Real, cell, ::Val{6}) = size(FE.grid.nodes4faces,1) + FE.grid.faces4cells[cell,3]

# LOCAL DOF TO GLOBAL DOF ON FACE
get_globaldof4face(FE::H1CRFiniteElement{T,2,1} where T <: Real, face, ::Val{1}) = face

get_globaldof4face(FE::H1CRFiniteElement{T,2,2} where T <: Real, face, ::Val{1}) = face
get_globaldof4face(FE::H1CRFiniteElement{T,2,2} where T <: Real, face, ::Val{2}) = size(FE.grid.nodes4faces,1) + face


# BASIS FUNCTIONS
function get_all_basis_functions_on_cell(FE::H1CRFiniteElement{T,2,1} where T <: Real)
    function closure(xref)
        return [1 - 2*xref[2],
                2*(xref[1]+xref[2]) - 1,
                1 - 2*xref[1]]
    end
end


function get_all_basis_functions_on_face(FE::H1CRFiniteElement{T,2,1} where T <: Real)
    function closure(xref)
        return [1.0]
    end
end

function get_all_basis_functions_on_cell(FE::H1CRFiniteElement{T,2,2} where T <: Real)
    temp = 0.0;
    temp2 = 0.0;
    temp3 = 0.0;
    function closure(xref)
        temp = 1 - 2*xref[2]
        temp2 = 2*(xref[1]+xref[2]) - 1
        temp3 = 1 - 2*xref[1]
        return [temp 0.0;
                temp2 0.0;
                temp3 0.0;
                0.0 temp;
                0.0 temp2;
                0.0 temp3]
    end
end


function get_all_basis_functions_on_face(FE::H1CRFiniteElement{T,2,2} where T <: Real)
    function closure(xref)
        return [1.0 0.0;
                0.0 1.0]
    end
end
