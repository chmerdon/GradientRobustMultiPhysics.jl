struct H1P1FiniteElement{T, dim, ncomponents} <: AbstractH1FiniteElement
    name::String;                 # full name of finite element (used in messages)
    grid::Grid.Mesh{T};           # link to grid
    xref4dofs4cell::Array{T,2};   # coordinates for degrees of freedom in reference domain
end

function getP1FiniteElement(grid,ncomponents)
    ensure_nodes4faces!(grid);
    ensure_volume4cells!(grid);
    T = eltype(grid.coords4nodes);
    dim = size(grid.nodes4cells,2) - 1;
    if dim == 1
        xref4dofs4cell = repeat(Array{Float64,2}([0,1]')',ncomponents);
    elseif dim == 2    
        xref4dofs4cell = repeat([0 0; 1 0; 0 1],ncomponents);
    end    
    return H1P1FiniteElement{T,dim,ncomponents}("P1 (H1FiniteElement, dim=$dim, ncomponents=$ncomponents)",grid, Array{T,2}(xref4dofs4cell))
end 

get_xref4dof(FE::H1P1FiniteElement, k) = xref4dofs4cell[k,:]

# POLYNOMIAL ORDER
get_polynomial_order(FE::H1P1FiniteElement) = 1;

# TOTAL NUMBER OF DOFS
get_ndofs(FE::H1P1FiniteElement{T,1,1} where T <: Real) = size(FE.grid.coords4nodes,1);
get_ndofs(FE::H1P1FiniteElement{T,2,1} where T <: Real) = size(FE.grid.coords4nodes,1);
get_ndofs(FE::H1P1FiniteElement{T,2,2} where T <: Real) = 2*size(FE.grid.coords4nodes,1);

# MAXIMAL DOFS ON CELL
get_maxndofs4cell(FE::H1P1FiniteElement{T,1,1} where T <: Real) = 2
get_maxndofs4cell(FE::H1P1FiniteElement{T,2,1} where T <: Real) = 3
get_maxndofs4cell(FE::H1P1FiniteElement{T,2,2} where T <: Real) = 6

# MAXIMAL DOFS ON FACE
get_maxndofs4face(FE::H1P1FiniteElement{T,1,1} where T <: Real) = 1
get_maxndofs4face(FE::H1P1FiniteElement{T,2,1} where T <: Real) = 2
get_maxndofs4face(FE::H1P1FiniteElement{T,2,2} where T <: Real) = 4

# NUMBER OF COMPONENTS
get_ncomponents(FE::H1P1FiniteElement{T,1,1} where T <: Real) = 1
get_ncomponents(FE::H1P1FiniteElement{T,2,1} where T <: Real) = 1
get_ncomponents(FE::H1P1FiniteElement{T,2,2} where T <: Real) = 2

# LOCAL DOF TO GLOBAL DOF ON CELL
get_globaldof4cell(FE::H1P1FiniteElement{T,1,1} where T <: Real, cell, ::Val{1}) = FE.grid.nodes4cells[cell,1]
get_globaldof4cell(FE::H1P1FiniteElement{T,1,1} where T <: Real, cell, ::Val{2}) = FE.grid.nodes4cells[cell,2]

get_globaldof4cell(FE::H1P1FiniteElement{T,2,1} where T <: Real, cell, ::Val{1}) = FE.grid.nodes4cells[cell,1]
get_globaldof4cell(FE::H1P1FiniteElement{T,2,1} where T <: Real, cell, ::Val{2}) = FE.grid.nodes4cells[cell,2]
get_globaldof4cell(FE::H1P1FiniteElement{T,2,1} where T <: Real, cell, ::Val{3}) = FE.grid.nodes4cells[cell,3]

get_globaldof4cell(FE::H1P1FiniteElement{T,2,2} where T <: Real, cell, ::Val{1}) = FE.grid.nodes4cells[cell,1]
get_globaldof4cell(FE::H1P1FiniteElement{T,2,2} where T <: Real, cell, ::Val{2}) = FE.grid.nodes4cells[cell,2]
get_globaldof4cell(FE::H1P1FiniteElement{T,2,2} where T <: Real, cell, ::Val{3}) = FE.grid.nodes4cells[cell,3]
get_globaldof4cell(FE::H1P1FiniteElement{T,2,2} where T <: Real, cell, ::Val{4}) = size(FE.grid.coords4nodes,1) + FE.grid.nodes4cells[cell,1]
get_globaldof4cell(FE::H1P1FiniteElement{T,2,2} where T <: Real, cell, ::Val{5}) = size(FE.grid.coords4nodes,1) + FE.grid.nodes4cells[cell,2]
get_globaldof4cell(FE::H1P1FiniteElement{T,2,2} where T <: Real, cell, ::Val{6}) = size(FE.grid.coords4nodes,1) +FE.grid.nodes4cells[cell,3]

# LOCAL DOF TO GLOBAL DOF ON FACE
get_globaldof4face(FE::H1P1FiniteElement{T,1,1} where T <: Real, face, ::Val{1}) = face

get_globaldof4face(FE::H1P1FiniteElement{T,2,1} where T <: Real, face, ::Val{1}) = FE.grid.nodes4faces[face,1]
get_globaldof4face(FE::H1P1FiniteElement{T,2,1} where T <: Real, face, ::Val{2}) = FE.grid.nodes4faces[face,2]

get_globaldof4face(FE::H1P1FiniteElement{T,2,2} where T <: Real, face, ::Val{1}) = FE.grid.nodes4faces[face,1]
get_globaldof4face(FE::H1P1FiniteElement{T,2,2} where T <: Real, face, ::Val{2}) = FE.grid.nodes4faces[face,2]
get_globaldof4face(FE::H1P1FiniteElement{T,2,2} where T <: Real, face, ::Val{3}) = size(FE.grid.coords4nodes,1) + FE.grid.nodes4faces[face,1]
get_globaldof4face(FE::H1P1FiniteElement{T,2,2} where T <: Real, face, ::Val{4}) = size(FE.grid.coords4nodes,1) + FE.grid.nodes4faces[face,2]


# BASIS FUNCTIONS
function get_all_basis_functions_on_cell(FE::H1P1FiniteElement{T,1,1} where T <: Real)
    function closure(xref)
        return [1 - xref[1],
                xref[1]]
    end
end

function get_all_basis_functions_on_face(FE::H1P1FiniteElement{T,1,1} where T <: Real)
    function closure(xref)
        return [1.0]
    end
end

function get_all_basis_functions_on_cell(FE::H1P1FiniteElement{T,2,1} where T <: Real)
    function closure(xref)
        return [1 - xref[1] - xref[2],
                xref[1],
                xref[2]]
    end
end


function get_all_basis_functions_on_face(FE::H1P1FiniteElement{T,2,1} where T <: Real)
    function closure(xref)
        return [1 - xref[1],
                xref[1]]
    end
end

function get_all_basis_functions_on_cell(FE::H1P1FiniteElement{T,2,2} where T <: Real)
    temp = 0.0;
    function closure(xref)
        temp = 1 - xref[1] - xref[2];
        return [temp 0.0;
                xref[1] 0.0;
                xref[2] 0.0;
                0.0 temp;
                0.0 xref[1];
                0.0 xref[2]]
    end
end


function get_all_basis_functions_on_face(FE::H1P1FiniteElement{T,2,2} where T <: Real)
    temp = 0.0;
    function closure(xref)
        temp = 1 - xref[1];
        return [temp 0.0;
                xref[1] 0.0;
                0.0 temp;
                0.0 xref[1]]
    end
end
