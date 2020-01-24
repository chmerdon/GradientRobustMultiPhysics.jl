struct HdivRT0FiniteElement{T, dim} <: AbstractHdivRTFiniteElement
    name::String;                 # full name of finite element (used in messages)
    grid::Grid.Mesh{T};           # link to grid
    xref4dofs4cell::Array{T,2};   # coordinates for degrees of freedom in reference domain
end

function getRT0FiniteElement(grid)
    ensure_nodes4faces!(grid);
    ensure_faces4cells!(grid);
    ensure_volume4cells!(grid);
    ensure_signs4cells!(grid);
    ensure_normal4faces!(grid);
    T = eltype(grid.coords4nodes);
    dim = size(grid.nodes4cells,2) - 1;
    @assert dim == 2
    xref4dofs4cell = repeat([0.5 0; 0.5 0.5; 0 0.5],dim);
    return HdivRT0FiniteElement{T,dim}("RT0 (HdivFiniteElement, dim=$dim)",grid, Array{T,2}(xref4dofs4cell))
end   


get_xref4dof(FE::HdivRT0FiniteElement, k) = xref4dofs4cell[k,:]

# POLYNOMIAL ORDER
get_polynomial_order(FE::HdivRT0FiniteElement) = 1;

# TOTAL NUMBER OF DOFS
get_ndofs(FE::HdivRT0FiniteElement{T,2} where T <: Real) = size(FE.grid.nodes4faces,1);

# MAXIMAL DOFS ON CELL
get_maxndofs4cell(FE::HdivRT0FiniteElement{T,2} where T <: Real) = 3

# MAXIMAL DOFS ON FACE
get_maxndofs4face(FE::HdivRT0FiniteElement{T,2} where T <: Real) = 1

# NUMBER OF COMPONENTS
get_ncomponents(FE::HdivRT0FiniteElement{T,2} where T <: Real) = 2

# LOCAL DOF TO GLOBAL DOF ON CELL
get_globaldof4cell(FE::HdivRT0FiniteElement{T,2} where T <: Real, cell, ::Val{1}) = FE.grid.faces4cells[cell,1]
get_globaldof4cell(FE::HdivRT0FiniteElement{T,2} where T <: Real, cell, ::Val{2}) = FE.grid.faces4cells[cell,2]
get_globaldof4cell(FE::HdivRT0FiniteElement{T,2} where T <: Real, cell, ::Val{3}) = FE.grid.faces4cells[cell,3]

# LOCAL DOF TO GLOBAL DOF ON FACE
get_globaldof4face(FE::HdivRT0FiniteElement{T,2} where T <: Real, face, ::Val{1}) = face

# BASIS FUNCTIONS
function get_all_basis_functions_on_cell(FE::HdivRT0FiniteElement{T,2} where T <: Real)
    function closure(xref)
        return [xref[1] xref[2]-1.0;
                xref[1] xref[2];
                xref[1]-1.0 xref[2]]
    end
end

function get_all_basis_function_fluxes_on_face(FE::HdivRT0FiniteElement{T,2} where T <: Real)
    function closure(xref)
        return [1.0]; # normal-flux of RT0 function on face
    end
end       

function set_basis_coefficients_on_cell!(coefficients, FE::HdivRT0FiniteElement{T,2} where T <: Real, cell::Int64)
    # multiply by signs to ensure continuity of normal fluxes
    coefficients[1,1] = FE.grid.signs4cells[cell,1];
    coefficients[1,2] = FE.grid.signs4cells[cell,1];
    coefficients[2,1] = FE.grid.signs4cells[cell,2];
    coefficients[2,2] = FE.grid.signs4cells[cell,2];
    coefficients[3,1] = FE.grid.signs4cells[cell,3];
    coefficients[3,2] = FE.grid.signs4cells[cell,3];
end   