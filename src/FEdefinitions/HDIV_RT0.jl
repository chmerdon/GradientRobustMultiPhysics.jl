struct HdivRT0FiniteElement{T} <: AbstractHdivFiniteElement
    name::String;                 # full name of finite element (used in messages)
    grid::Grid.Mesh{T};           # link to grid
end

function getRT0FiniteElement(grid)
    ensure_nodes4faces!(grid);
    ensure_faces4cells!(grid);
    ensure_volume4cells!(grid);
    ensure_signs4cells!(grid);
    ensure_normal4faces!(grid);
    T = eltype(grid.coords4nodes);
    return HdivRT0FiniteElement{T}("RT0 (HdivFiniteElement)",grid)
end   


function get_xref4dof(FE::HdivRT0FiniteElement, ::Grid.ElemType2DTriangle) 
    return Array{Float64,2}([0.5 0; 0.5 0.5; 0 0.5])
end    

# POLYNOMIAL ORDER
get_polynomial_order(FE::HdivRT0FiniteElement) = 1;

# TOTAL NUMBER OF DOFS
get_ndofs(FE::HdivRT0FiniteElement) = size(FE.grid.nodes4faces,1);

# NUMBER OF DOFS ON ELEMTYPE
get_ndofs4elemtype(FE::HdivRT0FiniteElement, ::Grid.Abstract1DElemType) = 1
get_ndofs4elemtype(FE::HdivRT0FiniteElement, ::Grid.ElemType2DTriangle) = 3

# NUMBER OF COMPONENTS
get_ncomponents(FE::HdivRT0FiniteElement) = 2

# LOCAL DOF TO GLOBAL DOF ON CELL
function get_dofs_on_cell!(dofs,FE::HdivRT0FiniteElement, cell::Int64, ::Grid.ElemType2DTriangle)
    dofs[:] = FE.grid.faces4cells[cell,:]
end

# LOCAL DOF TO GLOBAL DOF ON FACE
function get_dofs_on_face!(dofs,FE::HdivRT0FiniteElement, face::Int64, ::Grid.Abstract1DElemType)
    dofs[1] = face
end

# BASIS FUNCTIONS
function get_basis_on_cell(FE::HdivRT0FiniteElement, ::Grid.ElemType2DTriangle)
    function closure(xref)
        return [xref[1] xref[2]-1.0;
                xref[1] xref[2];
                xref[1]-1.0 xref[2]]
    end
end

function get_basis_fluxes_on_face(FE::HdivRT0FiniteElement, ::Grid.Abstract1DElemType)
    function closure(xref)
        return [1.0]; # normal-flux of RT0 function on single triangle face
    end
end       

function get_basis_coefficients_on_cell!(coefficients, FE::HdivRT0FiniteElement, cell::Int64,  ::Grid.ElemType2DTriangle)
    # multiply by signs to ensure continuity of normal fluxes
    coefficients[1,1] = FE.grid.signs4cells[cell,1];
    coefficients[1,2] = FE.grid.signs4cells[cell,1];
    coefficients[2,1] = FE.grid.signs4cells[cell,2];
    coefficients[2,2] = FE.grid.signs4cells[cell,2];
    coefficients[3,1] = FE.grid.signs4cells[cell,3];
    coefficients[3,2] = FE.grid.signs4cells[cell,3];
end   