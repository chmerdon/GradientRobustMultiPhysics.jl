struct HdivRT1FiniteElement{T} <: AbstractHdivFiniteElement
    name::String;                 # full name of finite element (used in messages)
    grid::Grid.Mesh{T};           # link to grid
end

function getRT1FiniteElement(grid)
    ensure_nodes4faces!(grid);
    ensure_faces4cells!(grid);
    ensure_volume4cells!(grid);
    ensure_signs4cells!(grid);
    ensure_normal4faces!(grid);
    T = eltype(grid.coords4nodes);
    return HdivRT1FiniteElement{T}("RT1 (HdivFiniteElement)",grid)
end   


function get_xref4dof(FE::HdivRT1FiniteElement, ::Grid.ElemType2DTriangle) 
    return Array{Float64,2}([0.5 0; 0.5 0.5; 0 0.5; 0 0.5; 0.5 0; 0.5 0.5; 0 0.5; 0 0.5])
end    

# POLYNOMIAL ORDER
get_polynomial_order(FE::HdivRT1FiniteElement) = 2;

# TOTAL NUMBER OF DOFS
get_ndofs(FE::HdivRT1FiniteElement) = 2*(size(FE.grid.nodes4faces,1) + size(FE.grid.nodes4cells,1));

# NUMBER OF DOFS ON ELEMTYPE
get_ndofs4elemtype(FE::HdivRT1FiniteElement, ::Grid.Abstract1DElemType) = 2
get_ndofs4elemtype(FE::HdivRT1FiniteElement, ::Grid.ElemType2DTriangle) = 8

# NUMBER OF COMPONENTS
get_ncomponents(FE::HdivRT1FiniteElement) = 2

# LOCAL DOF TO GLOBAL DOF ON CELL
function get_dofs_on_cell!(dofs,FE::HdivRT1FiniteElement, cell::Int64, ::Grid.ElemType2DTriangle)
    dofs[1:3] = FE.grid.faces4cells[cell,:]
    dofs[4:6] = size(FE.grid.nodes4faces,1) .+ FE.grid.faces4cells[cell,:]
    dofs[7] = 2*size(FE.grid.nodes4faces,1) + cell
    dofs[8] = dofs[7] + size(FE.grid.nodes4cells,1)
end

# LOCAL DOF TO GLOBAL DOF ON FACE
function get_dofs_on_face!(dofs,FE::HdivRT1FiniteElement, face::Int64, ::Grid.Abstract1DElemType)
    dofs[1] = face
    dofs[2] = size(FE.grid.nodes4faces,1) + face
end

# BASIS FUNCTIONS
function get_basis_on_cell(FE::HdivRT1FiniteElement, ::Grid.ElemType2DTriangle)
    temp = 0.0;
    a = [0.0 0.0];
    b = [0.0 0.0];
    c = [0.0 0.0];
    function closure(xref)
        temp = 0.5 - xref[1] - xref[2]
        a = [xref[1] xref[2]-1.0];
        b = [xref[1] xref[2]];
        c = [xref[1]-1.0 xref[2]];
        return [a;
                b;
                c;
                (12*temp).*a;
                (12*(xref[1] - 0.5)).*b;
                (12*(xref[2] - 0.5)).*c;
                12*xref[2].*a
                12*xref[1].*c]
    end
end

function get_basis_fluxes_on_face(FE::HdivRT1FiniteElement, ::Grid.Abstract1DElemType)
    function closure(xref)
        return [1.0 # normal-flux of RT0 function on single triangle face
                -12*(xref[1]-0.5)]; # linear normal-flux of RT1 function
    end
end       

function get_basis_coefficients_on_cell!(coefficients, FE::HdivRT1FiniteElement, cell::Int64,  ::Grid.ElemType2DTriangle)
    # multiply by signs to ensure continuity of normal fluxes
    coefficients[1,1] = FE.grid.signs4cells[cell,1];
    coefficients[1,2] = FE.grid.signs4cells[cell,1];
    coefficients[2,1] = FE.grid.signs4cells[cell,2];
    coefficients[2,2] = FE.grid.signs4cells[cell,2];
    coefficients[3,1] = FE.grid.signs4cells[cell,3];
    coefficients[3,2] = FE.grid.signs4cells[cell,3];
    coefficients[4,1] = 1;
    coefficients[4,2] = 1;
    coefficients[5,1] = 1;
    coefficients[5,2] = 1;
    coefficients[6,1] = 1;
    coefficients[6,2] = 1;
    coefficients[7,1] = 1;
    coefficients[7,2] = 1;
    coefficients[8,1] = 1;
    coefficients[8,2] = 1;
end  