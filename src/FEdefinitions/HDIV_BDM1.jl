struct HdivBDM1FiniteElement{T} <: AbstractHdivFiniteElement
    name::String;                 # full name of finite element (used in messages)
    grid::Grid.Mesh{T};           # link to grid
end

function getBDM1FiniteElement(grid)
    ensure_nodes4faces!(grid);
    ensure_faces4cells!(grid);
    ensure_volume4cells!(grid);
    ensure_signs4cells!(grid);
    ensure_normal4faces!(grid);
    T = eltype(grid.coords4nodes);
    return HdivBDM1FiniteElement{T}("BDM1 (HdivFiniteElement)",grid)
end   


function get_xref4dof(FE::HdivBDM1FiniteElement, ::Grid.ElemType2DTriangle) 
    return Array{Float64,2}([0.5 0; 0.5 0.5; 0 0.5; 0 0.5; 0.5 0; 0.5 0.5; 0 0.5; 0 0.5])
end    

# POLYNOMIAL ORDER
get_polynomial_order(FE::HdivBDM1FiniteElement) = 2;

# TOTAL NUMBER OF DOFS
get_ndofs(FE::HdivBDM1FiniteElement) = 2*size(FE.grid.nodes4faces,1);

# NUMBER OF DOFS ON ELEMTYPE
get_ndofs4elemtype(FE::HdivBDM1FiniteElement, ::Grid.Abstract1DElemType) = 2
get_ndofs4elemtype(FE::HdivBDM1FiniteElement, ::Grid.ElemType2DTriangle) = 6

# NUMBER OF COMPONENTS
get_ncomponents(FE::HdivBDM1FiniteElement) = 2

# LOCAL DOF TO GLOBAL DOF ON CELL
function get_dofs_on_cell!(dofs,FE::HdivBDM1FiniteElement, cell::Int64, ::Grid.ElemType2DTriangle)
    dofs[1:3] = FE.grid.faces4cells[cell,:]
    dofs[4:6] = size(FE.grid.nodes4faces,1) .+ FE.grid.faces4cells[cell,:]
end

# LOCAL DOF TO GLOBAL DOF ON FACE
function get_dofs_on_face!(dofs,FE::HdivBDM1FiniteElement, face::Int64, ::Grid.Abstract1DElemType)
    dofs[1] = face
    dofs[2] = size(FE.grid.nodes4faces,1) + face
end

# BASIS FUNCTIONS
function get_basis_on_elemtype(FE::HdivBDM1FiniteElement, ::Grid.ElemType2DTriangle)
    function closure(xref)
        return [[xref[1] xref[2]-1.0];
                [xref[1] xref[2]];
                [xref[1]-1.0 xref[2]];
                [3*xref[1] 3-6*xref[1]-3*xref[2]];
                [-3*xref[1] 3*xref[2]];
                [-3+3*xref[1]+6*xref[2] -3*xref[2]]]
    end
end

function get_basis_fluxes_on_elemtype(FE::HdivBDM1FiniteElement, ::Grid.Abstract1DElemType)
    function closure(xref)
        return [1.0 # normal-flux of RT0 function on single triangle face
                -3+6*xref[1]]; # linear normal-flux of BDM1 function
    end
end       

function get_basis_coefficients_on_cell!(coefficients, FE::HdivBDM1FiniteElement, cell::Int64,  ::Grid.ElemType2DTriangle)
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
end  