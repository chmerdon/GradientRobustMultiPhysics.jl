struct HdivABF0FiniteElement{T} <: AbstractHdivFiniteElement
    name::String;                 # full name of finite element (used in messages)
    grid::Grid.Mesh{T};           # link to grid
end

function getABF0FiniteElement(grid)
    ensure_nodes4faces!(grid);
    ensure_faces4cells!(grid);
    ensure_volume4cells!(grid);
    ensure_signs4cells!(grid);
    ensure_normal4faces!(grid);
    T = eltype(grid.coords4nodes);
    return HdivABF0FiniteElement{T}("ABF0 (HdivFiniteElement)",grid)
end   


function get_xref4dof(FE::HdivABF0FiniteElement, ::Grid.Abstract2DQuadrilateral) 
    return Array{Float64,2}([0.5 0; 0.5 0.5; 0 0.5; 0 0.5; 0.5 0; 0.5 0.5; 0 0.5; 0 0.5])
end    

# POLYNOMIAL ORDER
get_polynomial_order(FE::HdivABF0FiniteElement) = 2;

# TOTAL NUMBER OF DOFS
get_ndofs(FE::HdivABF0FiniteElement) = size(FE.grid.nodes4faces,1) + 2 * size(FE.grid.nodes4cells,1);

# NUMBER OF DOFS ON ELEMTYPE
get_ndofs4elemtype(FE::HdivABF0FiniteElement, ::Grid.Abstract1DElemType) = 1
get_ndofs4elemtype(FE::HdivABF0FiniteElement, ::Grid.Abstract2DQuadrilateral) = 6

# NUMBER OF COMPONENTS
get_ncomponents(FE::HdivABF0FiniteElement) = 2

# LOCAL DOF TO GLOBAL DOF ON CELL
function get_dofs_on_cell!(dofs,FE::HdivABF0FiniteElement, cell::Int64, ::Grid.Abstract2DQuadrilateral)
    dofs[1:4] = FE.grid.faces4cells[cell,:]
    dofs[5] = size(FE.grid.nodes4faces,1) + 2*cell - 1
    dofs[6] = size(FE.grid.nodes4faces,1) + 2*cell
end

# LOCAL DOF TO GLOBAL DOF ON FACE
function get_dofs_on_face!(dofs,FE::HdivABF0FiniteElement, face::Int64, ::Grid.Abstract1DElemType)
    dofs[1] = face
end

# BASIS FUNCTIONS
function get_basis_on_cell(FE::HdivABF0FiniteElement, ::Grid.Abstract2DQuadrilateral)
    a = 0.0
    b = 0.0
    d = -1.0/3.0
    function closure(xref)
        a = xref[1] - 1.0
        b = xref[2] - 1.0
        return [[-3*xref[1]*a -3*b*(xref[2]+d)];
                [-3*xref[1]*(a+d) -3*xref[2]*b];
                [-3*xref[1]*a -3*xref[2]*(b+d)];
                [-3*a*(xref[1]+d) -3*xref[2]*b];
                [6*xref[1]*a 0.0];
                [0.0 6*xref[2]*b]]
    end
end

function get_basis_fluxes_on_face(FE::HdivABF0FiniteElement, ::Grid.Abstract1DElemType)
    function closure(xref)
        return [1.0] # constant normal-flux of ABF0 function on single triangle face
    end
end       

function get_basis_coefficients_on_cell!(coefficients, FE::HdivABF0FiniteElement, cell::Int64,  ::Grid.Abstract2DQuadrilateral)
    # multiply by signs to ensure continuity of normal fluxes
    coefficients[1,1] = FE.grid.signs4cells[cell,1];
    coefficients[1,2] = FE.grid.signs4cells[cell,1];
    coefficients[2,1] = FE.grid.signs4cells[cell,2];
    coefficients[2,2] = FE.grid.signs4cells[cell,2];
    coefficients[3,1] = FE.grid.signs4cells[cell,3];
    coefficients[3,2] = FE.grid.signs4cells[cell,3];
    coefficients[4,1] = FE.grid.signs4cells[cell,4];
    coefficients[4,2] = FE.grid.signs4cells[cell,4];
    coefficients[5,1] = 1;
    coefficients[5,2] = 1;
    coefficients[6,1] = 1;
    coefficients[6,2] = 1;
end  