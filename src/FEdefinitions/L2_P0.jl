### BEWARE: P0 element currently enlisted as an H1FiniteElement, although not H1 conforming
###         this should be changed to AbstractL2FiniteElement later, once this case is handled in assembly loops

struct L2P0FiniteElement{T, dim, ncomponents} <: AbstractH1FiniteElement
    name::String;                 # full name of finite element (used in messages)
    grid::Grid.Mesh{T};           # link to grid
    xref4dofs4cell::Array{T,2};   # coordinates for degrees of freedom in reference domain
end

function getP0FiniteElement(grid,ncomponents)
    ensure_nodes4faces!(grid);
    ensure_volume4cells!(grid);
    T = eltype(grid.coords4nodes);
    dim = size(grid.nodes4cells,2) - 1;
    if dim == 1
        xref4dofs4cell = repeat(Array{Float64,2}([1//2]')',ncomponents);
    elseif dim == 2    
        xref4dofs4cell = repeat([1//3 1//3],ncomponents);
    end    
    return L2P0FiniteElement{T,dim,ncomponents}("P0 (H1FiniteElement, dim=$dim, ncomponents=$ncomponents)",grid, Array{T,2}(xref4dofs4cell))
end 

get_xref4dof(FE::L2P0FiniteElement, k) = xref4dofs4cell[k,:]

# POLYNOMIAL ORDER
get_polynomial_order(FE::L2P0FiniteElement) = 0;

# TOTAL NUMBER OF DOFS
get_ndofs(FE::L2P0FiniteElement{T,1,1} where T <: Real) = size(FE.grid.nodes4cells,1);
get_ndofs(FE::L2P0FiniteElement{T,2,1} where T <: Real) = size(FE.grid.nodes4cells,1);
get_ndofs(FE::L2P0FiniteElement{T,2,2} where T <: Real) = 2*size(FE.grid.nodes4cells,1);

# MAXIMAL DOFS ON CELL
get_maxndofs4cell(FE::L2P0FiniteElement{T,1,1} where T <: Real) = 1
get_maxndofs4cell(FE::L2P0FiniteElement{T,2,1} where T <: Real) = 1
get_maxndofs4cell(FE::L2P0FiniteElement{T,2,2} where T <: Real) = 2

# MAXIMAL DOFS ON FACE
get_maxndofs4face(FE::L2P0FiniteElement) = 0

# NUMBER OF COMPONENTS
get_ncomponents(FE::L2P0FiniteElement{T,1,1} where T <: Real) = 1
get_ncomponents(FE::L2P0FiniteElement{T,2,1} where T <: Real) = 1
get_ncomponents(FE::L2P0FiniteElement{T,2,2} where T <: Real) = 2

# LOCAL DOF TO GLOBAL DOF ON CELL
get_globaldof4cell(FE::L2P0FiniteElement{T,1,1} where T <: Real, cell, ::Val{1}) = cell
get_globaldof4cell(FE::L2P0FiniteElement{T,2,1} where T <: Real, cell, ::Val{1}) = cell
get_globaldof4cell(FE::L2P0FiniteElement{T,2,2} where T <: Real, cell, ::Val{1}) = cell
get_globaldof4cell(FE::L2P0FiniteElement{T,2,2} where T <: Real, cell, ::Val{2}) = size(FE.grid.nodes4cells,1) + cell

# LOCAL DOF TO GLOBAL DOF ON FACE
# none


# BASIS FUNCTIONS
function get_all_basis_functions_on_cell(FE::L2P0FiniteElement{T,1,1} where T <: Real)
    function closure(xref)
        return [1.0]
    end
end

function get_all_basis_functions_on_cell(FE::L2P0FiniteElement{T,2,1} where T <: Real)
    function closure(xref)
        return [1.0]
    end
end

function get_all_basis_functions_on_cell(FE::L2P0FiniteElement{T,2,2} where T <: Real)
    function closure(xref)
        return [1.0 0.0;
                0.0 1.0]
    end
end
