struct L2P0FiniteElement{T,ncomponents} <: AbstractH1FiniteElement where {T <: Real, ncomponents <: Int}
    name::String;                 # full name of finite element (used in messages)
    grid::Grid.Mesh{T};           # link to grid
end

function getP0FiniteElement(grid,ncomponents)
    ensure_volume4cells!(grid);
    T = eltype(grid.coords4nodes);
    return L2P0FiniteElement{T,ncomponents}("P0 (H1FiniteElement, ncomponents=$ncomponents)",grid)
end 

function get_xref4dof(FE::L2P0FiniteElement{T,1} where {T <: Real}, ::Grid.Grid.Abstract0DElemType) 
    return Array{Float64,2}([1]')'
end    
function get_xref4dof(FE::L2P0FiniteElement{T,1} where {T <: Real}, ::Grid.Abstract1DElemType) 
    return Array{Float64,2}([0.5]')'
end    
function get_xref4dof(FE::L2P0FiniteElement{T,2} where {T <: Real}, ::Grid.Abstract1DElemType) 
    return repeat(Array{Float64,2}([0.5]')',2)
end    
function get_xref4dof(FE::L2P0FiniteElement{T,1} where {T <: Real}, ::Grid.ElemType2DTriangle) 
    return Array{Float64,2}([0.5 0.5])
end    
function get_xref4dof(FE::L2P0FiniteElement{T,2} where {T <: Real}, ::Grid.ElemType2DTriangle) 
    return repeat(Array{Float64,2}([0.5 0.5]),2)
end    

# POLYNOMIAL ORDER
get_polynomial_order(FE::L2P0FiniteElement) = 0;

# TOTAL NUMBER OF DOFS
get_ndofs(FE::L2P0FiniteElement{T,1} where {T <: Real}) = size(FE.grid.nodes4cells,1);
get_ndofs(FE::L2P0FiniteElement{T,2} where {T <: Real}) = 2*size(FE.grid.nodes4cells,1);

# NUMBER OF DOFS ON ELEMTYPE
get_ndofs4elemtype(FE::L2P0FiniteElement{T,1} where {T <: Real}, ::Grid.Abstract1DElemType) = 0
get_ndofs4elemtype(FE::L2P0FiniteElement{T,1} where {T <: Real}, ::Grid.ElemType2DTriangle) = 1
get_ndofs4elemtype(FE::L2P0FiniteElement{T,2} where {T <: Real}, ::Grid.ElemType2DTriangle) = 2

# NUMBER OF COMPONENTS
get_ncomponents(FE::L2P0FiniteElement{T,1} where {T <: Real}) = 1
get_ncomponents(FE::L2P0FiniteElement{T,2} where {T <: Real}) = 2

# LOCAL DOF TO GLOBAL DOF ON CELL
function get_dofs_on_cell!(dofs,FE::L2P0FiniteElement{T,1} where {T <: Real}, cell::Int64, ::Grid.Grid.Abstract2DElemType)
    dofs[1] = cell
end
function get_dofs_on_cell!(dofs,FE::L2P0FiniteElement{T,2} where {T <: Real}, cell::Int64, ::Grid.Grid.Abstract2DElemType)
    dofs[1] = cell
    dofs[2] = cell + size(FE.grid.nodes4cells,1)
end


# BASIS FUNCTIONS
function get_basis_on_elemtype(FE::L2P0FiniteElement{T,1} where T <: Real, ::Grid.Abstract2DElemType)
    function closure(xref)
        return [1.0]
    end
end
function get_basis_on_elemtype(FE::L2P0FiniteElement{T,2} where T <: Real, ::Grid.Abstract2DElemType)
    function closure(xref)
        return [1.0 0.0;
                0.0 1.0]
    end
end
