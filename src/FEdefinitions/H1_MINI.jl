struct H1MINIFiniteElement{T,ncomponents} <: AbstractH1FiniteElement where {T <: Real, ncomponents <: Int}
    name::String;                 # full name of finite element (used in messages)
    grid::Grid.Mesh{T};           # link to grid
end

function getMINIFiniteElement(grid,ncomponents)
    ensure_nodes4faces!(grid);
    ensure_volume4cells!(grid);
    T = eltype(grid.coords4nodes);
    return H1MINIFiniteElement{T,ncomponents}("MINI (H1FiniteElement, ncomponents=$ncomponents)",grid)
end 

function get_xref4dof(FE::H1MINIFiniteElement{T,1} where {T <: Real}, ::Grid.ElemType2DTriangle) 
    return Array{Float64,2}([0 0; 1 0; 0 1; 1//3 1//3])
end    
function get_xref4dof(FE::H1MINIFiniteElement{T,2} where {T <: Real}, ::Grid.ElemType2DTriangle) 
    return repeat(Array{Float64,2}([0 0; 1 0; 0 1; 1//3 1//3]),2)
end    

# POLYNOMIAL ORDER
get_polynomial_order(FE::H1MINIFiniteElement) = 3;

# TOTAL NUMBER OF DOFS
get_ndofs(FE::H1MINIFiniteElement{T,1} where {T <: Real}) = size(FE.grid.coords4nodes,1) + size(FE.grid.nodes4cells,1);
get_ndofs(FE::H1MINIFiniteElement{T,2} where {T <: Real}) = 2*(size(FE.grid.coords4nodes,1)  + size(FE.grid.nodes4cells,1));

# NUMBER OF DOFS ON ELEMTYPE
get_ndofs4elemtype(FE::H1MINIFiniteElement{T,1} where {T <: Real}, ::Grid.Abstract1DElemType) = 2
get_ndofs4elemtype(FE::H1MINIFiniteElement{T,1} where {T <: Real}, ::Grid.ElemType2DTriangle) = 4
get_ndofs4elemtype(FE::H1MINIFiniteElement{T,2} where {T <: Real}, ::Grid.Abstract1DElemType) = 4
get_ndofs4elemtype(FE::H1MINIFiniteElement{T,2} where {T <: Real}, ::Grid.ElemType2DTriangle) = 8

# NUMBER OF COMPONENTS
get_ncomponents(FE::H1MINIFiniteElement{T,1} where {T <: Real}) = 1
get_ncomponents(FE::H1MINIFiniteElement{T,2} where {T <: Real}) = 2

# LOCAL DOF TO GLOBAL DOF ON CELL
function get_dofs_on_cell!(dofs,FE::H1MINIFiniteElement{T,1} where {T <: Real}, cell::Int64, ::Grid.ElemType2DTriangle)
    dofs[1:3] = FE.grid.nodes4cells[cell,:]
    dofs[4] = size(FE.grid.coords4nodes,1) + cell
end
function get_dofs_on_cell!(dofs,FE::H1MINIFiniteElement{T,2} where {T <: Real}, cell::Int64, ::Grid.ElemType2DTriangle)
    dofs[1:3] = FE.grid.nodes4cells[cell,:]
    dofs[4] = size(FE.grid.coords4nodes,1) + cell
    dofs[5:8] = (size(FE.grid.coords4nodes,1) + size(FE.grid.nodes4cells,1)) .+ dofs[1:4]
end

function get_dofs_on_face!(dofs,FE::H1MINIFiniteElement{T,1} where {T <: Real}, face::Int64, ::Grid.Abstract1DElemType)
    dofs[:] = FE.grid.nodes4faces[face,:]
end
function get_dofs_on_face!(dofs,FE::H1MINIFiniteElement{T,2} where {T <: Real}, face::Int64, ::Grid.Abstract1DElemType)
    dofs[1:2] = FE.grid.nodes4faces[face,:]
    dofs[3:4] = (size(FE.grid.coords4nodes,1) + size(FE.grid.nodes4cells,1)) .+ dofs[1:2]
end

# BASIS FUNCTIONS

function get_basis_on_elemtype(FE::H1MINIFiniteElement{T,1} where T <: Real, ::Grid.ElemType2DTriangle)
    temp = 0.0;
    cb = 0.0;
    function closure(xref)
        temp = 1 - xref[1] - xref[2];
        cb = 27*temp*xref[1]*xref[2]
        return [temp,
                xref[1],
                xref[2],
                cb]
    end
end

function get_basis_on_elemtype(FE::H1MINIFiniteElement{T,2} where T <: Real, ::Grid.ElemType2DTriangle)
    temp = 0.0;
    cb = 0.0;
    function closure(xref)
        temp = 1 - xref[1] - xref[2];
        cb = 27*temp*xref[1]*xref[2]
        return [temp 0.0;
                xref[1] 0.0;
                xref[2] 0.0;
                cb 0.0;
                0.0 temp;
                0.0 xref[1];
                0.0 xref[2];
                0.0 cb]
    end
end


function get_basis_on_elemtype(FE::H1P1FiniteElement{T,1} where T <: Real, ::Grid.Abstract1DElemType)
    function closure(xref)
        return [1 - xref[1],
                xref[1]]
    end
end

function get_basis_on_elemtype(FE::H1MINIFiniteElement{T,2} where T <: Real, ::Grid.Abstract1DElemType)
    temp = 0.0;
    function closure(xref)
        temp = 1 - xref[1];
        return [temp 0.0;
                xref[1] 0.0;
                0.0 temp;
                0.0 xref[1]]
    end
end
