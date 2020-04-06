struct H1P1FiniteElement{T,ncomponents} <: AbstractH1FiniteElement where {T <: Real, ncomponents <: Int}
    name::String;                 # full name of finite element (used in messages)
    grid::Grid.Mesh{T};           # link to grid
end

function getP1FiniteElement(grid,ncomponents)
    ensure_nodes4faces!(grid);
    ensure_volume4cells!(grid);
    T = eltype(grid.coords4nodes);
    return H1P1FiniteElement{T,ncomponents}("P1 (H1FiniteElement, ncomponents=$ncomponents)",grid)
end 


function get_xref4dof(FE::H1P1FiniteElement{T,1} where {T <: Real}, ::Grid.Abstract0DElemType) 
    return Array{Float64,2}([1]')', [sparse(I,1,1)]
end    
function get_xref4dof(FE::H1P1FiniteElement{T,1} where {T <: Real}, ::Grid.Abstract1DElemType) 
    xref = Array{Array{Int64,1},1}(undef,2)
    xref[1] = Array{Int64,1}([0])
    xref[2] = Array{Int64,1}([1])
    return xref, [sparse(I,2,2)]
end    
function get_xref4dof(FE::H1P1FiniteElement{T,2} where {T <: Real}, ::Grid.Abstract1DElemType) 
    xref = Array{Array{Int64,1},1}(undef,4)
    xref[1] = Array{Int64,1}([0])
    xref[2] = Array{Int64,1}([1])
    xref[[3,4]] = xref[[1,2]]
    return xref, [speyec(4,[3,4]), speyec(4,[1,2])]
end    
function get_xref4dof(FE::H1P1FiniteElement{T,1} where {T <: Real}, ::Grid.ElemType2DTriangle) 
    xref = Array{Array{Int64,1},1}(undef,3)
    xref[1] = Array{Int64,1}([0, 0])
    xref[2] = Array{Int64,1}([1, 0])
    xref[3] = Array{Int64,1}([0, 1])
    return xref, [sparse(I,3,3)]
end    
function get_xref4dof(FE::H1P1FiniteElement{T,1} where {T <: Real}, ::Grid.ElemType2DParallelogram) 
    xref = Array{Array{Int64,1},1}(undef,4)
    xref[1] = Array{Int64,1}([0, 0])
    xref[2] = Array{Int64,1}([1, 0])
    xref[3] = Array{Int64,1}([1, 1])
    xref[4] = Array{Int64,1}([0, 1])
    return xref, [sparse(I,4,4)]
end    
function get_xref4dof(FE::H1P1FiniteElement{T,2} where {T <: Real}, ::Grid.ElemType2DTriangle) 
    xref = Array{Array{Int64,1},1}(undef,6)
    xref[1] = Array{Int64,1}([0, 0])
    xref[2] = Array{Int64,1}([1, 0])
    xref[3] = Array{Int64,1}([0, 1])
    xref[[4,5,6]] = xref[[1,2,3]]
    return xref, [speyec(6,[4,5,6]), speyec(6,[1,2,3])]
end    

# POLYNOMIAL ORDER
get_polynomial_order(FE::H1P1FiniteElement) = 1;

# TOTAL NUMBER OF DOFS
get_ndofs(FE::H1P1FiniteElement{T,1} where {T <: Real}) = size(FE.grid.coords4nodes,1);
get_ndofs(FE::H1P1FiniteElement{T,2} where {T <: Real}) = 2*size(FE.grid.coords4nodes,1);

# NUMBER OF DOFS ON ELEMTYPE
get_ndofs4elemtype(FE::H1P1FiniteElement{T,1} where {T <: Real}, ::Grid.Grid.Abstract0DElemType) = 1
get_ndofs4elemtype(FE::H1P1FiniteElement{T,1} where {T <: Real}, ::Grid.Abstract1DElemType) = 2
get_ndofs4elemtype(FE::H1P1FiniteElement{T,1} where {T <: Real}, ::Grid.ElemType2DTriangle) = 3
get_ndofs4elemtype(FE::H1P1FiniteElement{T,1} where {T <: Real}, ::Grid.ElemType2DParallelogram) = 4
get_ndofs4elemtype(FE::H1P1FiniteElement{T,2} where {T <: Real}, ::Grid.Grid.Abstract0DElemType) = 2
get_ndofs4elemtype(FE::H1P1FiniteElement{T,2} where {T <: Real}, ::Grid.Abstract1DElemType) = 4
get_ndofs4elemtype(FE::H1P1FiniteElement{T,2} where {T <: Real}, ::Grid.ElemType2DTriangle) = 6
get_ndofs4elemtype(FE::H1P1FiniteElement{T,2} where {T <: Real}, ::Grid.ElemType2DParallelogram) = 8

# NUMBER OF COMPONENTS
get_ncomponents(FE::H1P1FiniteElement{T,1} where {T <: Real}) = 1
get_ncomponents(FE::H1P1FiniteElement{T,2} where {T <: Real}) = 2

# LOCAL DOF TO GLOBAL DOF ON CELL
function get_dofs_on_cell!(dofs,FE::H1P1FiniteElement{T,1} where {T <: Real}, cell::Int64, ::Grid.AbstractElemType)
    dofs[:] = FE.grid.nodes4cells[cell,:]
end
function get_dofs_on_cell!(dofs,FE::H1P1FiniteElement{T,2} where {T <: Real}, cell::Int64, ::Grid.Abstract1DElemType)
    dofs[1:2] = FE.grid.nodes4cells[cell,:]
    dofs[3:4] = size(FE.grid.coords4nodes,1) .+ dofs[1:2]
end
function get_dofs_on_cell!(dofs,FE::H1P1FiniteElement{T,2} where {T <: Real}, cell::Int64, ::Grid.ElemType2DTriangle)
    dofs[1:3] = FE.grid.nodes4cells[cell,:]
    dofs[4:6] = size(FE.grid.coords4nodes,1) .+ dofs[1:3]
end

function get_dofs_on_face!(dofs,FE::H1P1FiniteElement{T,1} where {T <: Real}, face::Int64, ::Grid.Grid.Abstract0DElemType)
    dofs[1] = FE.grid.nodes4faces[face,1]
end
function get_dofs_on_face!(dofs,FE::H1P1FiniteElement{T,1} where {T <: Real}, face::Int64, ::Grid.Abstract1DElemType)
    dofs[:] = FE.grid.nodes4faces[face,:]
end
function get_dofs_on_face!(dofs,FE::H1P1FiniteElement{T,2} where {T <: Real}, face::Int64, ::Grid.Grid.Abstract0DElemType)
    dofs[1] = FE.grid.nodes4faces[face,1]
    dofs[2] = size(FE.grid.coords4nodes,1) + dofs[1]
end
function get_dofs_on_face!(dofs,FE::H1P1FiniteElement{T,2} where {T <: Real}, face::Int64, ::Grid.Abstract1DElemType)
    dofs[1:2] = FE.grid.nodes4faces[face,:]
    dofs[3:4] = size(FE.grid.coords4nodes,1) .+ dofs[1:2]
end

# BASIS FUNCTIONS
function get_basis_on_cell(FE::H1P1FiniteElement{T,1} where T <: Real, ::Grid.Abstract0DElemType)
    function closure(xref)
        return [1]
    end
end

function get_basis_on_cell(FE::H1P1FiniteElement{T,2} where T <: Real, ::Grid.Abstract0DElemType)
    function closure(xref)
        return [1 0.0;
                0.0 1]
    end
end

function get_basis_on_cell(FE::H1P1FiniteElement{T,1} where T <: Real, ::Grid.Abstract1DElemType)
    function closure(xref)
        return [1 - xref[1],
                xref[1]]
    end
end

function get_basis_on_cell(FE::H1P1FiniteElement{T,2} where T <: Real, ::Grid.Abstract1DElemType)
    temp = 0.0;
    function closure(xref)
        temp = 1 - xref[1]
        return [temp 0.0;
                xref[1] 0.0;
                0.0 temp;
                0.0 xref[1]]
    end
end

function get_basis_on_cell(FE::H1P1FiniteElement{T,1} where T <: Real, ::Grid.ElemType2DParallelogram)
    a = 0.0;
    b = 0.0;
    function closure(xref)
        a = 1 - xref[1]
        b = 1 - xref[2]
        return [a*b,
                xref[1]*b,
                xref[1]*xref[2],
                xref[2]*a]
    end
end

function get_basis_on_cell(FE::H1P1FiniteElement{T,1} where T <: Real, ::Grid.ElemType2DTriangle)
    function closure(xref)
        return [1 - xref[1] - xref[2],
                xref[1],
                xref[2]]
    end
end

function get_basis_on_cell(FE::H1P1FiniteElement{T,2} where T <: Real, ::Grid.ElemType2DTriangle)
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

function get_basis_on_face(FE::H1P1FiniteElement, ET::Grid.AbstractElemType)
    return get_basis_on_cell(FE, ET)
end
