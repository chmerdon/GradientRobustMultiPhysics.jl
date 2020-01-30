struct H1CRFiniteElement{T,dim,ncomponents} <: AbstractH1FiniteElement
    name::String;                 # full name of finite element (used in messages)
    grid::Grid.Mesh{T};           # link to grid
end

function getCRFiniteElement(grid,dim,ncomponents::Int)
    ensure_nodes4faces!(grid);
    ensure_faces4cells!(grid);
    ensure_volume4cells!(grid);
    T = eltype(grid.coords4nodes);
    @assert dim > 1 # makes no sense for dim = 1 (or equal to P0 which is not H1 anymore)
    return H1CRFiniteElement{T,dim,ncomponents}("CR (H1FiniteElement, dim=$dim, ncomponents=$ncomponents)",grid)
end   


function get_xref4dof(FE::H1CRFiniteElement{T,2,1} where {T <: Real}, ::Grid.ElemType2DTriangle) 
    return Array{Float64,2}([0.5 0; 0.5 0.5; 0 0.5])
end    
function get_xref4dof(FE::H1CRFiniteElement{T,2,2} where {T <: Real}, ::Grid.ElemType2DTriangle) 
    return repeat(Array{Float64,2}([0.5 0; 0.5 0.5; 0 0.5],2))
end    

# POLYNOMIAL ORDER
get_polynomial_order(FE::H1CRFiniteElement) = 1;

# TOTAL NUMBER OF DOFS
get_ndofs(FE::H1CRFiniteElement{T,2,1} where {T <: Real}) = size(FE.grid.nodes4faces,1);
get_ndofs(FE::H1CRFiniteElement{T,2,2} where {T <: Real}) = 2*size(FE.grid.nodes4faces,1);

# NUMBER OF DOFS ON ELEMTYPE
get_ndofs4elemtype(FE::H1CRFiniteElement{T,2,1} where {T <: Real}, ::Grid.Abstract1DElemType) = 1
get_ndofs4elemtype(FE::H1CRFiniteElement{T,2,1} where {T <: Real}, ::Grid.ElemType2DTriangle) = 3
get_ndofs4elemtype(FE::H1CRFiniteElement{T,2,2} where {T <: Real}, ::Grid.Abstract1DElemType) = 2
get_ndofs4elemtype(FE::H1CRFiniteElement{T,2,2} where {T <: Real}, ::Grid.ElemType2DTriangle) = 6

# NUMBER OF COMPONENTS
get_ncomponents(FE::H1CRFiniteElement{T,2,1} where {T <: Real}) = 1
get_ncomponents(FE::H1CRFiniteElement{T,2,2} where {T <: Real}) = 2

# LOCAL DOF TO GLOBAL DOF ON CELL
function get_dofs_on_cell!(dofs,FE::H1CRFiniteElement{T,2,1} where {T <: Real}, cell::Int64, ::Grid.ElemType2DTriangle)
    dofs[:] = FE.grid.faces4cells[cell,:]
end
function get_dofs_on_cell!(dofs,FE::H1CRFiniteElement{T,2,2} where {T <: Real}, cell::Int64, ::Grid.ElemType2DTriangle)
    dofs[1:3] = FE.grid.faces4cells[cell,:]
    dofs[4:6] = size(FE.grid.nodes4faces,1) .+ dofs[1:3] 
end

# LOCAL DOF TO GLOBAL DOF ON FACE
function get_dofs_on_face!(dofs,FE::H1CRFiniteElement{T,2,1} where {T <: Real}, face::Int64, ::Grid.Abstract1DElemType)
    dofs[1] = face
end
function get_dofs_on_face!(dofs,FE::H1CRFiniteElement{T,2,2} where {T <: Real}, face::Int64, ::Grid.Abstract1DElemType)
    dofs[1] = face
    dofs[2] = size(FE.grid.nodes4faces,1) + face
end

# BASIS FUNCTIONS
function get_basis_on_elemtype(FE::H1CRFiniteElement{T,2,1} where {T <: Real}, ::Grid.ElemType2DTriangle)
    temp = 0.0;
    function closure(xref)
        return [1 - 2*xref[2];
                2*(xref[1]+xref[2]) - 1;
                1 - 2*xref[1]]
    end
end

function get_basis_on_elemtype(FE::H1CRFiniteElement{T,2,1} where {T <: Real}, ::Grid.Abstract1DElemType)
    function closure(xref)
        return [1.0]
    end
end

function get_basis_on_elemtype(FE::H1CRFiniteElement{T,2,2} where {T <: Real}, ::Grid.ElemType2DTriangle)
    temp = 0.0;
    temp2 = 0.0;
    temp3 = 0.0;
    function closure(xref)
        temp = 1 - 2*xref[2]
        temp2 = 2*(xref[1]+xref[2]) - 1
        temp3 = 1 - 2*xref[1]
        return [temp 0.0;
                temp2 0.0;
                temp3 0.0;
                0.0 temp;
                0.0 temp2;
                0.0 temp3]
    end            
end

function get_basis_on_elemtype(FE::H1CRFiniteElement{T,2,2} where {T <: Real}, ::Grid.Abstract1DElemType)
    function closure(xref)
        return [1.0 0.0;
                0.0 1.0]
    end
end
