struct H1P2FiniteElement{T, ncomponents} <: AbstractH1FiniteElement
    name::String;                 # full name of finite element (used in messages)
    grid::Grid.Mesh{T};           # link to grid
    nbubbles::Int;                # helper variable
end

function getP2FiniteElement(grid,ncomponents)
    ensure_faces4cells!(grid);
    ensure_nodes4faces!(grid);
    ensure_volume4cells!(grid);
    T = eltype(grid.coords4nodes);
    if typeof(grid.elemtypes[1]) <: Grid.Abstract1DElemType
        nbubbles = size(grid.nodes4cells,1)
    elseif typeof(grid.elemtypes[1]) <: Grid.Abstract2DElemType    
        nbubbles = size(grid.nodes4faces,1)
    end    
    return H1P2FiniteElement{T,ncomponents}("P2 (H1FiniteElement, ncomponents=$ncomponents)",grid,nbubbles)
end 

function get_xref4dof(FE::H1P2FiniteElement{T,1} where {T <: Real}, ::Grid.Abstract1DElemType) 
    return Array{Float64,2}([0,1,0.5]')'
end    
function get_xref4dof(FE::H1P2FiniteElement{T,2} where {T <: Real}, ::Grid.Abstract1DElemType) 
    return repeat(Array{Float64,2}([0,1,0.5]')',2)
end    
function get_xref4dof(FE::H1P2FiniteElement{T,1} where {T <: Real}, ::Grid.ElemType2DTriangle) 
    return Array{Float64,2}([0 0; 1 0; 0 1; 0.5 0; 0.5 0.5; 0 0.5])
end    
function get_xref4dof(FE::H1P2FiniteElement{T,2} where {T <: Real}, ::Grid.ElemType2DTriangle) 
    return repeat(Array{Float64,2}([0 0; 1 0; 0 1; 0.5 0; 0.5 0.5; 0 0.5]),2)
end    

# POLYNOMIAL ORDER
get_polynomial_order(FE::H1P2FiniteElement) = 2;

# TOTAL NUMBER OF DOFS
get_ndofs(FE::H1P2FiniteElement{T,1} where {T <: Real}) = size(FE.grid.coords4nodes,1) + FE.nbubbles;
get_ndofs(FE::H1P2FiniteElement{T,2} where {T <: Real}) = 2*(size(FE.grid.coords4nodes,1) + FE.nbubbles);

# NUMBER OF DOFS ON ELEMTYPE
get_ndofs4elemtype(FE::H1P2FiniteElement{T,1} where {T <: Real}, ::Grid.Abstract0DElemType) = 1
get_ndofs4elemtype(FE::H1P2FiniteElement{T,1} where {T <: Real}, ::Grid.Abstract1DElemType) = 3
get_ndofs4elemtype(FE::H1P2FiniteElement{T,1} where {T <: Real}, ::Grid.ElemType2DTriangle) = 6
get_ndofs4elemtype(FE::H1P2FiniteElement{T,2} where {T <: Real}, ::Grid.Abstract1DElemType) = 6
get_ndofs4elemtype(FE::H1P2FiniteElement{T,2} where {T <: Real}, ::Grid.ElemType2DTriangle) = 12

# NUMBER OF COMPONENTS
get_ncomponents(FE::H1P2FiniteElement{T,1} where {T <: Real}) = 1
get_ncomponents(FE::H1P2FiniteElement{T,2} where {T <: Real}) = 2

# LOCAL DOF TO GLOBAL DOF ON CELL
function get_dofs_on_cell!(dofs,FE::H1P2FiniteElement{T,1} where {T <: Real}, cell::Int64, ::Grid.Abstract1DElemType)
    dofs[1:2] = FE.grid.nodes4cells[cell,:]
    dofs[3] = size(FE.grid.coords4nodes,1) + cell
end
function get_dofs_on_cell!(dofs,FE::H1P2FiniteElement{T,1} where {T <: Real}, cell::Int64, ::Grid.ElemType2DTriangle)
    dofs[1:3] = FE.grid.nodes4cells[cell,:]
    dofs[4:6] = size(FE.grid.coords4nodes,1) .+ FE.grid.faces4cells[cell,:]
end
function get_dofs_on_cell!(dofs,FE::H1P2FiniteElement{T,2} where {T <: Real}, cell::Int64, ::Grid.ElemType2DTriangle)
    dofs[1:3] = FE.grid.nodes4cells[cell,:]
    dofs[4:6] = size(FE.grid.coords4nodes,1) .+ FE.grid.faces4cells[cell,:]
    dofs[7:12] = (size(FE.grid.coords4nodes,1) + size(FE.grid.nodes4faces,1)) .+ dofs[1:6]
end

function get_dofs_on_face!(dofs,FE::H1P2FiniteElement{T,1} where {T <: Real}, face::Int64, ::Grid.Abstract0DElemType)
    dofs[1] = FE.grid.nodes4faces[face,1]
end
function get_dofs_on_face!(dofs,FE::H1P2FiniteElement{T,1} where {T <: Real}, face::Int64, ::Grid.Abstract1DElemType)
    dofs[1:2] = FE.grid.nodes4faces[face,:]
    dofs[3] = size(FE.grid.coords4nodes,1) + face
end
function get_dofs_on_face!(dofs,FE::H1P2FiniteElement{T,2} where {T <: Real}, face::Int64, ::Grid.Abstract1DElemType)
    dofs[1:2] = FE.grid.nodes4faces[face,:]
    dofs[3] = size(FE.grid.coords4nodes,1) + face
    dofs[4:6] = (size(FE.grid.coords4nodes,1) + size(FE.grid.nodes4faces,1)) .+ dofs[1:3]
end

# BASIS FUNCTIONS
function get_basis_on_elemtype(FE::H1P2FiniteElement{T,1} where T <: Real, ::Grid.Abstract1DElemType)
    temp = 0.0;
    function closure(xref)
        temp = 1 - xref[1]
        return [2*temp*(temp - 1//2),
                2*xref[1]*(xref[1] - 1//2),
                4*temp*xref[1]]
    end
end

function get_basis_on_elemtype(FE::H1P2FiniteElement{T,2} where T <: Real, ::Grid.Abstract1DElemType)
    temp = 0.0;
    a = 0.0;
    b = 0.0;
    function closure(xref)
        temp = 1 - xref[1]
        a = 2*temp*(temp - 1//2)
        b = 2*xref[1]*(xref[1] - 1//2)
        c = 4*temp*xref[1]
        return [a 0.0;
                b 0.0;
                c 0.0;
                0.0 a;
                0.0 b;
                0.0 c]
    end
end

function get_basis_on_elemtype(FE::H1P2FiniteElement{T,1} where T <: Real, ::Grid.ElemType2DTriangle)
    temp = 0.0;
    function closure(xref)
        temp = 1 - xref[1] - xref[2]
        return [2*temp*(temp - 1//2),
                2*xref[1]*(xref[1] - 1//2),
                2*xref[2]*(xref[2] - 1//2),
                4*temp*xref[1],
                4*xref[1]*xref[2],
                4*temp*xref[2]]
    end
end

function get_basis_on_elemtype(FE::H1P2FiniteElement{T,2} where T <: Real, ::Grid.ElemType2DTriangle)
    temp = 0.0;
    a = 0.0;
    b = 0.0;
    c = 0.0;
    d = 0.0;
    e = 0.0;
    f = 0.0;
    function closure(xref)
        temp = 1 - xref[1] - xref[2];
        a = 2*temp*(temp - 1//2);
        b = 2*xref[1]*(xref[1] - 1//2);
        c = 2*xref[2]*(xref[2] - 1//2);
        d = 4*temp*xref[1];
        e = 4*xref[1]*xref[2];
        f = 4*temp*xref[2];
        return [a 0.0;    
                b 0.0;
                c 0.0;
                d 0.0;
                e 0.0;
                f 0.0;
                0.0 a;
                0.0 b;
                0.0 c;
                0.0 d;
                0.0 e;
                0.0 f]
                
    end
end