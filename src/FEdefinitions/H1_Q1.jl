struct H1Q1FiniteElement{T,ncomponents} <: AbstractH1FiniteElement where {T <: Real, ncomponents <: Int}
    name::String;                 # full name of finite element (used in messages)
    grid::Grid.Mesh{T};           # link to grid
end

function getQ1FiniteElement(grid,ncomponents)
    ensure_nodes4faces!(grid);
    ensure_volume4cells!(grid);
    T = eltype(grid.coords4nodes);
    return H1Q1FiniteElement{T,ncomponents}("Q1 (H1FiniteElement, ncomponents=$ncomponents)",grid)
end 
  
function get_xref4dof(FE::H1Q1FiniteElement{T,1} where {T <: Real}, ::Grid.Abstract2DQuadrilateral) 
    xref = Array{Array{Int64,1},1}(undef,4)
    xref[1] = Array{Int64,1}([0, 0])
    xref[2] = Array{Int64,1}([1, 0])
    xref[3] = Array{Int64,1}([1, 1])
    xref[4] = Array{Int64,1}([0, 1])
    return xref, [sparse(I,4,4)]
end    
function get_xref4dof(FE::H1Q1FiniteElement{T,2} where {T <: Real}, ::Grid.Abstract2DQuadrilateral) 
    xref = Array{Array{Int64,1},1}(undef,8)
    xref[1] = Array{Int64,1}([0, 0])
    xref[2] = Array{Int64,1}([1, 0])
    xref[3] = Array{Int64,1}([1, 1])
    xref[4] = Array{Int64,1}([0, 1])
    xref[[5,6,7,8]] = xref[[1,2,3,4]]
    return xref, [speyec(8,[5,6,7,8]), speyec(8,[1,2,3,4])]
end    

# POLYNOMIAL ORDER
get_polynomial_order(FE::H1Q1FiniteElement) = 2;

# TOTAL NUMBER OF DOFS
get_ndofs(FE::H1Q1FiniteElement{T,1} where {T <: Real}) = size(FE.grid.coords4nodes,1);
get_ndofs(FE::H1Q1FiniteElement{T,2} where {T <: Real}) = 2*size(FE.grid.coords4nodes,1);

# NUMBER OF DOFS ON ELEMTYPE
get_ndofs4elemtype(FE::H1Q1FiniteElement{T,1} where {T <: Real}, ::Grid.Grid.Abstract0DElemType) = 1
get_ndofs4elemtype(FE::H1Q1FiniteElement{T,1} where {T <: Real}, ::Grid.Abstract1DElemType) = 2
get_ndofs4elemtype(FE::H1Q1FiniteElement{T,1} where {T <: Real}, ::Grid.Abstract2DQuadrilateral) = 4
get_ndofs4elemtype(FE::H1Q1FiniteElement{T,2} where {T <: Real}, ::Grid.Grid.Abstract0DElemType) = 2
get_ndofs4elemtype(FE::H1Q1FiniteElement{T,2} where {T <: Real}, ::Grid.Abstract1DElemType) = 4
get_ndofs4elemtype(FE::H1Q1FiniteElement{T,2} where {T <: Real}, ::Grid.Abstract2DQuadrilateral) = 8

# NUMBER OF COMPONENTS
get_ncomponents(FE::H1Q1FiniteElement{T,1} where {T <: Real}) = 1
get_ncomponents(FE::H1Q1FiniteElement{T,2} where {T <: Real}) = 2

# LOCAL DOF TO GLOBAL DOF ON CELL
function get_dofs_on_cell!(dofs,FE::H1Q1FiniteElement{T,1} where {T <: Real}, cell::Int64, ::Grid.AbstractElemType)
    dofs[:] = FE.grid.nodes4cells[cell,:]
end
function get_dofs_on_cell!(dofs,FE::H1Q1FiniteElement{T,2} where {T <: Real}, cell::Int64, ::Grid.Abstract2DQuadrilateral)
    dofs[1:4] = FE.grid.nodes4cells[cell,:]
    dofs[5:8] = size(FE.grid.coords4nodes,1) .+ dofs[1:4]
end
function get_dofs_on_cell!(dofs,FE::H1Q1FiniteElement{T,2} where {T <: Real}, cell::Int64, ::Grid.Abstract1DElemType)
    dofs[1:2] = FE.grid.nodes4cells[cell,:]
    dofs[3:4] = size(FE.grid.coords4nodes,1) .+ dofs[1:2]
end
function get_dofs_on_face!(dofs,FE::H1Q1FiniteElement{T,1} where {T <: Real}, face::Int64, ::Grid.Abstract0DElemType)
    dofs[1] = FE.grid.nodes4faces[face,1]
end
function get_dofs_on_face!(dofs,FE::H1Q1FiniteElement{T,1} where {T <: Real}, face::Int64, ::Grid.Abstract1DElemType)
    dofs[:] = FE.grid.nodes4faces[face,:]
end
function get_dofs_on_face!(dofs,FE::H1Q1FiniteElement{T,2} where {T <: Real}, face::Int64, ::Grid.Abstract0DElemType)
    dofs[1] = FE.grid.nodes4faces[face,1]
    dofs[2] = size(FE.grid.coords4nodes,1) + dofs[1]
end
function get_dofs_on_face!(dofs,FE::H1Q1FiniteElement{T,2} where {T <: Real}, face::Int64, ::Grid.Abstract1DElemType)
    dofs[1:2] = FE.grid.nodes4faces[face,:]
    dofs[3:4] = size(FE.grid.coords4nodes,1) .+ dofs[1:2]
end

# BASIS FUNCTIONS
function get_basis_on_cell(FE::H1Q1FiniteElement{T,1} where T <: Real, ::Grid.Abstract1DElemType)
    function closure(xref)
        return [1 - xref[1],
                xref[1]]
    end
end

function get_basis_on_cell(FE::H1Q1FiniteElement{T,2} where T <: Real, ::Grid.Abstract1DElemType)
    temp = 0.0;
    function closure(xref)
        temp = 1 - xref[1]
        return [temp 0.0;
                xref[1] 0.0;
                0.0 temp;
                0.0 xref[1]]
    end
end

function get_basis_on_cell(FE::H1Q1FiniteElement{T,1} where T <: Real, ::Grid.Abstract2DQuadrilateral)
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

function get_basis_on_cell(FE::H1Q1FiniteElement{T,2} where T <: Real, ::Grid.Abstract2DQuadrilateral)
    a = 0.0;
    b = 0.0;
    function closure(xref)
        a = 1 - xref[1]
        b = 1 - xref[2]
        return [a*b 0.0;
                xref[1]*b 0.0;
                xref[1]*xref[2] 0.0;
                xref[2]*a 0.0
                0.0 a*b;
                0.0 xref[1]*b;
                0.0 xref[1]*xref[2];
                0.0 xref[2]*a]
    end
end

function get_basis_on_face(FE::H1Q1FiniteElement, ET::Grid.AbstractElemType)
    return get_basis_on_cell(FE, ET)
end



# DISCRETE DIVERGENCE-PRESERVING HDIV-RECONSTRUCTION

function get_Hdivreconstruction_space(FE::H1Q1FiniteElement{T,2} where T <: Real, ::Grid.Abstract2DQuadrilateral, variant::Int = 1)
    if (variant == 1)
        return getRT0FiniteElement(FE.grid)
    end    
end


function get_Hdivreconstruction_trafo!(T,FE::H1Q1FiniteElement{T,2} where T <: Real, FE_hdiv::HdivRT0FiniteElement)
    ensure_length4faces!(FE.grid);
    nfaces = size(FE.grid.nodes4faces,1)
    nnodes = size(FE.grid.coords4nodes,1)

    # coefficient for facial ABF dofs
    # = integral of normal flux
    for face = 1 : nfaces
        # reconstruction coefficients for quadratic Q1 basis functions
        # (at the boundary they are linear like the triangular P1 functions)
        for k = 1 : 2
            node = FE.grid.nodes4faces[face,k]
            T[node,face] = 1 // 2 * FE.grid.length4faces[face] * FE.grid.normal4faces[face,1]
            T[nnodes+node,face] = 1 // 2 * FE.grid.length4faces[face] * FE.grid.normal4faces[face,2]
        end
    end
    return T
end