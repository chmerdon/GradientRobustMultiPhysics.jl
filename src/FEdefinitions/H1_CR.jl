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
    xref = Array{Array{Float64,1},1}(undef,3)
    xref[1] = Array{Float64,1}([0.5, 0])
    xref[2] = Array{Float64,1}([0.5, 0.5])
    xref[3] = Array{Float64,1}([0, 0.5])
    return xref, [sparse(I,3,3)]
end    
function get_xref4dof(FE::H1CRFiniteElement{T,2,2} where {T <: Real}, ::Grid.ElemType2DTriangle) 
    xref = Array{Array{Float64,1},1}(undef,6)
    xref[1] = Array{Float64,1}([0.5, 0])
    xref[2] = Array{Float64,1}([0.5, 0.5])
    xref[3] = Array{Float64,1}([0, 0.5])
    xref[[4,5,6]] = xref[[1,2,3]]
    return xref, [speyec(6,[4,5,6]), speyec(6,[1,2,3])]
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
function get_basis_on_cell(FE::H1CRFiniteElement{T,2,1} where {T <: Real}, ::Grid.ElemType2DTriangle)
    temp = 0.0;
    function closure(xref)
        return [1 - 2*xref[2];
                2*(xref[1]+xref[2]) - 1;
                1 - 2*xref[1]]
    end
end

function get_basis_on_face(FE::H1CRFiniteElement{T,2,1} where {T <: Real}, ::Grid.Abstract1DElemType)
    function closure(xref)
        return [1.0]
    end
end

function get_basis_on_cell(FE::H1CRFiniteElement{T,2,2} where {T <: Real}, ::Grid.ElemType2DTriangle)
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

function get_basis_on_face(FE::H1CRFiniteElement{T,2,2} where {T <: Real}, ::Grid.Abstract1DElemType)
    function closure(xref)
        return [1.0 0.0;
                0.0 1.0]
    end
end



# DISCRETE DIVERGENCE-PRESERVING HDIV-RECONSTRUCTION

function get_Hdivreconstruction_space(FE::H1CRFiniteElement{T,2,2} where T <: Real, ET::Grid.AbstractElemType, variant::Int = 1)
    if (variant == 1)
        return getRT0FiniteElement(FE.grid)
    elseif (variant == 2)
        return getBDM1FiniteElement(FE.grid)    
    end    
end

function get_Hdivreconstruction_trafo!(T,FE::H1CRFiniteElement{T,2,2} where T <: Real, FE_hdiv::HdivRT0FiniteElement)
    ensure_length4faces!(FE.grid);
    nfaces = size(FE.grid.nodes4faces,1)
    nnodes = size(FE.grid.coords4nodes,1)
    for face = 1 : nfaces
        # reconstruction coefficients for CR basis functions
        T[face,face] = 1 * FE.grid.length4faces[face] * FE.grid.normal4faces[face,1]
        T[nfaces+face,face] = 1 * FE.grid.length4faces[face] * FE.grid.normal4faces[face,2]
    end
    return T
end


function get_Hdivreconstruction_trafo!(T,FE::H1CRFiniteElement{T,2,2} where T <: Real, FE_hdiv::HdivBDM1FiniteElement)
    ensure_length4faces!(FE.grid);
    ensure_cells4faces!(FE.grid);
    nfaces = size(FE.grid.nodes4faces,1)
    nnodes = size(FE.grid.coords4nodes,1)
    ncells = size(FE.grid.nodes4cells,1)
    for face = 1 : nfaces
        # reconstruction coefficients for CR basis functions
        T[face,face] = 1 * FE.grid.length4faces[face] * FE.grid.normal4faces[face,1]
        T[nfaces+face,face] = 1 * FE.grid.length4faces[face] * FE.grid.normal4faces[face,2]
    end

    cellfaces = [0 0 0 0 0]
    neighbourfactor = [0.5 0.5 0.5 0.5 0.5]
    factor = 1 // 3
    for cell = 1 : ncells
        cellfaces[2:4] = FE.grid.faces4cells[cell,:]
        cellfaces[1] = cellfaces[4]
        cellfaces[5] = cellfaces[2]
        for j = 1 : 5
            if FE.grid.cells4faces[cellfaces[j],2] > 0
                neighbourfactor[j] = 1 // 2
            else
                neighbourfactor[j] = 0
            end
        end
        

        for j = 2 : 4
            T[cellfaces[j],nfaces + cellfaces[j-1]] = factor * neighbourfactor[j-1] * FE.grid.length4faces[cellfaces[j-1]] * FE.grid.normal4faces[cellfaces[j-1],1]
            T[nfaces + cellfaces[j],nfaces + cellfaces[j-1]] = factor * neighbourfactor[j-1] * FE.grid.length4faces[cellfaces[j-1]] * FE.grid.normal4faces[cellfaces[j-1],2]
            T[cellfaces[j],nfaces + cellfaces[j+1]] = factor * neighbourfactor[j+1] * FE.grid.length4faces[cellfaces[j+1]] * FE.grid.normal4faces[cellfaces[j+1],1]
            T[nfaces + cellfaces[j],nfaces + cellfaces[j+1]] = factor * neighbourfactor[j+1] * FE.grid.length4faces[cellfaces[j+1]] * FE.grid.normal4faces[cellfaces[j+1],2]
        end    
    end
    return T
end

