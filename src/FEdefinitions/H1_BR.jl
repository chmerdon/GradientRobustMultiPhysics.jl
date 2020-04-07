struct H1BRFiniteElement{T, ncomponents} <: AbstractH1FiniteElement
    name::String;                 # full name of finite element (used in messages)
    grid::Grid.Mesh{T};           # link to grid
end

function getBRFiniteElement(grid,ncomponents)
    ensure_nodes4faces!(grid);
    ensure_faces4cells!(grid);
    ensure_volume4cells!(grid);
    ensure_normal4faces!(grid);
    T = eltype(grid.coords4nodes);
    return H1BRFiniteElement{T,ncomponents}("BR (H1FiniteElement, dim=ncomponents=$ncomponents)",grid)
end 

function get_xref4dof(FE::H1BRFiniteElement{T,2} where {T <: Real}, ::Grid.ElemType2DTriangle) 
    xref = Array{Array{Float64,1},1}(undef,9)
    xref[1] = Array{Float64,1}([0, 0])
    xref[2] = Array{Float64,1}([1, 0])
    xref[3] = Array{Float64,1}([0, 1])
    xref[1] = Array{Float64,1}([0, 0])
    xref[[4,5,6]] = xref[[1,2,3]]
    xref[7] = Array{Float64,1}([0.5, 0])
    xref[8] = Array{Float64,1}([0.5, 0.5])
    xref[9] = Array{Float64,1}([0, 0.5])
    InterpolationMatrix = zeros(Float64,9,9)
    InterpolationMatrix[1:3,1:3] = Matrix{Float64}(I,3,3)
    InterpolationMatrix[[1,2,3,7],7] = [-1//2, -1//2, 0, 1]
    InterpolationMatrix[[1,2,3,8],8] = [0, -1//2, -1//2, 1]
    InterpolationMatrix[[1,2,3,9],9] = [-1//2, 0, -1//2, 1]
    InterpolationMatrix2 = zeros(Float64,9,9)
    InterpolationMatrix2[4:6,4:6] = Matrix{Float64}(I,3,3)
    InterpolationMatrix2[[4,5,6,7],7] = [-1//2, -1//2, 0, 1]
    InterpolationMatrix2[[4,5,6,8],8] = [0, -1//2, -1//2, 1]
    InterpolationMatrix2[[4,5,6,9],9] = [-1//2, 0, -1//2, 1]
    return xref, [sparse(InterpolationMatrix), sparse(InterpolationMatrix2)]
end    

# POLYNOMIAL ORDER
get_polynomial_order(FE::H1BRFiniteElement) = typeof(FE.grid.elemtypes[1]) == Grid.ElemType2DParallelogram ? 3 : 2;

# TOTAL NUMBER OF DOFS
get_ndofs(FE::H1BRFiniteElement{T,2} where {T <: Real}) = 2*size(FE.grid.coords4nodes,1) + size(FE.grid.nodes4faces,1);

# NUMBER OF DOFS ON ELEMTYPE
get_ndofs4elemtype(FE::H1BRFiniteElement{T,2} where {T <: Real}, ::Grid.Abstract1DElemType) = 5
get_ndofs4elemtype(FE::H1BRFiniteElement{T,2} where {T <: Real}, ::Grid.ElemType2DTriangle) = 9
get_ndofs4elemtype(FE::H1BRFiniteElement{T,2} where {T <: Real}, ::Grid.ElemType2DParallelogram) = 12

# NUMBER OF COMPONENTS
get_ncomponents(FE::H1BRFiniteElement{T,2} where {T <: Real}) = 2

# LOCAL DOF TO GLOBAL DOF ON CELL
function get_dofs_on_cell!(dofs,FE::H1BRFiniteElement{T,2} where {T <: Real}, cell::Int64, ::Grid.ElemType2DTriangle)
    dofs[1:3] = FE.grid.nodes4cells[cell,:]
    dofs[4:6] = size(FE.grid.coords4nodes,1) .+ dofs[1:3]
    dofs[7:9] = 2*size(FE.grid.coords4nodes,1) .+ FE.grid.faces4cells[cell,:]
end
function get_dofs_on_cell!(dofs,FE::H1BRFiniteElement{T,2} where {T <: Real}, cell::Int64, ::Grid.ElemType2DParallelogram)
    dofs[1:4] = FE.grid.nodes4cells[cell,:]
    dofs[5:8] = size(FE.grid.coords4nodes,1) .+ dofs[1:4]
    dofs[9:12] = 2*size(FE.grid.coords4nodes,1) .+ FE.grid.faces4cells[cell,:]
end

function get_dofs_on_face!(dofs,FE::H1BRFiniteElement{T,2} where {T <: Real}, face::Int64, ::Grid.Abstract1DElemType)
    dofs[1:2] = FE.grid.nodes4faces[face,:]
    dofs[3:4] = size(FE.grid.coords4nodes,1) .+ dofs[1:2]
    dofs[5] = 2*size(FE.grid.coords4nodes,1) + face
end

# BASIS FUNCTIONS
function get_basis_on_face(FE::H1BRFiniteElement{T,2} where T <: Real, ::Grid.Abstract1DElemType)
    temp = 0.0;
    fb = 0.0;
    function closure(xref)
        temp = 1 - xref[1];
        fb = 4*temp*xref[1];
        return [temp 0.0;
                xref[1] 0.0;
                0.0 temp;
                0.0 xref[1];
                fb fb]
    end            
end

function get_basis_on_cell(FE::H1BRFiniteElement{T,2} where T <: Real, ::Grid.ElemType2DTriangle)
    temp = 0.0;
    fb1 = 0.0;
    fb2 = 0.0;
    fb3 = 0.0;
    function closure(xref)
        temp = 1 - xref[1] - xref[2];
        fb1 = 4*temp*xref[1];
        fb2 = 4*xref[1]*xref[2]
        fb3 = 4*temp*xref[2];
        return [temp 0.0;
                xref[1] 0.0;
                xref[2] 0.0;
                0.0 temp;
                0.0 xref[1];
                0.0 xref[2];
                fb1 fb1;
                fb2 fb2;
                fb3 fb3]
    end
end

function get_basis_on_cell(FE::H1BRFiniteElement{T,2} where T <: Real, ::Grid.ElemType2DParallelogram)
    a = 0.0;
    b = 0.0;
    fb1 = 0.0;
    fb2 = 0.0;
    fb3 = 0.0;
    fb4 = 0.0;
    function closure(xref)
        a = 1 - xref[1]
        b = 1 - xref[2]
        fb1 = xref[1]*a*b
        fb2 = xref[1]*b*xref[2]
        fb3 = xref[1]*xref[2]*a
        fb4 = xref[2]*a*b
        return [a*b 0.0;
                xref[1]*b 0.0;
                xref[1]*xref[2] 0.0;
                xref[2]*a 0.0;
                0.0 a*b;
                0.0 xref[1]*b;
                0.0 xref[1]*xref[2];
                0.0 xref[2]*a;
                fb1 fb1;
                fb2 fb2;
                fb3 fb3;
                fb4 fb4]
    end
end

function get_basis_coefficients_on_cell!(coefficients, FE::H1BRFiniteElement{T,2} where T <: Real, cell::Int64, ::Grid.ElemType2DTriangle)
    # multiplication with normal vectors
    fill!(coefficients,1.0)
    coefficients[7,1] = FE.grid.normal4faces[FE.grid.faces4cells[cell,1],1];
    coefficients[7,2] = FE.grid.normal4faces[FE.grid.faces4cells[cell,1],2];
    coefficients[8,1] = FE.grid.normal4faces[FE.grid.faces4cells[cell,2],1];
    coefficients[8,2] = FE.grid.normal4faces[FE.grid.faces4cells[cell,2],2];
    coefficients[9,1] = FE.grid.normal4faces[FE.grid.faces4cells[cell,3],1];
    coefficients[9,2] = FE.grid.normal4faces[FE.grid.faces4cells[cell,3],2];
end    
function get_basis_coefficients_on_cell!(coefficients, FE::H1BRFiniteElement{T,2} where T <: Real, cell::Int64, ::Grid.ElemType2DParallelogram)
    # multiplication with normal vectors
    fill!(coefficients,1.0)
    coefficients[9,1] = FE.grid.normal4faces[FE.grid.faces4cells[cell,1],1];
    coefficients[9,2] = FE.grid.normal4faces[FE.grid.faces4cells[cell,1],2];
    coefficients[10,1] = FE.grid.normal4faces[FE.grid.faces4cells[cell,2],1];
    coefficients[10,2] = FE.grid.normal4faces[FE.grid.faces4cells[cell,2],2];
    coefficients[11,1] = FE.grid.normal4faces[FE.grid.faces4cells[cell,3],1];
    coefficients[11,2] = FE.grid.normal4faces[FE.grid.faces4cells[cell,3],2];
    coefficients[12,1] = FE.grid.normal4faces[FE.grid.faces4cells[cell,4],1];
    coefficients[12,2] = FE.grid.normal4faces[FE.grid.faces4cells[cell,4],2];
end    


function get_basis_coefficients_on_face!(coefficients, FE::H1BRFiniteElement{T,2} where T <: Real, face::Int64, ::Grid.Abstract1DElemType)
    # multiplication with normal vectors
    fill!(coefficients,1.0)
    coefficients[5,1] = FE.grid.normal4faces[face,1];
    coefficients[5,2] = FE.grid.normal4faces[face,2];
end    


# DISCRETE DIVERGENCE-PRESERVING HDIV-RECONSTRUCTION

function Hdivreconstruction_available(FE::H1BRFiniteElement{T,2} where T <: Real)
    return true
end

function get_Hdivreconstruction_space(FE::H1BRFiniteElement{T,2} where T <: Real, variant::Int = 1)
    if (variant == 1)
        return getRT0FiniteElement(FE.grid)
    elseif (variant == 2)
        return getBDM1FiniteElement(FE.grid)    
    end    
end

function get_Hdivreconstruction_trafo!(T,FE::H1BRFiniteElement{T,2} where T <: Real, FE_hdiv::HdivRT0FiniteElement)
    ensure_length4faces!(FE.grid);
    nfaces = size(FE.grid.nodes4faces,1)
    nnodes = size(FE.grid.coords4nodes,1)
    for face = 1 : nfaces
        # reconstruction coefficients for P1 basis functions
        for k = 1 : 2
            node = FE.grid.nodes4faces[face,k]
            T[node,face] = 1 // 2 * FE.grid.length4faces[face] * FE.grid.normal4faces[face,1]
            T[nnodes+node,face] = 1 // 2 * FE.grid.length4faces[face] * FE.grid.normal4faces[face,2]
        end
        # reconstruction coefficient for quadratic face bubbles
        T[2*nnodes+face,face] = 2 // 3 * FE.grid.length4faces[face]
    end
    return T
end


function get_Hdivreconstruction_trafo!(T,FE::H1BRFiniteElement{T,2} where T <: Real, FE_hdiv::HdivBDM1FiniteElement)
    ensure_length4faces!(FE.grid);
    nfaces = size(FE.grid.nodes4faces,1)
    nnodes = size(FE.grid.coords4nodes,1)
    for face = 1 : nfaces
        # reconstruction coefficients for P1 basis functions
        for k = 1 : 2
            node = FE.grid.nodes4faces[face,k]
            T[node,face] = 1 // 2 * FE.grid.length4faces[face] * FE.grid.normal4faces[face,1]
            T[nnodes+node,face] = 1 // 2 * FE.grid.length4faces[face] * FE.grid.normal4faces[face,2]
            T[node,nfaces+face] = -1 // 6 * FE.grid.length4faces[face] * FE.grid.normal4faces[face,1]
            T[nnodes+node,nfaces+face] = -1 // 6 * FE.grid.length4faces[face] * FE.grid.normal4faces[face,2]
        end
        # reconstruction coefficient for quadratic face bubbles
        T[2*nnodes+face,face] = 2 // 3 * FE.grid.length4faces[face]
    end
    return T
end
