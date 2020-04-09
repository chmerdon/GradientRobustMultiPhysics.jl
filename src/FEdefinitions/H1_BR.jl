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


function get_xref4dof(FE::H1BRFiniteElement{T,2} where {T <: Real}, ::Grid.Abstract2DQuadrilateral) 
    xref = Array{Array{Float64,1},1}(undef,12)
    xref[1] = Array{Float64,1}([0, 0])
    xref[2] = Array{Float64,1}([1, 0])
    xref[3] = Array{Float64,1}([1, 1])
    xref[4] = Array{Float64,1}([0, 1])
    xref[[5,6,7,8]] = xref[[1,2,3,4]]
    xref[9] = Array{Float64,1}([0.5, 0])
    xref[10] = Array{Float64,1}([1, 0.5])
    xref[11] = Array{Float64,1}([0.5, 1])
    xref[12] = Array{Float64,1}([0, 0.5])
    InterpolationMatrix = zeros(Float64,12,12)
    InterpolationMatrix[1:4,1:4] = Matrix{Float64}(I,4,4)
    InterpolationMatrix[[1,2,3,4,9],9] = [-1//2, -1//2, 0, 0, 1]
    InterpolationMatrix[[1,2,3,4,10],10] = [0, -1//2, -1//2, 0, 1]
    InterpolationMatrix[[1,2,3,4,11],11] = [0, 0, -1//2, -1//2, 1]
    InterpolationMatrix[[1,2,3,4,12],12] = [-1//2, 0, 0, -1//2, 1]
    InterpolationMatrix2 = zeros(Float64,12,12)
    InterpolationMatrix2[5:8,5:8] = Matrix{Float64}(I,4,4)
    InterpolationMatrix2[[5,6,7,8,9],9] = [-1//2, -1//2, 0, 0, 1]
    InterpolationMatrix2[[5,6,7,8,10],10] = [0, -1//2, -1//2, 0, 1]
    InterpolationMatrix2[[5,6,7,8,11],11] = [0, 0, -1//2, -1//2, 1]
    InterpolationMatrix2[[5,6,7,8,12],12] = [-1//2, 0, 0, -1//2, 1]
    return xref, [sparse(InterpolationMatrix), sparse(InterpolationMatrix2)]
end   

# POLYNOMIAL ORDER
get_polynomial_order(FE::H1BRFiniteElement) = typeof(FE.grid.elemtypes[1]) <: Grid.Abstract2DQuadrilateral ? 3 : 2;

# TOTAL NUMBER OF DOFS
get_ndofs(FE::H1BRFiniteElement{T,2} where {T <: Real}) = 2*size(FE.grid.coords4nodes,1) + size(FE.grid.nodes4faces,1);

# NUMBER OF DOFS ON ELEMTYPE
get_ndofs4elemtype(FE::H1BRFiniteElement{T,2} where {T <: Real}, ::Grid.Abstract1DElemType) = 5
get_ndofs4elemtype(FE::H1BRFiniteElement{T,2} where {T <: Real}, ::Grid.ElemType2DTriangle) = 9
get_ndofs4elemtype(FE::H1BRFiniteElement{T,2} where {T <: Real}, ::Grid.Abstract2DQuadrilateral) = 12

# NUMBER OF COMPONENTS
get_ncomponents(FE::H1BRFiniteElement{T,2} where {T <: Real}) = 2

# LOCAL DOF TO GLOBAL DOF ON CELL
function get_dofs_on_cell!(dofs,FE::H1BRFiniteElement{T,2} where {T <: Real}, cell::Int64, ::Grid.ElemType2DTriangle)
    dofs[1:3] = FE.grid.nodes4cells[cell,:]
    dofs[4:6] = size(FE.grid.coords4nodes,1) .+ dofs[1:3]
    dofs[7:9] = 2*size(FE.grid.coords4nodes,1) .+ FE.grid.faces4cells[cell,:]
end
function get_dofs_on_cell!(dofs,FE::H1BRFiniteElement{T,2} where {T <: Real}, cell::Int64, ::Grid.Abstract2DQuadrilateral)
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

function get_basis_on_cell(FE::H1BRFiniteElement{T,2} where T <: Real, ::Grid.Abstract2DQuadrilateral)
    a = 0.0;
    b = 0.0;
    fb1 = 0.0;
    fb2 = 0.0;
    fb3 = 0.0;
    fb4 = 0.0;
    function closure(xref)
        a = 1 - xref[1]
        b = 1 - xref[2]
        fb1 = 4*xref[1]*a*b
        fb2 = 4*xref[2]*xref[1]*b
        fb3 = 4*xref[1]*xref[2]*a
        fb4 = 4*xref[2]*a*b
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
function get_basis_coefficients_on_cell!(coefficients, FE::H1BRFiniteElement{T,2} where T <: Real, cell::Int64, ::Grid.Abstract2DQuadrilateral)
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

function get_Hdivreconstruction_space(FE::H1BRFiniteElement{T,2} where T <: Real, ::Grid.ElemType2DTriangle, variant::Int = 1)
    if (variant == 1)
        return getRT0FiniteElement(FE.grid)
    elseif (variant == 2)
        return getBDM1FiniteElement(FE.grid)    
    end    
end

function get_Hdivreconstruction_space(FE::H1BRFiniteElement{T,2} where T <: Real, ::Grid.Abstract2DQuadrilateral, variant::Int = 1)
    if (variant == 1)
        return getRT0FiniteElement(FE.grid)
    elseif (variant == 2)
        return getABF0FiniteElement(FE.grid)    
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


# ABF0 reconstruction not working properly yet!
# coefficients for interior ABF0 functions seem not correct
function get_Hdivreconstruction_trafo!(T,FE::H1BRFiniteElement{T,2} where T <: Real, FE_hdiv::HdivABF0FiniteElement)
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
        # reconstruction coefficient for cubic face bubbles
        # (at the boundary they are quadratic like the triangular face bubbles)
        T[2*nnodes+face,face] = 2 // 3 * FE.grid.length4faces[face]
    end

    # coefficients for interior ABF dofs
    # = first moments of the divergence (integral of div(v)x and div(v)y)
    a = 0.0
    b = 0.0
    face = 0
    node = 0
    c1 = [-1//4, 1//4,1//4,-1//4]
    c2 = [-1//6, 1//6,1//3,-1//3]
    c3 = [-1//6,-1//3,1//3, 1//6]
    c4 = [-1//4,-1//4,1//4, 1//4]
    for cell = 1 : size(FE.grid.nodes4cells,1)
        # Q1 basis functions
        for k = 1 : 4
            node = FE.grid.nodes4cells[cell,k]
            T[node,nfaces+2*cell-1]          =  c1[k] .* FE.grid.volume4cells[cell]
            T[node,nfaces+2*cell]            =  c2[k] .* FE.grid.volume4cells[cell]
            T[nnodes + node,nfaces+2*cell-1] =  c3[k] .* FE.grid.volume4cells[cell]
            T[nnodes + node,nfaces+2*cell]   =  c4[k] .* FE.grid.volume4cells[cell]
            
        end
        # face bubble 1
        face = FE.grid.faces4cells[cell,1]
        a = FE.grid.normal4faces[face,1] * FE.grid.volume4cells[cell]
        b = FE.grid.normal4faces[face,2] * FE.grid.volume4cells[cell]
        T[2*nnodes+face,nfaces+2*cell-1] = -1 // 3 * (a + b)
        T[2*nnodes+face,nfaces+2*cell] = - 1 // 3 * b
        
        # face bubble 2
        face = FE.grid.faces4cells[cell,2]
        a = FE.grid.normal4faces[face,1] * FE.grid.volume4cells[cell]
        b = FE.grid.normal4faces[face,2] * FE.grid.volume4cells[cell]
        T[2*nnodes+face,nfaces+2*cell-1] = 1 // 3 * a
        T[2*nnodes+face,nfaces+2*cell] = 1 // 3 * (a-b)
        
        # face bubble 3
        face = FE.grid.faces4cells[cell,3]
        a = FE.grid.normal4faces[face,1] * FE.grid.volume4cells[cell]
        b = FE.grid.normal4faces[face,2] * FE.grid.volume4cells[cell]
        T[2*nnodes+face,nfaces+2*cell-1] = 1 // 3 * (b-a)
        T[2*nnodes+face,nfaces+2*cell] = 1 // 3 * b

        # face bubble 4
        face = FE.grid.faces4cells[cell,4]
        a = FE.grid.normal4faces[face,1] * FE.grid.volume4cells[cell]
        b = FE.grid.normal4faces[face,2] * FE.grid.volume4cells[cell]
        T[2*nnodes+face,nfaces+2*cell-1] = -1 // 3 * a
        T[2*nnodes+face,nfaces+2*cell] = -1 // 3 * (a+b)
    end
    return T
end