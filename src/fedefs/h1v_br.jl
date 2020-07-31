
"""
$(TYPEDEF)

vector-valued (ncomponents = edim) Bernardi--Raugel element
(first-order polynomials + normal-weighted face bubbles)

allowed ElementGeometries:
- Triangle2D (piecewise linear + normale-weighted face bubbles)
- Quadrilateral2D (Q1 space + normal-weighted face bubbles)
"""
abstract type H1BR{edim} <: AbstractH1FiniteElementWithCoefficients where {edim<:Int} end

get_ncomponents(FEType::Type{<:H1BR}) = FEType.parameters[1]

get_polynomialorder(::Type{<:H1BR{2}}, ::Type{<:Edge1D}) = 2;
get_polynomialorder(::Type{<:H1BR{2}}, ::Type{<:Triangle2D}) = 2;
get_polynomialorder(::Type{<:H1BR{2}}, ::Type{<:Quadrilateral2D}) = 3;
get_polynomialorder(::Type{<:H1BR{3}}, ::Type{<:Triangle2D}) = 3;
get_polynomialorder(::Type{<:H1BR{3}}, ::Type{<:Tetrahedron3D}) = 3;
get_polynomialorder(::Type{<:H1BR{3}}, ::Type{<:Parallelogram2D}) = 4;
get_polynomialorder(::Type{<:H1BR{3}}, ::Type{<:Hexahedron3D}) = 5;


function init!(FES::FESpace{FEType}; dofmap_needed = true) where {FEType <: H1BR}
    ncomponents = get_ncomponents(FEType)
    FES.name = "BR (H1, $(ncomponents)d)"

    # count number of dofs
    nnodes = num_sources(FES.xgrid[Coordinates])
    nfaces = num_sources(FES.xgrid[FaceNodes])
    FES.ndofs = nnodes * ncomponents + nfaces

    # register coefficients
    FES.xFaceNormals = FES.xgrid[FaceNormals]
    FES.xFaceVolumes = FES.xgrid[FaceVolumes]
    FES.xCellFaces = FES.xgrid[CellFaces]
end

function init_dofmap!(FES::FESpace{FEType}, ::Type{AssemblyTypeCELL}) where {FEType <: H1BR}
    xCellNodes = FES.xgrid[CellNodes]
    xCellFaces = FES.xgrid[CellFaces]
    xCellGeometries = FES.xgrid[CellGeometries]
    ncomponents = get_ncomponents(FEType)
    dofs4item = zeros(Int32,ncomponents*(max_num_targets_per_source(xCellNodes))+max_num_targets_per_source(xCellFaces))
    ncells = num_sources(xCellNodes)
    xCellDofs = VariableTargetAdjacency(Int32)
    nnodes = num_sources(FES.xgrid[Coordinates])
    nnodes4item = 0
    for cell = 1 : ncells
        nnodes4item = num_targets(xCellNodes,cell)
        for k = 1 : nnodes4item
            dofs4item[k] = xCellNodes[k,cell]
            for n = 1 : ncomponents-1
                dofs4item[k+n*nnodes4item] = n*nnodes + dofs4item[k]
            end    
        end
        nnodes4item = num_targets(xCellFaces,cell)
        for k = 1 : nnodes4item
            dofs4item[ncomponents*nnodes4item+k] = ncomponents*nnodes + xCellFaces[k,cell]
        end
        append!(xCellDofs,dofs4item[1:(ncomponents+1)*nnodes4item])
    end
    # save dofmap
    FES.CellDofs = xCellDofs
end

function init_dofmap!(FES::FESpace{FEType}, ::Type{AssemblyTypeFACE}) where {FEType <: H1BR}
    xFaceNodes = FES.xgrid[FaceNodes]
    xBFaces = FES.xgrid[BFaces]
    nfaces = num_sources(xFaceNodes)
    xFaceDofs = VariableTargetAdjacency(Int32)
    ncomponents = get_ncomponents(FEType)
    dofs4item = zeros(Int32,ncomponents*max_num_targets_per_source(xFaceNodes)+1)
    nnodes = num_sources(FES.xgrid[Coordinates])
    nnodes4item = 0
    for face = 1 : nfaces
        nnodes4item = num_targets(xFaceNodes,face)
        for k = 1 : nnodes4item
            dofs4item[k] = xFaceNodes[k,face]
            for n = 1 : ncomponents-1
                dofs4item[k+n*nnodes4item] = n*nnodes + dofs4item[k]
            end    
        end
        dofs4item[ncomponents*nnodes4item+1] = ncomponents*nnodes + face
        append!(xFaceDofs,dofs4item[1:ncomponents*nnodes4item+1])
    end
    # save dofmap
    FES.FaceDofs = xFaceDofs
end

function init_dofmap!(FES::FESpace{FEType}, ::Type{AssemblyTypeBFACE}) where {FEType <: H1BR}
    xBFaceNodes = FES.xgrid[BFaceNodes]
    xBFaces = FES.xgrid[BFaces]
    nbfaces = num_sources(xBFaceNodes)
    xBFaceDofs = VariableTargetAdjacency(Int32)
    ncomponents = get_ncomponents(FEType)
    dofs4item = zeros(Int32,ncomponents*max_num_targets_per_source(xBFaceNodes)+1)
    nnodes = num_sources(FES.xgrid[Coordinates])
    nnodes4item = 0
    for bface = 1: nbfaces
        nnodes4item = num_targets(xBFaceNodes,bface)
        for k = 1 : nnodes4item
            dofs4item[k] = xBFaceNodes[k,bface]
            for n = 1 : ncomponents-1
                dofs4item[k+n*nnodes4item] = n*nnodes + dofs4item[k]
            end    
        end
        dofs4item[ncomponents*nnodes4item+1] = ncomponents*nnodes + xBFaces[bface]
        append!(xBFaceDofs,dofs4item[1:ncomponents*nnodes4item+1])
    end
    # save dofmap
    FES.BFaceDofs = xBFaceDofs
end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{<:H1BR}, exact_function!::Function; dofs = [], bonus_quadorder::Int = 0)
    xCoords = FE.xgrid[Coordinates]
    xFaceNodes = FE.xgrid[FaceNodes]
    xFaceNormals = FE.xgrid[FaceNormals]
    FEType = eltype(FE)
    ncomponents = get_ncomponents(FEType)
    result = zeros(Float64,ncomponents)
    xdim = size(xCoords,1)
    nnodes = num_sources(xCoords)
    nfaces = num_sources(xFaceNodes)
    x = zeros(Float64,xdim)
    if length(dofs) == 0 # interpolate at all dofs
        for j = 1 : num_sources(xCoords)
            for k=1:xdim
                x[k] = xCoords[k,j]
            end    
            exact_function!(result,x)
            for n = 1 : ncomponents
                Target[j+(n-1)*nnodes] = result[n]
            end    
        end
        # interpolate at face midpoints
        linpart = zeros(Float64,ncomponents)
        node = 0
        for face = 1 : nfaces
            nnodes4item = num_targets(xFaceNodes,face)
            # compute midpoint
            for k=1:xdim
                x[k] = 0
                for n=1:nnodes4item
                    node = xFaceNodes[n,face]
                    x[k] += xCoords[k,node]
                end
                x[k] /= nnodes4item    
            end    
            # compute linear part
            fill!(linpart,0)
            for n=1:nnodes4item
                node = xFaceNodes[n,face]
                for c=1:ncomponents
                    linpart[c] += Target[node+(c-1)*nnodes] / nnodes4item
                end
            end
            exact_function!(result,x)
            Target[ncomponents*nnodes+face] = 0.0
            for k = 1 : ncomponents
                Target[ncomponents*nnodes+face] = (result[k] - linpart[k]) * xFaceNormals[k,face]
            end    
        end
    else
        item = 0
        for j in dofs 
            item = mod(j-1,nnodes)+1
            c = Int(ceil(j/nnodes))
            if item <= nnodes
                for k=1:xdim
                    x[k] = xCoords[k,item]
                end    
                exact_function!(result,x)
                Target[j] = result[c]
            elseif c > ncomponents
                face = j - ncomponents*nnodes
                nnodes4item = num_targets(xFaceNodes,face)
                for k=1:xdim
                    x[k] = 0
                    for n=1:nnodes4item
                        x[k] += xCoords[k,xFaceNodes[n,face]]
                    end
                    x[k] /= nnodes4item    
                end 
                exact_function!(result,x)
                Target[ncomponents*nnodes+face] = 0.0
                for k = 1 : ncomponents
                    Target[ncomponents*nnodes+face] = result[k]*xFaceNormals[k,face]
                end    
            end
        end    
    end    
end


function nodevalues!(Target::AbstractArray{<:Real,2}, Source::AbstractArray{<:Real,1}, FE::FESpace{<:H1BR})
    nnodes = num_sources(FE.xgrid[Coordinates])
    nfaces = num_sources(FE.xgrid[FaceNodes])
    FEType = eltype(FE)
    ncomponents = get_ncomponents(FEType)
    offset4component = 0:nnodes:ncomponents*nnodes
    for node = 1 : nnodes
        for c = 1 : ncomponents
            Target[c,node] = Source[offset4component[c]+node]
        end    
    end    
end


############
# 2D basis #
############

function get_basis_on_face(::Type{H1BR{2}}, ::Type{<:Edge1D})
    function closure(xref)
        temp = 1 - xref[1];
        bf = 4 * xref[1] * temp;
        return [temp 0.0;
                xref[1] 0.0;
                0.0 temp;
                0.0 xref[1];
                bf bf]
    end
end

function get_basis_on_cell(::Type{H1BR{2}}, ::Type{<:Triangle2D})
    function closure(xref)
        temp = 1 - xref[1] - xref[2];
        bf1 = 4 * xref[1] * temp;
        bf2 = 4 * xref[2] * xref[1];
        bf3 = 4 * temp * xref[2];
        return [temp 0.0;
                xref[1] 0.0;
                xref[2] 0.0;
                0.0 temp;
                0.0 xref[1];
                0.0 xref[2];
                bf1 bf1;
                bf2 bf2;
                bf3 bf3]
    end
end

function get_basis_on_cell(::Type{H1BR{2}}, ::Type{<:Quadrilateral2D})
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

function get_coefficients_on_cell!(coefficients, FE::FESpace{H1BR{2}}, ::Type{<:Triangle2D}, cell::Int)
    # multiplication with normal vectors
    fill!(coefficients,1.0)
    coefficients[1,7] = FE.xFaceNormals[1, FE.xCellFaces[1,cell]];
    coefficients[2,7] = FE.xFaceNormals[2, FE.xCellFaces[1,cell]];
    coefficients[1,8] = FE.xFaceNormals[1, FE.xCellFaces[2,cell]];
    coefficients[2,8] = FE.xFaceNormals[2, FE.xCellFaces[2,cell]];
    coefficients[1,9] = FE.xFaceNormals[1, FE.xCellFaces[3,cell]];
    coefficients[2,9] = FE.xFaceNormals[2, FE.xCellFaces[3,cell]];
end    

function get_coefficients_on_cell!(coefficients, FE::FESpace{H1BR{2}}, ::Type{<:Quadrilateral2D}, cell::Int,)
    # multiplication with normal vectors
    fill!(coefficients,1.0)
    coefficients[1,9] = FE.xFaceNormals[1, FE.xCellFaces[1,cell]];
    coefficients[2,9] = FE.xFaceNormals[2, FE.xCellFaces[1,cell]];
    coefficients[1,10] = FE.xFaceNormals[1, FE.xCellFaces[2,cell]];
    coefficients[2,10] = FE.xFaceNormals[2, FE.xCellFaces[2,cell]];
    coefficients[1,11] = FE.xFaceNormals[1, FE.xCellFaces[3,cell]];
    coefficients[2,11] = FE.xFaceNormals[2, FE.xCellFaces[3,cell]];
    coefficients[1,12] = FE.xFaceNormals[1, FE.xCellFaces[4,cell]];
    coefficients[2,12] = FE.xFaceNormals[2, FE.xCellFaces[4,cell]];
end    


function get_coefficients_on_face!(coefficients, FE::FESpace{H1BR{2}}, ::Type{<:Edge1D}, face::Int)
    # multiplication of face bubble with normal vector of face
    fill!(coefficients,1.0)
    coefficients[1,5] = FE.xFaceNormals[1, face];
    coefficients[2,5] = FE.xFaceNormals[2, face];
end    


############
# 3D basis #
############

function get_basis_on_face(::Type{H1BR{3}}, ::Type{<:Triangle2D})
    function closure(xref)
        temp = 1 - xref[1] - xref[2];
        bf = 9 * xref[1] * temp * xref[2];
        return [temp 0.0 0.0;
                xref[1] 0.0 0.0;
                xref[2] 0.0 0.0;
                0.0 temp 0.0;
                0.0 xref[1] 0.0;
                0.0 xref[2] 0.0;
                0.0 0.0 temp;
                0.0 0.0 xref[1];
                0.0 0.0 xref[2];
                bf bf bf]
    end
end

function get_basis_on_face(::Type{H1BR{3}}, ::Type{<:Quadrilateral2D})
    function closure(xref)
        a = 1 - xref[1]
        b = 1 - xref[2]
        bf = 16 * xref[1] * a * b * xref[2];
        return [a*b 0.0 0.0;
                xref[1]*b 0.0 0.0;
                xref[1]*xref[2] 0.0 0.0;
                xref[2]*a 0.0 0.0;
                0.0 a*b 0.0;
                0.0 xref[1]*b 0.0;
                0.0 xref[1]*xref[2] 0.0;
                0.0 xref[2]*a 0.0;
                0.0 0.0 a*b;
                0.0 0.0 xref[1]*b;
                0.0 0.0 xref[1]*xref[2];
                0.0 0.0 xref[2]*a;
                bf bf bf]
    end
end

function get_basis_on_cell(::Type{H1BR{3}}, ::Type{<:Tetrahedron3D})
    function closure(xref)
        temp = 1 - xref[1] - xref[2] - xref[3];
        bf1 = 9 * xref[1] * temp * xref[2];
        bf2 = 9 * temp * xref[2] * xref[3];
        bf3 = 9 * xref[1] * xref[2] * xref[3];
        bf4 = 9 * temp * xref[2] * xref[3];
        return [temp 0.0 0.0;
                xref[1] 0.0 0.0;
                xref[2] 0.0 0.0;
                xref[3] 0.0 0.0;
                0.0 temp 0.0;
                0.0 xref[1] 0.0;
                0.0 xref[2] 0.0;
                0.0 xref[3] 0.0;
                0.0 0.0 temp;
                0.0 0.0 xref[1];
                0.0 0.0 xref[2];
                0.0 0.0 xref[3];
                bf1 bf1 bf1;
                bf2 bf2 bf2;
                bf3 bf3 bf3;
                bf4 bf4 bf4]
    end
end


function get_basis_on_cell(::Type{H1BR{3}}, ::Type{<:Hexahedron3D})
    function closure(xref)
        a = 1 - xref[1]
        b = 1 - xref[2]
        c = 1 - xref[3]
        fb1 = 16*a*b*xref[1]*xref[2]*c # bottom
        fb2 = 16*a*xref[1]*c*xref[3]*b # front
        fb3 = 16*a*b*c*xref[2]*xref[3] # left
        fb4 = 16*a*xref[1]*c*xref[3]*xref[2] # back
        fb5 = 16*xref[1]*b*c*xref[2]*xref[3] # right
        fb6 = 16*a*b*xref[1]*xref[2]*xref[3] # top
        return [a*b*c 0.0 0.0;
                xref[1]*b*c 0.0 0.0;
                xref[2]*a*c 0.0 0.0;
                xref[3]*a*b 0.0 0.0;
                xref[1]*xref[2]*c 0.0 0.0;
                xref[1]*b*xref[3]c 0.0 0.0;
                a*xref[2]*xref[3] 0.0 0.0;
                xref[1]*xref[2]*xref[3] 0.0 0.0;
                0.0 a*b*c 0.0;
                0.0 xref[1]*b*c 0.0;
                0.0 xref[2]*a*c 0.0;
                0.0 xref[3]*a*b 0.0;
                0.0 xref[1]*xref[2]*c 0.0;
                0.0 xref[1]*b*xref[3] 0.0;
                0.0 a*xref[2]*xref[3] 0.0;
                0.0 xref[1]*xref[2]*xref[3] 0.0;
                0.0 0.0 a*b*c;
                0.0 0.0 xref[1]*b*c;
                0.0 0.0 xref[2]*a*c;
                0.0 0.0 xref[3]*a*b;
                0.0 0.0 xref[1]*xref[2]*c;
                0.0 0.0 xref[1]*b*xref[3];
                0.0 0.0 a*xref[2]*xref[3];
                0.0 0.0 xref[1]*xref[2]*xref[3];
                fb1 fb1 fb1;
                fb2 fb2 fb2;
                fb3 fb3 fb3;
                fb4 fb4 fb4;
                fb5 fb5 fb5;
                fb6 fb6 fb6]
    end
end


function get_coefficients_on_cell!(coefficients, FE::FESpace{H1BR{3}}, ::Type{<:Tetrahedron3D}, cell::Int)
    # multiplication with normal vectors
    fill!(coefficients,1.0)
    coefficients[1,13] = FE.xFaceNormals[1, FE.xCellFaces[1,cell]];
    coefficients[2,13] = FE.xFaceNormals[2, FE.xCellFaces[1,cell]];
    coefficients[3,13] = FE.xFaceNormals[3, FE.xCellFaces[1,cell]];
    coefficients[1,14] = FE.xFaceNormals[1, FE.xCellFaces[2,cell]];
    coefficients[2,14] = FE.xFaceNormals[2, FE.xCellFaces[2,cell]];
    coefficients[3,14] = FE.xFaceNormals[3, FE.xCellFaces[2,cell]];
    coefficients[1,15] = FE.xFaceNormals[1, FE.xCellFaces[3,cell]];
    coefficients[2,15] = FE.xFaceNormals[2, FE.xCellFaces[3,cell]];
    coefficients[3,15] = FE.xFaceNormals[3, FE.xCellFaces[3,cell]];
    coefficients[1,16] = FE.xFaceNormals[1, FE.xCellFaces[4,cell]];
    coefficients[2,16] = FE.xFaceNormals[2, FE.xCellFaces[4,cell]];
    coefficients[3,16] = FE.xFaceNormals[3, FE.xCellFaces[4,cell]];
end    


function get_coefficients_on_face!(coefficients, FE::FESpace{H1BR{3}}, ::Type{<:Triangle2D}, face::Int)
    # multiplication of face bubble with normal vector of face
    fill!(coefficients,1.0)
    coefficients[1,10] = FE.xFaceNormals[1, face];
    coefficients[2,10] = FE.xFaceNormals[2, face];
    coefficients[3,10] = FE.xFaceNormals[3, face];
end    


function get_coefficients_on_cell!(coefficients, FE::FESpace{H1BR{3}}, ::Type{<:Hexahedron3D}, cell::Int)
    # multiplication with normal vectors
    fill!(coefficients,1.0)
    coefficients[1,25] = FE.xFaceNormals[1, FE.xCellFaces[1,cell]];
    coefficients[2,25] = FE.xFaceNormals[2, FE.xCellFaces[1,cell]];
    coefficients[3,25] = FE.xFaceNormals[3, FE.xCellFaces[1,cell]];
    coefficients[1,26] = FE.xFaceNormals[1, FE.xCellFaces[2,cell]];
    coefficients[2,26] = FE.xFaceNormals[2, FE.xCellFaces[2,cell]];
    coefficients[3,26] = FE.xFaceNormals[3, FE.xCellFaces[2,cell]];
    coefficients[1,27] = FE.xFaceNormals[1, FE.xCellFaces[3,cell]];
    coefficients[2,27] = FE.xFaceNormals[2, FE.xCellFaces[3,cell]];
    coefficients[3,27] = FE.xFaceNormals[3, FE.xCellFaces[3,cell]];
    coefficients[1,28] = FE.xFaceNormals[1, FE.xCellFaces[4,cell]];
    coefficients[2,28] = FE.xFaceNormals[2, FE.xCellFaces[4,cell]];
    coefficients[3,28] = FE.xFaceNormals[3, FE.xCellFaces[4,cell]];
    coefficients[1,29] = FE.xFaceNormals[1, FE.xCellFaces[5,cell]];
    coefficients[2,29] = FE.xFaceNormals[2, FE.xCellFaces[5,cell]];
    coefficients[3,29] = FE.xFaceNormals[3, FE.xCellFaces[5,cell]];
    coefficients[1,30] = FE.xFaceNormals[1, FE.xCellFaces[6,cell]];
    coefficients[2,30] = FE.xFaceNormals[2, FE.xCellFaces[6,cell]];
    coefficients[3,30] = FE.xFaceNormals[3, FE.xCellFaces[6,cell]];
end    


function get_coefficients_on_face!(coefficients, FE::FESpace{H1BR{3}}, ::Type{<:Quadrilateral2D}, face::Int)
    # multiplication of face bubble with normal vector of face
    fill!(coefficients,1.0)
    coefficients[1,13] = FE.xFaceNormals[1, face];
    coefficients[2,13] = FE.xFaceNormals[2, face];
    coefficients[3,13] = FE.xFaceNormals[3, face];
end    

######################
# RT0 Reconstruction #
######################



function get_reconstruction_coefficients_on_cell!(coefficients, FE::FESpace{H1BR{2}}, ::Type{HDIVRT0{2}}, ::Type{<:Triangle2D}, cell::Int)
    # reconstruction coefficients for P1 basis functions on reference element
    fill!(coefficients,0.0)
    coefficients[1,3] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[3,cell]] * FE.xFaceNormals[1, FE.xCellFaces[3,cell]]
    coefficients[1,1] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[1,cell]] * FE.xFaceNormals[1, FE.xCellFaces[1,cell]]
    coefficients[2,1] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[1,cell]] * FE.xFaceNormals[1, FE.xCellFaces[1,cell]]
    coefficients[2,2] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[2,cell]] * FE.xFaceNormals[1, FE.xCellFaces[2,cell]]
    coefficients[3,2] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[2,cell]] * FE.xFaceNormals[1, FE.xCellFaces[2,cell]]
    coefficients[3,3] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[3,cell]] * FE.xFaceNormals[1, FE.xCellFaces[3,cell]]
    coefficients[4,3] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[3,cell]] * FE.xFaceNormals[2, FE.xCellFaces[3,cell]]
    coefficients[4,1] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[1,cell]] * FE.xFaceNormals[2, FE.xCellFaces[1,cell]]
    coefficients[5,1] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[1,cell]] * FE.xFaceNormals[2, FE.xCellFaces[1,cell]]
    coefficients[5,2] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[2,cell]] * FE.xFaceNormals[2, FE.xCellFaces[2,cell]]
    coefficients[6,2] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[2,cell]] * FE.xFaceNormals[2, FE.xCellFaces[2,cell]]
    coefficients[6,3] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[3,cell]] * FE.xFaceNormals[2, FE.xCellFaces[3,cell]]
    coefficients[7,1] = 2 // 3 * FE.xFaceVolumes[FE.xCellFaces[1,cell]]
    coefficients[8,2] = 2 // 3 * FE.xFaceVolumes[FE.xCellFaces[2,cell]]
    coefficients[9,3] = 2 // 3 * FE.xFaceVolumes[FE.xCellFaces[3,cell]]
end


function get_reconstruction_coefficients_on_cell!(coefficients, FE::FESpace{H1BR{2}}, ::Type{HDIVRT0{2}}, ::Type{<:Parallelogram2D}, cell::Int)
    # reconstruction coefficients for P1 basis functions on reference element
    fill!(coefficients,0.0)
    coefficients[1,4] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[4,cell]] * FE.xFaceNormals[1, FE.xCellFaces[4,cell]]
    coefficients[1,1] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[1,cell]] * FE.xFaceNormals[1, FE.xCellFaces[1,cell]]
    coefficients[2,1] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[1,cell]] * FE.xFaceNormals[1, FE.xCellFaces[1,cell]]
    coefficients[2,2] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[2,cell]] * FE.xFaceNormals[1, FE.xCellFaces[2,cell]]
    coefficients[3,2] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[2,cell]] * FE.xFaceNormals[1, FE.xCellFaces[2,cell]]
    coefficients[3,3] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[3,cell]] * FE.xFaceNormals[1, FE.xCellFaces[3,cell]]
    coefficients[4,3] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[3,cell]] * FE.xFaceNormals[1, FE.xCellFaces[3,cell]]
    coefficients[4,4] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[4,cell]] * FE.xFaceNormals[1, FE.xCellFaces[4,cell]]
    coefficients[5,4] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[4,cell]] * FE.xFaceNormals[2, FE.xCellFaces[4,cell]]
    coefficients[5,1] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[1,cell]] * FE.xFaceNormals[2, FE.xCellFaces[1,cell]]
    coefficients[6,1] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[1,cell]] * FE.xFaceNormals[2, FE.xCellFaces[1,cell]]
    coefficients[6,2] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[2,cell]] * FE.xFaceNormals[2, FE.xCellFaces[2,cell]]
    coefficients[7,2] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[2,cell]] * FE.xFaceNormals[2, FE.xCellFaces[2,cell]]
    coefficients[7,3] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[3,cell]] * FE.xFaceNormals[2, FE.xCellFaces[3,cell]]
    coefficients[8,3] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[3,cell]] * FE.xFaceNormals[2, FE.xCellFaces[3,cell]]
    coefficients[8,4] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[4,cell]] * FE.xFaceNormals[2, FE.xCellFaces[4,cell]]
    coefficients[9,1] = 2 // 3 * FE.xFaceVolumes[FE.xCellFaces[1,cell]]
    coefficients[10,2] = 2 // 3 * FE.xFaceVolumes[FE.xCellFaces[2,cell]]
    coefficients[11,3] = 2 // 3 * FE.xFaceVolumes[FE.xCellFaces[3,cell]]
    coefficients[12,4] = 2 // 3 * FE.xFaceVolumes[FE.xCellFaces[4,cell]]
end