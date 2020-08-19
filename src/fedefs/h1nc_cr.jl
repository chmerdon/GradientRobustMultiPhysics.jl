
"""
$(TYPEDEF)

Crouzeix-Raviart element (only continuous at face centers)

allowed ElementGeometries:
- Triangle2D (piecewise linear, similar to P1)
- Quadrilateral2D (similar to Q1 space)
"""
abstract type H1CR{ncomponents} <: AbstractH1FiniteElement where {ncomponents<:Int} end


get_ncomponents(FEType::Type{<:H1CR}) = FEType.parameters[1]

get_polynomialorder(::Type{<:H1CR}, ::Type{<:Edge1D}) = 0;
get_polynomialorder(::Type{<:H1CR}, ::Type{<:Triangle2D}) = 1;
get_polynomialorder(::Type{<:H1CR}, ::Type{<:Quadrilateral2D}) = 2;
get_polynomialorder(::Type{<:H1CR}, ::Type{<:Tetrahedron3D}) = 1;



function init!(FES::FESpace{FEType}; dofmap_needed = true) where {FEType <: H1CR}
    ncomponents = get_ncomponents(FEType)
    name = "CR"
    for n = 1 : ncomponents-1
        name = name * "xCR"
    end
    FES.name = name * " (H1nc)"   

    # count number of dofs
    nfaces = num_sources(FES.xgrid[FaceNodes])
    FES.ndofs = nfaces * ncomponents

    # register coefficients (needed in case of reconstruction)
    FES.xFaceNormals = FES.xgrid[FaceNormals]
    FES.xFaceVolumes = FES.xgrid[FaceVolumes]
    FES.xCellFaces = FES.xgrid[CellFaces]
end

function init_dofmap!(FES::FESpace{FEType}, ::Type{AssemblyTypeCELL}) where {FEType <: H1CR}
    xCellFaces = FES.xgrid[CellFaces]
    xCellGeometries = FES.xgrid[CellGeometries]
    ncomponents = get_ncomponents(FEType)
    dofs4item = zeros(Int32,ncomponents*max_num_targets_per_source(xCellFaces))
    ncells = num_sources(xCellFaces)
    nfaces = num_sources(FES.xgrid[FaceNodes])
    xCellDofs = VariableTargetAdjacency(Int32)
    nfaces4item = 0
    for cell = 1 : ncells
        nfaces4item = num_targets(xCellFaces,cell)
        for k = 1 : nfaces4item, c = 1 : ncomponents
            dofs4item[k+(c-1)*nfaces4item] = xCellFaces[k,cell] + (c-1)*nfaces
        end
        append!(xCellDofs,dofs4item[1:nfaces4item*ncomponents])
    end
    # save dofmap
    FES.CellDofs = xCellDofs
end

function init_dofmap!(FES::FESpace{FEType}, ::Type{AssemblyTypeFACE}) where {FEType <: H1CR}
    xBFaces = FES.xgrid[BFaces]
    nfaces = num_sources(FES.xgrid[FaceNodes])
    xFaceDofs = VariableTargetAdjacency(Int32)
    ncomponents = get_ncomponents(FEType)
    dofs4item = zeros(Int32,ncomponents)
    for face = 1 : nfaces
        for c = 1 : ncomponents
            dofs4item[c] = face + (c-1)*nfaces
        end
        append!(xFaceDofs,dofs4item[1:ncomponents])
    end
    # save dofmap
    FES.FaceDofs = xFaceDofs
end

function init_dofmap!(FES::FESpace{FEType}, ::Type{AssemblyTypeBFACE}) where {FEType <: H1CR}
    xBFaceNodes = FES.xgrid[BFaceNodes]
    xBFaces = FES.xgrid[BFaces]
    nfaces = num_sources(FES.xgrid[FaceNodes])
    nbfaces = num_sources(xBFaceNodes)
    xBFaceDofs = VariableTargetAdjacency(Int32)
    ncomponents = get_ncomponents(FEType)
    dofs4item = zeros(Int32,ncomponents)
    for bface = 1: nbfaces
        for c = 1 : ncomponents
            dofs4item[c] = xBFaces[bface] + (c-1)*nfaces
        end
        append!(xBFaceDofs,dofs4item[1:ncomponents])
    end
    # save dofmap
    FES.BFaceDofs = xBFaceDofs
end


function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{<:H1CR}, exact_function!::Function; dofs = [], bonus_quadorder::Int = 0)
    xCoords = FE.xgrid[Coordinates]
    xdim = size(xCoords,1)
    x = zeros(Float64,xdim)
    xFaceNodes = FE.xgrid[FaceNodes]
    nfaces = num_sources(xFaceNodes)
    nnodes4item::Int = 0
    FEType = eltype(FE)
    ncomponents::Int = get_ncomponents(FEType)
    result = zeros(Float64,ncomponents)
    if length(dofs) == 0 # interpolate at all dofs
        for face = 1 : nfaces
            #compute face midpoints
            fill!(x,0.0)
            nnodes4item = num_targets(xFaceNodes,face)
            for j=1:xdim
                for k=1:nnodes4item
                    x[j] += xCoords[j,xFaceNodes[k,face]]
                end
                x[j] /= nnodes4item
            end
            exact_function!(result,x)
            for c = 1 : ncomponents
                Target[face+(c-1)*nfaces] = result[c]
            end
        end
    else
        face = 0
        for j in dofs 
            face = mod(j-1,nfaces)+1
            c = Int(ceil(j/nfaces))
            #compute face midpoints
            fill!(x,0.0)
            nnodes4item = num_targets(xFaceNodes,face)
            for j=1:xdim
                for k=1:nnodes4item
                    x[j] += xCoords[j,xFaceNodes[k,face]]
                end
                x[j] /= nnodes4item
            end
            exact_function!(result,x)
            Target[j] = result[c]
        end    
    end    
end

# BEWARE
#
# all basis functions on a cell are nonzero on all edges,
# but only the dof associated to the face is eveluated
# when using get_basis_on_face
# this leads to lumped bestapproximations along the boundary for example

function get_basis_on_face(FEType::Type{<:H1CR}, ::Type{<:AbstractElementGeometry})
    function closure(xref)
        ncomponents = get_ncomponents(FEType)
        refbasis = zeros(eltype(xref),ncomponents,ncomponents)
        for k = 1 : ncomponents
            refbasis[k,k] = 1
        end
        return refbasis
    end
end

function get_basis_on_cell(FEType::Type{<:H1CR}, ET::Type{<:Triangle2D})
    function closure(xref)
        ncomponents = get_ncomponents(FEType)
        refbasis = zeros(eltype(xref),ncomponents*3,ncomponents)
        temp = 2*(xref[1]+xref[2]) - 1
        for k = 1 : ncomponents
            refbasis[3*k-2,k] = 1 - 2*xref[2]
            refbasis[3*k-1,k] = temp
            refbasis[3*k,k] = 1 - 2*xref[1]
        end
        return refbasis
    end
end

function get_basis_on_cell(FEType::Type{<:H1CR}, ET::Type{<:Tetrahedron3D})
    function closure(xref)
        ncomponents = get_ncomponents(FEType)
        refbasis = zeros(eltype(xref),ncomponents*4,ncomponents)
        temp = 3*(xref[1]+xref[2]+xref[3]) - 2
        for k = 1 : ncomponents
            refbasis[4*k-3,k] = 1 - 3*xref[3]
            refbasis[4*k-2,k] = 1 - 3*xref[2]
            refbasis[4*k-1,k] = temp
            refbasis[4*k,k] = 1 - 3*xref[1]
        end
        return refbasis
    end
end

function get_basis_on_cell(FEType::Type{<:H1CR}, ET::Type{<:Quadrilateral2D})
    function closure(xref)
        ncomponents = get_ncomponents(FEType)
        refbasis = zeros(eltype(xref),ncomponents*4,ncomponents)
        a = 1 - xref[1]
        b = 1 - xref[2]
        temp = xref[1]*a + b*b - 1//4
        temp2 = xref[2]*b + xref[1]*xref[1] - 1//4
        temp3 = xref[1]*a + xref[2]*xref[2] - 1//4
        temp4 = xref[2]*b + a*a - 1//4
        for k = 1 : ncomponents
            refbasis[4*k-3,k] = temp
            refbasis[4*k-2,k] = temp2
            refbasis[4*k-1,k] = temp3
            refbasis[4*k,k] = temp4
        end
        return refbasis
    end
end

function get_reconstruction_coefficients_on_cell!(coefficients, FE::FESpace{H1CR{2}}, ::FESpace{HDIVRT0{2}}, ::Type{<:Triangle2D}, cell::Int)
    # reconstruction coefficients for P1 basis functions on reference element
    fill!(coefficients,0.0)
    coefficients[1,1] = FE.xFaceVolumes[FE.xCellFaces[1,cell]] * FE.xFaceNormals[1, FE.xCellFaces[1,cell]]
    coefficients[4,1] = FE.xFaceVolumes[FE.xCellFaces[1,cell]] * FE.xFaceNormals[2, FE.xCellFaces[1,cell]]

    coefficients[2,2] = FE.xFaceVolumes[FE.xCellFaces[2,cell]] * FE.xFaceNormals[1, FE.xCellFaces[2,cell]]
    coefficients[5,2] = FE.xFaceVolumes[FE.xCellFaces[2,cell]] * FE.xFaceNormals[2, FE.xCellFaces[2,cell]]
    
    coefficients[3,3] = FE.xFaceVolumes[FE.xCellFaces[3,cell]] * FE.xFaceNormals[1, FE.xCellFaces[3,cell]]
    coefficients[6,3] = FE.xFaceVolumes[FE.xCellFaces[3,cell]] * FE.xFaceNormals[2, FE.xCellFaces[3,cell]]
end

function get_reconstruction_coefficients_on_cell!(coefficients, FE::FESpace{H1CR{3}}, ::FESpace{HDIVRT0{3}}, ::Type{<:Tetrahedron3D}, cell::Int)
    # reconstruction coefficients for P1 basis functions on reference element
    fill!(coefficients,0.0)
    coefficients[1,1] = FE.xFaceVolumes[FE.xCellFaces[1,cell]] * FE.xFaceNormals[1, FE.xCellFaces[1,cell]]
    coefficients[5,1] = FE.xFaceVolumes[FE.xCellFaces[1,cell]] * FE.xFaceNormals[2, FE.xCellFaces[1,cell]]
    coefficients[9,1] = FE.xFaceVolumes[FE.xCellFaces[1,cell]] * FE.xFaceNormals[3, FE.xCellFaces[1,cell]]

    coefficients[2,2] = FE.xFaceVolumes[FE.xCellFaces[2,cell]] * FE.xFaceNormals[1, FE.xCellFaces[2,cell]]
    coefficients[6,2] = FE.xFaceVolumes[FE.xCellFaces[2,cell]] * FE.xFaceNormals[2, FE.xCellFaces[2,cell]]
    coefficients[10,2] = FE.xFaceVolumes[FE.xCellFaces[2,cell]] * FE.xFaceNormals[3, FE.xCellFaces[2,cell]]

    coefficients[3,3] = FE.xFaceVolumes[FE.xCellFaces[3,cell]] * FE.xFaceNormals[1, FE.xCellFaces[3,cell]]
    coefficients[7,3] = FE.xFaceVolumes[FE.xCellFaces[3,cell]] * FE.xFaceNormals[2, FE.xCellFaces[3,cell]]
    coefficients[11,3] = FE.xFaceVolumes[FE.xCellFaces[3,cell]] * FE.xFaceNormals[3, FE.xCellFaces[3,cell]]

    coefficients[4,4] = FE.xFaceVolumes[FE.xCellFaces[4,cell]] * FE.xFaceNormals[1, FE.xCellFaces[4,cell]]
    coefficients[8,4] = FE.xFaceVolumes[FE.xCellFaces[4,cell]] * FE.xFaceNormals[2, FE.xCellFaces[4,cell]]
    coefficients[12,4] = FE.xFaceVolumes[FE.xCellFaces[4,cell]] * FE.xFaceNormals[3, FE.xCellFaces[4,cell]]
end


function get_reconstruction_coefficients_on_cell!(coefficients, FE::FESpace{H1CR{2}}, ::FESpace{HDIVRT0{2}}, ::Type{<:Parallelogram2D}, cell::Int)
    # reconstruction coefficients for P1 basis functions on reference element
    fill!(coefficients,0.0)
    coefficients[1,1] = FE.xFaceVolumes[FE.xCellFaces[1,cell]] * FE.xFaceNormals[1, FE.xCellFaces[1,cell]]
    coefficients[2,2] = FE.xFaceVolumes[FE.xCellFaces[2,cell]] * FE.xFaceNormals[1, FE.xCellFaces[2,cell]]
    coefficients[3,3] = FE.xFaceVolumes[FE.xCellFaces[3,cell]] * FE.xFaceNormals[1, FE.xCellFaces[3,cell]]
    coefficients[4,4] = FE.xFaceVolumes[FE.xCellFaces[4,cell]] * FE.xFaceNormals[1, FE.xCellFaces[4,cell]]
    coefficients[5,1] = FE.xFaceVolumes[FE.xCellFaces[1,cell]] * FE.xFaceNormals[2, FE.xCellFaces[1,cell]]
    coefficients[6,2] = FE.xFaceVolumes[FE.xCellFaces[2,cell]] * FE.xFaceNormals[2, FE.xCellFaces[2,cell]]
    coefficients[7,3] = FE.xFaceVolumes[FE.xCellFaces[3,cell]] * FE.xFaceNormals[2, FE.xCellFaces[3,cell]]
    coefficients[8,4] = FE.xFaceVolumes[FE.xCellFaces[4,cell]] * FE.xFaceNormals[2, FE.xCellFaces[4,cell]]
end

