
"""
$(TYPEDEF)

Continuous piecewise first-order polynomials with arbitrary number of components.

allowed ElementGeometries:
- Edge1D (linear polynomials)
- Triangle2D (linear polynomials)
- Quadrilateral2D (Q1 space)
- Tetrahedron3D (linear polynomials)
- Hexahedron3D (Q1 space)
"""
abstract type H1P1{ncomponents} <: AbstractH1FiniteElement where {ncomponents<:Int} end

get_ncomponents(FEType::Type{<:H1P1}) = FEType.parameters[1] # is this okay?

get_polynomialorder(::Type{<:H1P1}, ::Type{<:Edge1D}) = 1;
get_polynomialorder(::Type{<:H1P1}, ::Type{<:Triangle2D}) = 1;
get_polynomialorder(::Type{<:H1P1}, ::Type{<:Tetrahedron3D}) = 1;
get_polynomialorder(::Type{<:H1P1}, ::Type{<:Quadrilateral2D}) = 2;
get_polynomialorder(::Type{<:H1P1}, ::Type{<:Hexahedron3D}) = 3;


function init!(FES::FESpace{FEType}) where {FEType <: H1P1}
    ncomponents = get_ncomponents(FEType)
    name = "P1"
    for n = 1 : ncomponents-1
        name = name * "xP1"
    end
    FES.name = name * " (H1)"   

    # count number of dofs
    nnodes = num_sources(FES.xgrid[Coordinates]) 
    FES.ndofs = nnodes * ncomponents
end

function init_dofmap!(FES::FESpace{FEType}, ::Type{CellDofs}) where {FEType <: H1P1}
    xCellNodes = FES.xgrid[CellNodes]
    xCellGeometries = FES.xgrid[CellGeometries]
    ncomponents = get_ncomponents(FEType)
    dofs4item = zeros(Int32,ncomponents*max_num_targets_per_source(xCellNodes))
    ncells = num_sources(xCellNodes)
    nnodes = num_sources(FES.xgrid[Coordinates]) 
    xCellDofs = VariableTargetAdjacency(Int32)
    nnodes4item = 0
    for cell = 1 : ncells
        nnodes4item = num_targets(xCellNodes,cell)
        for k = 1 : nnodes4item
            dofs4item[k] = xCellNodes[k,cell]
            for n = 1 : ncomponents-1
                dofs4item[k+n*nnodes4item] = n*nnodes + dofs4item[k]
            end    
        end
        append!(xCellDofs,dofs4item[1:ncomponents*nnodes4item])
    end
    # save dofmap
    FES.dofmaps[CellDofs] = xCellDofs
end

function init_dofmap!(FES::FESpace{FEType}, ::Type{FaceDofs}) where {FEType <: H1P1}
    xFaceNodes = FES.xgrid[FaceNodes]
    xBFaces = FES.xgrid[BFaces]
    nfaces = num_sources(xFaceNodes)
    nnodes = num_sources(FES.xgrid[Coordinates]) 
    xFaceDofs = VariableTargetAdjacency(Int32)
    ncomponents = get_ncomponents(FEType)
    dofs4item = zeros(Int32,ncomponents*max_num_targets_per_source(xFaceNodes))
    nnodes4item = 0
    for face = 1 : nfaces
        nnodes4item = num_targets(xFaceNodes,face)
        for k = 1 : nnodes4item
            dofs4item[k] = xFaceNodes[k,face]
            for n = 1 : ncomponents-1
                dofs4item[k+n*nnodes4item] = n*nnodes + dofs4item[k]
            end    
        end
        append!(xFaceDofs,dofs4item[1:ncomponents*nnodes4item])
    end
    # save dofmap
    FES.dofmaps[FaceDofs] = xFaceDofs
end

function init_dofmap!(FES::FESpace{FEType}, ::Type{BFaceDofs}) where {FEType <: H1P1}
    xBFaceNodes = FES.xgrid[BFaceNodes]
    nbfaces = num_sources(xBFaceNodes)
    nnodes = num_sources(FES.xgrid[Coordinates]) 
    xBFaceDofs = VariableTargetAdjacency(Int32)
    ncomponents = get_ncomponents(FEType)
    dofs4item = zeros(Int32,ncomponents*max_num_targets_per_source(xBFaceNodes))
    nnodes4item = 0
    for bface = 1: nbfaces
        nnodes4item = num_targets(xBFaceNodes,bface)
        for k = 1 : nnodes4item
            dofs4item[k] = xBFaceNodes[k,bface]
            for n = 1 : ncomponents-1
                dofs4item[k+n*nnodes4item] = n*nnodes + dofs4item[k]
            end    
        end
        append!(xBFaceDofs,dofs4item[1:ncomponents*nnodes4item])
    end
    # save dofmap
    FES.dofmaps[BFaceDofs] = xBFaceDofs
end


function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{<:H1P1}, exact_function!::Function; dofs = [], bonus_quadorder::Int = 0)
    xCoords = FE.xgrid[Coordinates]
    xdim = size(xCoords,1)
    x = zeros(Float64,xdim)
    nnodes = num_sources(xCoords)
    xCellNodes = FE.xgrid[CellNodes]
    ncells = num_sources(xCellNodes)
    nnodes4item::Int = 0
    FEType = eltype(FE)
    ncomponents::Int = get_ncomponents(FEType)
    result = zeros(Float64,ncomponents)
    if length(dofs) == 0 # interpolate at all dofs
        for j = 1 : num_sources(xCoords)
            for k=1:xdim
                x[k] = xCoords[k,j]
            end    
            exact_function!(result,x)
            for c = 1 : ncomponents
                Target[j+(c-1)*nnodes] = result[c]
            end
        end
    else
        item = 0
        for j in dofs 
            item = mod(j-1,nnodes)+1
            c = Int(ceil(j/nnodes))
            for k=1:xdim
                x[k] = xCoords[k,item]
            end    
            exact_function!(result,x)
            Target[j] = result[c]
        end    
    end    
end

function nodevalues!(Target::AbstractArray{<:Real,2}, Source::AbstractArray{<:Real,1}, FE::FESpace{<:H1P1})
    nnodes = num_sources(FE.xgrid[Coordinates])
    FEType = eltype(FE)
    ncomponents::Int = get_ncomponents(FEType)
    offset4component = 0:nnodes:ncomponents*nnodes
    for node = 1 : nnodes
        for c = 1 : ncomponents
            Target[c,node] = Source[offset4component[c]+node]
        end    
    end    
end

function get_basis_on_cell(FEType::Type{<:H1P1}, ET::Type{<:Union{Vertex0D,AbstractElementGeometry1D,Triangle2D,Tetrahedron3D}})
    function closure(xref)
        ncomponents = get_ncomponents(FEType)
        edim = dim_element(ET)
        refbasis = zeros(eltype(xref),ncomponents*(edim+1),ncomponents)
        temp = 1 - sum(xref[1:edim])
        for k = 1 : ncomponents
            refbasis[(edim+1)*k-edim,k] = temp
            for j = 1 : edim
                refbasis[(edim+1)*k-edim+j,k] = xref[j]
            end
        end
        return refbasis
    end
end

function get_basis_on_cell(FEType::Type{<:H1P1}, ::Type{<:Quadrilateral2D})
    function closure(xref)
        ncomponents = get_ncomponents(FEType)
        refbasis = zeros(eltype(xref),ncomponents*4,ncomponents)
        a = 1 - xref[1]
        b = 1 - xref[2]
        for k = 1 : ncomponents
            refbasis[4*k-3,k] = a*b
            refbasis[4*k-2,k] = xref[1]*b
            refbasis[4*k-1,k] = xref[1]*xref[2]
            refbasis[4*k,k]   = xref[2]*a
        end
        return refbasis
    end
end


function get_basis_on_cell(FEType::Type{<:H1P1}, ::Type{<:Hexahedron3D})
    function closure(xref)
        ncomponents = get_ncomponents(FEType)
        refbasis = zeros(eltype(xref),ncomponents*8,ncomponents)
        a = 1 - xref[1]
        b = 1 - xref[2]
        c = 1 - xref[3]
        for k = 1 : ncomponents
            refbasis[8*k-7,k] = a*b*c
            refbasis[8*k-6,k] = xref[1]*b*c 
            refbasis[8*k-5,k] = xref[1]*xref[2]*c
            refbasis[8*k-4,k] = xref[2]*a*c
            refbasis[8*k-3,k] = xref[3]*a*b
            refbasis[8*k-2,k] = xref[1]*b*xref[3]
            refbasis[8*k-1,k] = xref[1]*xref[2]*xref[3]
            refbasis[8*k,k] = a*xref[2]*xref[3]
        end
        return refbasis
    end
end

function get_basis_on_face(FE::Type{<:H1P1}, EG::Type{<:AbstractElementGeometry})
    function closure(xref)
        return get_basis_on_cell(FE, EG)(xref)
    end    
end
