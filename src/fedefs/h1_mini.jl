
"""
$(TYPEDEF)

Mini finite element (continuous piecewise linear + cell bubbles)

allowed element geometries:
- Triangle2D (linear polynomials + cubic cell bubble)
- Quadrilateral2D (Q1 space + quartic cell bubble)
- Tetrahedron3D (linear polynomials + cubic cell bubble)
"""
abstract type H1MINI{ncomponents,edim} <: AbstractH1FiniteElement where {ncomponents<:Int,edim<:Int} end

get_ncomponents(FEType::Type{<:H1MINI}) = FEType.parameters[1]
get_edim(FEType::Type{<:H1MINI}) = FEType.parameters[2]
get_ndofs_on_face(FEType::Type{<:H1MINI}, EG::Type{<:AbstractElementGeometry}) = nnodes_for_geometry(EG) * FEType.parameters[1]
get_ndofs_on_cell(FEType::Type{<:H1MINI}, EG::Type{<:AbstractElementGeometry}) = (1+nnodes_for_geometry(EG)) * FEType.parameters[1]

get_polynomialorder(::Type{<:H1MINI{2,2}}, ::Type{<:Edge1D}) = 1
get_polynomialorder(::Type{<:H1MINI{2,2}}, ::Type{<:Triangle2D}) = 3;
get_polynomialorder(::Type{<:H1MINI{2,2}}, ::Type{<:Quadrilateral2D}) = 4;

get_polynomialorder(::Type{<:H1MINI{3,3}}, ::Type{<:Triangle2D}) = 1;
get_polynomialorder(::Type{<:H1MINI{3,3}}, ::Type{<:Tetrahedron3D}) = 4;

function init!(FES::FESpace{FEType}) where {FEType <: H1MINI}
    ncomponents = get_ncomponents(FEType)
    name = "MINI"
    for n = 1 : ncomponents-1
        name = name * "xMINI"
    end
    FES.name = name * " (H1)"   

    # count number of dofs
    nnodes = num_sources(FES.xgrid[Coordinates]) 
    ncells = num_sources(FES.xgrid[CellNodes])
    FES.ndofs = (nnodes + ncells) * ncomponents

end


function init_dofmap!(FES::FESpace{FEType}, ::Type{CellDofs}) where {FEType <: H1MINI}
    xCellNodes = FES.xgrid[CellNodes]
    xCellGeometries = FES.xgrid[CellGeometries]
    nnodes = num_sources(FES.xgrid[Coordinates]) 
    ncomponents = get_ncomponents(FEType)
    dofs4item = zeros(Int32,ncomponents*(max_num_targets_per_source(xCellNodes)+1))
    ncells = num_sources(xCellNodes)
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
        for k = 1 : ncomponents
            dofs4item[ncomponents*nnodes4item+k] = ncomponents*nnodes + (k-1)*ncells + cell
        end
        append!(xCellDofs,dofs4item[1:ncomponents*(nnodes4item+1)])
    end
    # save dofmap
    FES.dofmaps[CellDofs] = xCellDofs
end

function init_dofmap!(FES::FESpace{FEType}, ::Type{FaceDofs}) where {FEType <: H1MINI}
    xFaceNodes = FES.xgrid[FaceNodes]
    xBFaces = FES.xgrid[BFaces]
    nnodes = num_sources(FES.xgrid[Coordinates]) 
    nfaces = num_sources(xFaceNodes)
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

function init_dofmap!(FES::FESpace{FEType}, ::Type{BFaceDofs}) where {FEType <: H1MINI}
    xBFaceNodes = FES.xgrid[BFaceNodes]
    nnodes = num_sources(FES.xgrid[Coordinates]) 
    nbfaces = num_sources(xBFaceNodes)
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


function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{<:H1MINI}, exact_function!::Function; dofs = [], bonus_quadorder::Int = 0)
    xCoords = FE.xgrid[Coordinates]
    xdim = size(xCoords,1)
    x = zeros(Float64,xdim)
    nnodes = num_sources(xCoords)
    xCellNodes = FE.xgrid[CellNodes]
    ncells = num_sources(xCellNodes)
    nnodes4item::Int = 0
    FEType = eltype(FE)
    ncomponents = get_ncomponents(FEType)
    result = zeros(Float64,ncomponents)
    linpart = 0.0
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

        for cell=1:ncells
            nnodes4item = num_targets(xCellNodes,cell)
            fill!(x,0.0)
            for j=1:xdim
                for k=1:nnodes4item
                    x[j] += xCoords[j,xCellNodes[k,cell]]
                end
                x[j] /= nnodes4item
            end
            exact_function!(result,x)
            for c = 1 : ncomponents
                linpart = 0.0
                for k=1:nnodes4item
                    linpart += Target[xCellNodes[k,cell]+(c-1)*nnodes]
                end
                Target[(c-1)*ncells+ncomponents*nnodes+cell] = result[c] - linpart / nnodes4item
            end
        end
    else
        item = 0
        for j in dofs 
            if j <= ncomponents*nnodes
                item = mod(j-1,nnodes)+1
                c = Int(ceil(j/nnodes))
                for k=1:xdim
                    x[k] = xCoords[k,item]
                end    
                exact_function!(result,x)
                Target[j] = result[c]
            else # cell bubble
                j = j - ncomponents*nnodes
                cell = mod(j-1,ncells)+1
                c = Int(ceil(j/ncells))
                nnodes4item = num_targets(xCellNodes,cell)
                fill!(x,0.0)
                for j=1:xdim
                    for k=1:nnodes4item
                        x[j] += xCoords[j,xCellNodes[k,cell]]
                    end
                    x[j] /= nnodes4item
                end
                exact_function!(result,x)
                linpart = 0.0
                for k=1:nnodes4item
                    linpart += Target[xCellNodes[k,cell]+(c-1)*nnodes]
                end
                Target[(c-1)*ncells+ncomponents*nnodes+cell] = result[c] - linpart / nnodes4item    
            end    
        end    
    end    
end

function nodevalues!(Target::AbstractArray{<:Real,2}, Source::AbstractArray{<:Real,1}, FE::FESpace{<:H1MINI})
    nnodes = num_sources(FE.xgrid[Coordinates])
    FEType = eltype(FE)
    ncomponents = get_ncomponents(FEType)
    offset4component = 0:nnodes:ncomponents*nnodes
    for node = 1 : nnodes
        for c = 1 : ncomponents
            Target[c,node] = Source[offset4component[c]+node]
        end    
    end    
end


function get_basis_on_face(FEType::Type{<:H1MINI}, EG::Type{<:AbstractElementGeometry})
    ncomponents = get_ncomponents(FEType)
    # same as P1
    return get_basis_on_face(H1P1{ncomponents}, EG)
end

function get_basis_on_cell(FEType::Type{<:H1MINI}, EG::Type{<:Triangle2D})
    ncomponents = get_ncomponents(FEType)
    refbasis_P1 = get_basis_on_cell(H1P1{ncomponents}, EG)
    offset = get_ndofs_on_cell(H1P1{ncomponents}, EG)
    cb = 0.0
    function closure(refbasis, xref)
        refbasis_P1(refbasis, xref)
        # add cell bubbles to P1 basis
        cb = 27*(1-xref[1]-xref[2])*xref[1]*xref[2]
        for k = 1 : ncomponents
            refbasis[offset+k,k] = cb
        end
    end
end

function get_basis_on_cell(FEType::Type{<:H1MINI}, EG::Type{<:Quadrilateral2D})
    ncomponents = get_ncomponents(FEType)
    refbasis_P1 = get_basis_on_cell(H1P1{ncomponents}, EG)
    offset = get_ndofs_on_cell(H1P1{ncomponents}, EG)
    cb = 0.0
    function closure(refbasis, xref)
        refbasis_P1(refbasis, xref)
        # add cell bubbles to P1 basis
        cb = 16*(1-xref[1])*(1-xref[2])*xref[1]*xref[2]
        for k = 1 : ncomponents
            refbasis[offset+k,k] = cb
        end
    end
end

function get_basis_on_cell(FEType::Type{<:H1MINI}, EG::Type{<:Tetrahedron3D})
    ncomponents = get_ncomponents(FEType)
    refbasis_P1 = get_basis_on_cell(H1P1{ncomponents}, EG)
    offset = get_ndofs_on_cell(H1P1{ncomponents}, EG)
    cb = 0.0
    function closure(refbasis, xref)
        refbasis_P1(refbasis, xref)
        # add cell bubbles to P1 basis
        cb = 81*(1-xref[1]-xref[2]-xref[3])*xref[1]*xref[2]*xref[3]
        for k = 1 : ncomponents
            refbasis[offset+k,k] = cb
        end
    end
end