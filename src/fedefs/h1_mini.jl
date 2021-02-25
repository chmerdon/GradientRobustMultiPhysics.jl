
"""
````
abstract type H1MINI{ncomponents,edim} <: AbstractH1FiniteElement where {ncomponents<:Int,edim<:Int}
````

Mini finite element.

allowed element geometries:
- Triangle2D (linear polynomials + cubic cell bubble)
- Quadrilateral2D (Q1 space + quartic cell bubble)
- Tetrahedron3D (linear polynomials + cubic cell bubble)
"""
abstract type H1MINI{ncomponents,edim} <: AbstractH1FiniteElement where {ncomponents<:Int,edim<:Int} end

get_ncomponents(FEType::Type{<:H1MINI}) = FEType.parameters[1]
get_edim(FEType::Type{<:H1MINI}) = FEType.parameters[2]
get_ndofs(::Union{Type{<:ON_FACES}, Type{<:ON_BFACES}}, FEType::Type{<:H1MINI}, EG::Type{<:AbstractElementGeometry}) = nnodes_for_geometry(EG) * FEType.parameters[1]
get_ndofs(::Type{<:ON_CELLS}, FEType::Type{<:H1MINI}, EG::Type{<:AbstractElementGeometry}) = (1+nnodes_for_geometry(EG)) * FEType.parameters[1]

get_polynomialorder(FEType::Type{<:H1MINI}, ::Type{<:Edge1D}) = FEType.parameters[2] == 1 ? 2 : 1
get_polynomialorder(FEType::Type{<:H1MINI}, ::Type{<:Triangle2D}) = FEType.parameters[2] == 2 ? 3 : 1;
get_polynomialorder(FEType::Type{<:H1MINI}, ::Type{<:Quadrilateral2D}) = FEType.parameters[2] == 2 ? 4 : 2;
get_polynomialorder(FEType::Type{<:H1MINI}, ::Type{<:Tetrahedron3D}) = 4;

get_dofmap_pattern(FEType::Type{<:H1MINI}, ::Type{CellDofs}, EG::Type{<:AbstractElementGeometry}) = "N1I1"
get_dofmap_pattern(FEType::Type{<:H1MINI}, ::Type{FaceDofs}, EG::Type{<:AbstractElementGeometry}) = "N1C1" # quick and dirty: C1 is ignored on faces, but need to calculate offset
get_dofmap_pattern(FEType::Type{<:H1MINI}, ::Type{BFaceDofs}, EG::Type{<:AbstractElementGeometry}) = "N1C1" # quick and dirty: C1 is ignored on faces, but need to calculate offset

function ensure_cell_moments!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, exact_function!; items = [], time = 0) where {FEType <: H1MINI}

    xgrid = FE.xgrid
    xItemVolumes = xgrid[CellVolumes]
    xItemNodes = xgrid[CellNodes]
    xItemDofs = FE[CellDofs]
    xCellGeometries = xgrid[CellGeometries]
    ncells = num_sources(xItemNodes)
    nnodes = size(xgrid[Coordinates],2)
    ncomponents = get_ncomponents(FEType)
    offset4component = 0:(nnodes+ncells):ncomponents*(nnodes+ncells)
    if items == []
        items = 1 : ncells
    end

    # compute exact cell integrals
    cellintegrals = zeros(Float64,ncomponents,ncells)
    integrate!(cellintegrals, xgrid, ON_CELLS, exact_function!; items = items, time = 0)
    cellEG = Triangle2D
    nitemnodes::Int = 0
    for item in items
        cellEG = xCellGeometries[item]
        nitemnodes = nnodes_for_geometry(cellEG)
        for c = 1 : ncomponents
            # subtract integral of P1 part
            for dof = 1 : nitemnodes
                cellintegrals[c,item] -= Target[xItemDofs[(c-1)*(nitemnodes+1) + dof,item]] * xItemVolumes[item] / nitemnodes
            end
            # set cell bubble such that cell mean is preserved
            Target[offset4component[c]+nnodes+item] = cellintegrals[c,item] / xItemVolumes[item]
        end
    end
end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{AT_NODES}, exact_function!; items = [], time = 0) where {FEType <: H1MINI}
    nnodes = size(FE.xgrid[Coordinates],2)
    ncells = num_sources(FE.xgrid[CellNodes])
    point_evaluation!(Target, FE, AT_NODES, exact_function!; items = items, component_offset = nnodes + ncells, time = time)
end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{ON_EDGES}, exact_function!; items = [], time = 0) where {FEType <: H1MINI}
    # delegate edge nodes to node interpolation
    subitems = slice(FE.xgrid[EdgeNodes], items)
    interpolate!(Target, FE, AT_NODES, exact_function!; items = subitems, time = time)
end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{ON_FACES}, exact_function!; items = [], time = 0) where {FEType <: H1MINI}
    # delegate face nodes to node interpolation
    subitems = slice(FE.xgrid[FaceNodes], items)
    interpolate!(Target, FE, AT_NODES, exact_function!; items = subitems, time = time)
end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{ON_CELLS}, exact_function!; items = [], time = 0) where {FEType <: H1MINI}
    # delegate cell nodes to node interpolation
    subitems = slice(FE.xgrid[CellNodes], items)
    interpolate!(Target, FE, AT_NODES, exact_function!; items = subitems, time = time)

    # fix cell bubble value by preserving integral mean
    ensure_cell_moments!(Target, FE, exact_function!; items = items, time = time)
end

function nodevalues!(Target::AbstractArray{<:Real,2}, Source::AbstractArray{<:Real,1}, FE::FESpace{<:H1MINI})
    nnodes = num_sources(FE.xgrid[Coordinates])
    ncells = num_sources(FE.xgrid[CellNodes])
    FEType = eltype(FE)
    ncomponents = get_ncomponents(FEType)
    offset4component = 0:(nnodes+ncells):ncomponents*(nnodes+ncells)
    for node = 1 : nnodes
        for c = 1 : ncomponents
            Target[c,node] = Source[offset4component[c]+node]
        end    
    end    
end

function get_basis(AT::Union{Type{<:ON_FACES},Type{<:ON_BFACES}}, FEType::Type{<:H1MINI}, EG::Type{<:AbstractElementGeometry})
    # on faces same as P1
    return get_basis(AT, H1P1{get_ncomponents(FEType)}, EG)
end

function get_basis(AT::Type{<:ON_CELLS}, FEType::Type{<:H1MINI}, EG::Type{<:Triangle2D})
    ncomponents = get_ncomponents(FEType)
    refbasis_P1 = get_basis(AT, H1P1{1}, EG)
    offset = get_ndofs(AT, H1P1{1}, EG) + 1
    function closure(refbasis, xref)
        refbasis_P1(refbasis, xref)
        # add cell bubbles to P1 basis (scaled to have unit integral)
        refbasis[offset,1] = 60*(1-xref[1]-xref[2])*xref[1]*xref[2]
        for k = 1 : ncomponents-1, j = 1 : offset
            refbasis[k*offset+j,k+1] = refbasis[j,1]
        end
    end
end

function get_basis(AT::Type{<:ON_CELLS}, FEType::Type{<:H1MINI}, EG::Type{<:Quadrilateral2D})
    ncomponents = get_ncomponents(FEType)
    refbasis_P1 = get_basis(AT, H1P1{1}, EG)
    offset = get_ndofs(AT, H1P1{1}, EG) + 1
    function closure(refbasis, xref)
        refbasis_P1(refbasis, xref)
        # add cell bubbles to P1 basis (scaled to have unit integral)
        refbasis[offset,1] = 36 *(1-xref[1])*(1-xref[2])*xref[1]*xref[2]
        for k = 1 : ncomponents-1, j = 1 : offset
            refbasis[k*offset+j,k+1] = refbasis[j,1]
        end
    end
end

function get_basis(AT::Type{<:ON_CELLS}, FEType::Type{<:H1MINI}, EG::Type{<:Tetrahedron3D})
    ncomponents = get_ncomponents(FEType)
    refbasis_P1 = get_basis(AT, H1P1{1}, EG)
    offset = get_ndofs(AT, H1P1{1}, EG) + 1
    function closure(refbasis, xref)
        refbasis_P1(refbasis, xref)
        # add cell bubbles to P1 basis (scaled to have unit integral)
        refbasis[offset,1] = 840*(1-xref[1]-xref[2]-xref[3])*xref[1]*xref[2]*xref[3]
        for k = 1 : ncomponents-1, j = 1 : offset
            refbasis[k*offset+j,k+1] = refbasis[j,1]
        end
    end
end