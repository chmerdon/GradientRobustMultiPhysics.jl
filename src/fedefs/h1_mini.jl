
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

get_polynomialorder(FEType::Type{<:H1MINI}, ::Type{<:Edge1D}) = FEType.parameters[2] == 1 ? 2 : 1
get_polynomialorder(FEType::Type{<:H1MINI}, ::Type{<:Triangle2D}) = FEType.parameters[2] == 2 ? 3 : 1;
get_polynomialorder(FEType::Type{<:H1MINI}, ::Type{<:Quadrilateral2D}) = FEType.parameters[2] == 2 ? 4 : 2;
get_polynomialorder(FEType::Type{<:H1MINI}, ::Type{<:Tetrahedron3D}) = 4;

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

get_dofmap_pattern(FEType::Type{<:H1MINI}, ::Type{CellDofs}, EG::Type{<:AbstractElementGeometry}) = "N1I1"
get_dofmap_pattern(FEType::Type{<:H1MINI}, ::Type{FaceDofs}, EG::Type{<:AbstractElementGeometry}) = "N1C1" # quick and dirty: C1 is ignored on faces, but need to calculate offset
get_dofmap_pattern(FEType::Type{<:H1MINI}, ::Type{BFaceDofs}, EG::Type{<:AbstractElementGeometry}) = "N1C1" # quick and dirty: C1 is ignored on faces, but need to calculate offset



function ensure_cell_moments!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, exact_function!::Function; items = [], bonus_quadorder::Int = 0) where {FEType <: H1MINI}

    xgrid = FE.xgrid
    xItemVolumes = xgrid[CellVolumes]
    xItemNodes = xgrid[CellNodes]
    xItemDofs = FE.dofmaps[CellDofs]
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
    integrate!(cellintegrals, xgrid, ON_CELLS, exact_function!, bonus_quadorder, ncomponents; items = items)
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


function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{AT_NODES}, exact_function!::Function; items = [], bonus_quadorder::Int = 0) where {FEType <: H1MINI}
    nnodes = size(FE.xgrid[Coordinates],2)
    ncells = num_sources(FE.xgrid[CellNodes])
    point_evaluation!(Target, FE, AT_NODES, exact_function!; items = items, component_offset = nnodes + ncells)
end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{ON_EDGES}, exact_function!::Function; items = [], bonus_quadorder::Int = 0) where {FEType <: H1MINI}
    # delegate edge nodes to node interpolation
    subitems = slice(FE.xgrid[EdgeNodes], items)
    interpolate!(Target, FE, AT_NODES, exact_function!; items = subitems, bonus_quadorder = bonus_quadorder)
end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{ON_FACES}, exact_function!::Function; items = [], bonus_quadorder::Int = 0) where {FEType <: H1MINI}
    # delegate face nodes to node interpolation
    subitems = slice(FE.xgrid[FaceNodes], items)
    interpolate!(Target, FE, AT_NODES, exact_function!; items = subitems, bonus_quadorder = bonus_quadorder)
end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{ON_CELLS}, exact_function!::Function; items = [], bonus_quadorder::Int = 0) where {FEType <: H1MINI}
    # delegate cell nodes to node interpolation
    subitems = slice(FE.xgrid[CellNodes], items)
    interpolate!(Target, FE, AT_NODES, exact_function!; items = subitems, bonus_quadorder = bonus_quadorder)

    # fix cell bubble value by preserving integral mean
    ensure_cell_moments!(Target, FE, exact_function!; items = items, bonus_quadorder = bonus_quadorder)
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


function get_basis_on_face(FEType::Type{<:H1MINI}, EG::Type{<:AbstractElementGeometry})
    ncomponents = get_ncomponents(FEType)
    # same as P1
    return get_basis_on_face(H1P1{ncomponents}, EG)
end

function get_basis_on_cell(FEType::Type{<:H1MINI}, EG::Type{<:Triangle2D})
    ncomponents = get_ncomponents(FEType)
    refbasis_P1 = get_basis_on_cell(H1P1{1}, EG)
    offset = get_ndofs_on_cell(H1P1{1}, EG) + 1
    function closure(refbasis, xref)
        refbasis_P1(refbasis, xref)
        # add cell bubbles to P1 basis (scaled to have unit integral)
        refbasis[offset,1] = 60*(1-xref[1]-xref[2])*xref[1]*xref[2]
        for k = 1 : ncomponents-1, j = 1 : offset
            refbasis[k*offset+j,k+1] = refbasis[j,1]
        end
    end
end

function get_basis_on_cell(FEType::Type{<:H1MINI}, EG::Type{<:Quadrilateral2D})
    ncomponents = get_ncomponents(FEType)
    refbasis_P1 = get_basis_on_cell(H1P1{1}, EG)
    offset = get_ndofs_on_cell(H1P1{1}, EG) + 1
    function closure(refbasis, xref)
        refbasis_P1(refbasis, xref)
        # add cell bubbles to P1 basis (scaled to have unit integral)
        refbasis[offset,1] = 36 *(1-xref[1])*(1-xref[2])*xref[1]*xref[2]
        for k = 1 : ncomponents-1, j = 1 : offset
            refbasis[k*offset+j,k+1] = refbasis[j,1]
        end
    end
end

function get_basis_on_cell(FEType::Type{<:H1MINI}, EG::Type{<:Tetrahedron3D})
    ncomponents = get_ncomponents(FEType)
    refbasis_P1 = get_basis_on_cell(H1P1{1}, EG)
    offset = get_ndofs_on_cell(H1P1{1}, EG) + 1
    function closure(refbasis, xref)
        refbasis_P1(refbasis, xref)
        # add cell bubbles to P1 basis (scaled to have unit integral)
        refbasis[offset,1] = 840*(1-xref[1]-xref[2]-xref[3])*xref[1]*xref[2]*xref[3]
        for k = 1 : ncomponents-1, j = 1 : offset
            refbasis[k*offset+j,k+1] = refbasis[j,1]
        end
    end
end