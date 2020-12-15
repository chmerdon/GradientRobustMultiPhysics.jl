
"""
$(TYPEDEF)

Continuous piecewise second-order polynomials

allowed ElementGeometries:
- Triangle2D (quadratic polynomials + cell bubble)
"""
abstract type H1P2B{ncomponents,edim} <: AbstractH1FiniteElement where {ncomponents<:Int,edim<:Int} end

get_ncomponents(FEType::Type{<:H1P2B}) = FEType.parameters[1]
get_edim(FEType::Type{<:H1P2B}) = FEType.parameters[2]

get_ndofs_on_cell(FEType::Type{<:H1P2B}, EG::Type{<:Triangle2D}) = 7*FEType.parameters[1]
get_ndofs_on_face(FEType::Type{<:H1P2B}, EG::Type{<:AbstractElementGeometry1D}) = 3*FEType.parameters[1]

get_polynomialorder(::Type{<:H1P2B}, ::Type{<:Edge1D}) = 2;
get_polynomialorder(::Type{<:H1P2B}, ::Type{<:Triangle2D}) = 3;
get_polynomialorder(::Type{<:H1P2B}, ::Type{<:Tetrahedron3D}) = 4;


get_dofmap_pattern(FEType::Type{<:H1P2B}, ::Type{CellDofs}, EG::Type{<:AbstractElementGeometry2D}) = "N1F1I1"
get_dofmap_pattern(FEType::Type{<:H1P2B}, ::Type{FaceDofs}, EG::Type{<:AbstractElementGeometry1D}) = "N1I1C1"
get_dofmap_pattern(FEType::Type{<:H1P2B}, ::Type{BFaceDofs}, EG::Type{<:AbstractElementGeometry1D}) = "N1I1C1"


function ensure_cell_moments!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, exact_function!; items = [], time = 0) where {FEType <: H1P2B}

    xgrid = FE.xgrid
    xItemVolumes = xgrid[CellVolumes]
    xItemNodes = xgrid[CellNodes]
    xItemDofs = FE[CellDofs]
    xCellGeometries = xgrid[CellGeometries]
    ncells = num_sources(xItemNodes)
    nnodes = size(xgrid[Coordinates],2)
    edim = get_edim(FEType)
    ncomponents = get_ncomponents(FEType)
    offset = nnodes + num_sources(FE.xgrid[CellNodes])
    if edim == 2
        offset += num_sources(FE.xgrid[FaceNodes])
    elseif edim == 3
        offset += num_sources(FE.xgrid[EdgeNodes])
    end
    offset4component = 0:offset:ncomponents*offset
    if items == []
        items = 1 : num_sources(xItemNodes)
    end

    # compute exact cell integrals
    cellintegrals = zeros(Float64,ncomponents,ncells)
    integrate!(cellintegrals, xgrid, ON_CELLS, exact_function!; items = items, time = time)
    cellEG = Triangle2D
    nitemnodes::Int = 0
    for item in items
        cellEG = xCellGeometries[item]
        nitemnodes = nnodes_for_geometry(cellEG)
        for c = 1 : ncomponents
            # subtract integral of P2 part 
            # note: P2 vertex basis functions have cell integral zero !
            for dof = 1 : nitemnodes
                cellintegrals[c,item] -= Target[xItemDofs[(c-1)*(2*nitemnodes+1) + nitemnodes + dof,item]] * xItemVolumes[item] / nitemnodes
            end
            # in 3D subtract integral of face bubble
            if edim == 3
                # todo
            end
            # set cell bubble such that cell mean is preserved
            Target[offset4component[c]+offset-1] = cellintegrals[c,item] / xItemVolumes[item]
        end
    end
end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{AT_NODES}, exact_function!; items = [], time = 0) where {FEType <: H1P2B}
    edim = get_edim(FEType)
    nnodes = size(FE.xgrid[Coordinates],2)
    offset = nnodes + num_sources(FE.xgrid[CellNodes])
    if edim == 2
        offset += num_sources(FE.xgrid[FaceNodes])
    elseif edim == 3
        offset += num_sources(FE.xgrid[EdgeNodes])
    end

    point_evaluation!(Target, FE, AT_NODES, exact_function!; items = items, component_offset = offset, time = time)

end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{ON_EDGES}, exact_function!; items = [], time = 0) where {FEType <: H1P2B}
    edim = get_edim(FEType)
    if edim == 3
        # delegate edge nodes to node interpolation
        subitems = slice(FE.xgrid[EdgeNodes], items)
        interpolate!(Target, FE, AT_NODES, exact_function!; items = subitems, time = time)

        # perform edge mean interpolation
        ensure_edge_moments!(Target, FE, ON_EDGES, exact_function!; items = items, time = time)
    end
end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{ON_FACES}, exact_function!; items = [], time = 0) where {FEType <: H1P2B}
    edim = get_edim(FEType)
    if edim == 2
        # delegate face nodes to node interpolation
        subitems = slice(FE.xgrid[FaceNodes], items)
        interpolate!(Target, FE, AT_NODES, exact_function!; items = subitems, time = time)

        # perform face mean interpolation
        ensure_edge_moments!(Target, FE, ON_FACES, exact_function!; items = items, time = time)
    elseif edim == 3
        # delegate face edges to edge interpolation
        subitems = slice(FE.xgrid[FaceEdges], items)
        interpolate!(Target, FE, ON_EDGES, exact_function!; items = subitems, time = time)

        # perform face mean interpolation
        # todo
    elseif edim == 1
        # delegate face nodes to node interpolation
        subitems = slice(FE.xgrid[FaceNodes], items)
        interpolate!(Target, FE, AT_NODES, exact_function!; items = subitems, time = time)
    end
end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{ON_CELLS}, exact_function!; items = [], time = 0) where {FEType <: H1P2B}
    edim = get_edim(FEType)
    ncells = num_sources(FE.xgrid[CellNodes])
    if edim == 2
        # delegate cell faces to face interpolation
        subitems = slice(FE.xgrid[CellFaces], items)
        interpolate!(Target, FE, ON_FACES, exact_function!; items = subitems, time = time)
    elseif edim == 3
        # delegate cell edges to edge interpolation
        subitems = slice(FE.xgrid[CellEdges], items)
        interpolate!(Target, FE, ON_EDGES, exact_function!; items = subitems, time = time)
    elseif edim == 1
        # delegate cell nodes to node interpolation
        subitems = slice(FE.xgrid[CellNodes], items)
        interpolate!(Target, FE, AT_NODES, exact_function!; items = subitems, time = time)
    end

    # fix cell bubble value by preserving integral mean
    ensure_cell_moments!(Target, FE, exact_function!; items = items, time = time)
end

function get_basis_on_face(FEType::Type{<:H1P2B}, EG::Type{<:AbstractElementGeometry})
    ncomponents = get_ncomponents(FEType)
    edim = get_edim(FEType)
    # same as P2
    return get_basis_on_face(H1P2{ncomponents,edim}, EG)
end

function get_basis_on_cell(FEType::Type{<:H1P2B}, EG::Type{<:Triangle2D})
    ncomponents = get_ncomponents(FEType)
    edim = get_edim(FEType)
    refbasis_P2 = get_basis_on_cell(H1P2{1,edim}, EG)
    offset = get_ndofs_on_cell(H1P2{1,edim}, EG) + 1
    function closure(refbasis, xref)
        refbasis_P2(refbasis, xref)
        # add cell bubbles to P2 basis
        refbasis[offset,1] = 27*(1-xref[1]-xref[2])*xref[1]*xref[2]
        for k = 1 : ncomponents-1, j = 1 : offset
            refbasis[k*offset+j,k+1] = refbasis[j,1]
        end
    end
end