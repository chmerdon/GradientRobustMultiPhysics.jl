
"""
````
abstract type H1P2B{ncomponents,edim} <: AbstractH1FiniteElement where {ncomponents<:Int,edim<:Int}
````

Continuous piecewise second-order polynomials.

allowed ElementGeometries:
- Triangle2D
"""
abstract type H1P2B{ncomponents,edim} <: AbstractH1FiniteElement where {ncomponents<:Int,edim<:Int} end

function Base.show(io::Core.IO, FEType::Type{<:H1P2B})
    print(io,"H1P2B{$(FEType.parameters[1]),$(FEType.parameters[2])}")
end

get_ncomponents(FEType::Type{<:H1P2B}) = FEType.parameters[1]
get_edim(FEType::Type{<:H1P2B}) = FEType.parameters[2]

get_ndofs(::Type{ON_CELLS}, FEType::Type{<:H1P2B}, EG::Type{<:Triangle2D}) = 7*FEType.parameters[1]
get_ndofs(::Union{Type{<:ON_FACES}, Type{<:ON_BFACES}}, FEType::Type{<:H1P2B}, EG::Type{<:AbstractElementGeometry1D}) = 3*FEType.parameters[1]

get_polynomialorder(::Type{<:H1P2B}, ::Type{<:Edge1D}) = 2;
get_polynomialorder(::Type{<:H1P2B}, ::Type{<:Triangle2D}) = 3;
get_polynomialorder(::Type{<:H1P2B}, ::Type{<:Tetrahedron3D}) = 4;

get_dofmap_pattern(FEType::Type{<:H1P2B}, ::Type{CellDofs}, EG::Type{<:AbstractElementGeometry2D}) = "N1F1I1"
get_dofmap_pattern(FEType::Type{<:H1P2B}, ::Type{FaceDofs}, EG::Type{<:AbstractElementGeometry1D}) = "N1I1C1"
get_dofmap_pattern(FEType::Type{<:H1P2B}, ::Type{BFaceDofs}, EG::Type{<:AbstractElementGeometry1D}) = "N1I1C1"

isdefined(FEType::Type{<:H1P2B}, ::Type{<:Triangle2D}) = (FEType.parameters[2] == 2)

get_ref_cellmoments(::Type{<:H1P2B}, ::Type{<:Triangle2D}) = [0//1, 0//1, 0//1, 1//3, 1//3, 1//3, 1//1] # integrals of 1D basis functions over reference cell (divided by volume)

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
    ensure_cell_moments!(Target, FE, exact_function!; facedofs = 1, edgedofs = 0, items = items, time = time)
end

function get_basis(AT::Union{Type{<:ON_FACES}, Type{<:ON_BFACES}}, FEType::Type{<:H1P2B}, EG::Type{<:AbstractElementGeometry})
    # on faces same as P2
    return get_basis(AT, H1P2{get_ncomponents(FEType), get_edim(FEType)}, EG)
end

function get_basis(AT::Type{ON_CELLS}, FEType::Type{<:H1P2B}, EG::Type{<:Triangle2D})
    ncomponents = get_ncomponents(FEType)
    edim = get_edim(FEType)
    refbasis_P2 = get_basis(AT, H1P2{1,edim}, EG)
    offset = get_ndofs(AT, H1P2{1,edim}, EG) + 1
    function closure(refbasis, xref)
        refbasis_P2(refbasis, xref)
        # add cell bubbles to P2 basis
        refbasis[offset,1] = 60*(1-xref[1]-xref[2])*xref[1]*xref[2]
        for k = 1 : ncomponents-1, j = 1 : offset
            refbasis[k*offset+j,k+1] = refbasis[j,1]
        end
    end
end