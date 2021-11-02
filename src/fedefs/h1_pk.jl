
"""
````
abstract type H1PK{ncomponents,edim,order} <: AbstractH1FiniteElement where {ncomponents<:Int,edim<:Int,order<:Int}
````

Continuous piecewise polynomials of arbitrary order >= 1.

allowed ElementGeometries:
- Edge1D
"""
abstract type H1Pk{ncomponents,edim,order} <: AbstractH1FiniteElement where {ncomponents<:Int,edim<:Int,order<:Int} end

function Base.show(io::Core.IO, ::Type{<:H1Pk{ncomponents,edim,order}}) where {ncomponents,edim,order}
    print(io,"H1Pk{$ncomponents,$edim,$order}")
end

get_ncomponents(FEType::Type{<:H1Pk}) = FEType.parameters[1]
get_edim(FEType::Type{<:H1Pk}) = FEType.parameters[2]

get_ndofs(::Type{<:AssemblyType}, FEType::Type{<:H1Pk}, EG::Type{<:AbstractElementGeometry0D}) = FEType.parameters[1]
get_ndofs(::Type{<:AssemblyType}, FEType::Type{H1Pk{n,e,order}}, EG::Type{<:AbstractElementGeometry1D}) where {n,e,order} = 1 + order

get_polynomialorder(::Type{H1Pk{n,e,order}}, ::Type{<:AbstractElementGeometry}) where {n,e,order} = order

get_dofmap_pattern(::Type{H1Pk{n,e,order}}, ::Type{CellDofs}, EG::Type{<:AbstractElementGeometry1D}) where {n,e,order} = (order == 1) ? "N1" : "N1I$(order-1)"
get_dofmap_pattern(::Type{H1Pk{n,e,order}}, ::Union{Type{FaceDofs},Type{BFaceDofs}}, EG::Type{<:AbstractElementGeometry0D}) where {n,e,order} = "N1"

isdefined(FEType::Type{<:H1Pk}, ::Type{<:AbstractElementGeometry1D}) = true

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{Tv,Ti,H1Pk{ncomponents,edim,order},APT}, ::Type{AT_NODES}, exact_function!; items = [], bonus_quadorder::Int = 0, time = 0) where {ncomponents,edim,order,Tv,Ti,APT}
    coffset = size(FE.xgrid[Coordinates],2)
    if edim == 1
        coffset += (order-1)*num_sources(FE.xgrid[CellNodes])
    elseif edim == 2
       # coffset += 2*num_sources(FE.xgrid[FaceNodes]) + num_sources(FE.xgrid[CellNodes])
    elseif edim == 3
       # coffset += 2*num_sources(FE.xgrid[EdgeNodes]) + num_sources(FE.xgrid[FaceNodes]) + num_sources(FE.xgrid[CellNodes])
    end

    point_evaluation!(Target, FE, AT_NODES, exact_function!; items = items, component_offset = coffset, time = time)
end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{Tv,Ti,H1Pk{ncomponents,edim,order},APT}, ::Type{ON_EDGES}, exact_function!; items = [], bonus_quadorder::Int = 0, time = 0) where {ncomponents,edim,order,Tv,Ti,APT}
    # edim = get_edim(FEType)
    # if edim == 3
    #     # delegate edge nodes to node interpolation
    #     subitems = slice(FE.xgrid[EdgeNodes], items)
    #     interpolate!(Target, FE, AT_NODES, exact_function!; items = subitems, time = time)

    #     # perform edge mean interpolation
    #     ensure_edge_moments!(Target, FE, ON_EDGES, exact_function!; order = 1, items = items, time = time)
    # end
end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{Tv,Ti,H1Pk{ncomponents,edim,order},APT}, ::Type{ON_FACES}, exact_function!; items = [], bonus_quadorder::Int = 0, time = 0) where {ncomponents,edim,order,Tv,Ti,APT}
    # edim = get_edim(FEType)
    if edim == 2
    #     # delegate face nodes to node interpolation
    #     subitems = slice(FE.xgrid[FaceNodes], items)
    #     interpolate!(Target, FE, AT_NODES, exact_function!; items = subitems, time = time)

    #     # perform face mean interpolation
    #     ensure_edge_moments!(Target, FE, ON_FACES, exact_function!; items = items, order = 1, time = time)
    # elseif edim == 3
    #     # delegate face edges to edge interpolation
    #     subitems = slice(FE.xgrid[FaceEdges], items)
    #     interpolate!(Target, FE, ON_EDGES, exact_function!; items = subitems, time = time)
    elseif edim == 1
         # delegate face nodes to node interpolation
         subitems = slice(FE.xgrid[FaceNodes], items)
         interpolate!(Target, FE, AT_NODES, exact_function!; items = subitems, time = time)
    end
end


function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{Tv,Ti,H1Pk{ncomponents,edim,order},APT}, ::Type{ON_CELLS}, exact_function!; items = [], bonus_quadorder::Int = 0, time = 0) where {ncomponents,edim,order,Tv,Ti,APT}
    if edim == 2
    #     # delegate cell faces to face interpolation
    #     subitems = slice(FE.xgrid[CellFaces], items)
    #     interpolate!(Target, FE, ON_FACES, exact_function!; items = subitems, time = time)
        
    #     # fix cell bubble value by preserving integral mean
    #     ensure_cell_moments!(Target, FE, exact_function!; facedofs = 2, items = items, time = time)
    # elseif edim == 3
    #     # todo
    #     # delegate cell edges to edge interpolation
    #     subitems = slice(FE.xgrid[CellEdges], items)
    #     interpolate!(Target, FE, ON_EDGES, exact_function!; items = subitems, time = time)

    #     # fixe face means

    #     # fix cell bubble value by preserving integral mean
    #     ensure_cell_moments!(Target, FE, exact_function!; facedofs = 1, edgedofs = 2, items = items, time = time)
    elseif edim == 1
        # delegate cell nodes to node interpolation
        subitems = slice(FE.xgrid[CellNodes], items)
        interpolate!(Target, FE, AT_NODES, exact_function!; items = subitems, time = time)

        # preserve cell integral
        if order > 1
            ensure_edge_moments!(Target, FE, ON_CELLS, exact_function!; order = order-2, items = items, time = time)
        end
    end
end


function get_basis(::Type{<:AssemblyType},::Type{H1Pk{ncomponents,edim,order}}, ::Type{<:Vertex0D}) where {ncomponents,edim,order}
    function closure(refbasis,xref)
        for k = 1 : ncomponents
            refbasis[k,k] = 1
        end
    end
end

function get_basis(::Type{<:AssemblyType}, ::Type{H1Pk{ncomponents,edim,order}}, ::Type{<:Edge1D}) where {ncomponents,edim,order}
    coeffs::Array{Rational{Int},1} = 0//1:(1//order):1//1
    # node functions first
    ordering::Array{Int,1} = [1,order+1]
    for j = 2:order
        push!(ordering,j)
    end
    # precalculate scaling factors
    factors::Array{Rational{Int},1} = ones(Rational{Int},order+1)
    for j = 1 : length(ordering), k = 1 : order + 1
        if k != ordering[j]
            factors[j] *= coeffs[ordering[j]] - coeffs[k]
        end
    end
    function closure(refbasis, xref)
        for j = 1 : length(ordering)
            ## find basis function that is 1 at x = coeffs[order[j]] and 0 at other positions
            refbasis[j,1] = 1
            for k = 1 : order + 1
                if k != ordering[j]
                    refbasis[j,1] *= (xref[1] - coeffs[k])
                end
            end
            refbasis[j,1] /= factors[j]
        end
        # copy to other components
        for j = 1 : order+1, k = 2 : ncomponents
            refbasis[(order+1)*(k-1)+j,k] = refbasis[j,1]
        end
    end

    return closure
end
