"""
````
abstract type H1P0{ncomponents} <: AbstractH1FiniteElement where {ncomponents<:Int}
````

Piecewise constant polynomials.

allowed on every ElementGeometry
"""
abstract type H1P0{ncomponents} <: AbstractH1FiniteElement where {ncomponents<:Int} end

get_ncomponents(FEType::Type{<:H1P0}) = FEType.parameters[1]
get_ndofs(::Union{Type{<:ON_CELLS},Type{<:ON_BFACES}}, FEType::Type{<:H1P0}, EG::Type{<:AbstractElementGeometry}) = FEType.parameters[1]

get_polynomialorder(::Type{<:H1P0}, ::Type{<:AbstractElementGeometry}) = 0;

get_dofmap_pattern(FEType::Type{<:H1P0}, ::Type{CellDofs}, EG::Type{<:AbstractElementGeometry}) = "I1"
get_dofmap_pattern(FEType::Type{<:H1P0}, ::Type{BFaceDofs}, EG::Type{<:AbstractElementGeometry}) = "C1"

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{ON_CELLS}, exact_function!; items = [], time = time) where {FEType <: H1P0}
    xCoords = FE.xgrid[Coordinates]
    xCellVolumes = FE.xgrid[CellVolumes]
    ncells = num_sources(FE.xgrid[CellNodes])
    if items == []
        items = 1 : ncells
    else
        items = filter(!iszero, items)
    end
    ncomponents = get_ncomponents(FEType)
    xdim = size(xCoords,1)
    integrals4cell = zeros(Float64,ncomponents,ncells)
    integrate!(integrals4cell, FE.xgrid, ON_CELLS, exact_function!; items = items, time = time)
    for cell in items
        if cell != 0
            for c = 1 : ncomponents
                Target[(cell-1)*ncomponents + c] = integrals4cell[c, cell] / xCellVolumes[cell]
            end
        end
    end    
end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{ON_FACES}, exact_function!; items = [], time = 0) where {FEType <: H1P0}
    # delegate to node cell interpolation
    subitems = slice(FE.xgrid[FaceCells], items)
    interpolate!(Target, FE, ON_CELLS, exact_function!; items = subitems, time = time)
end

function nodevalues!(Target::AbstractArray{<:Real,2}, Source::AbstractArray{<:Real,1}, FE::FESpace{<:H1P0})
    xCoords = FE.xgrid[Coordinates]
    xCellNodes = FE.xgrid[CellNodes]
    xNodeCells = atranspose(xCellNodes)
    ncells = num_sources(xCellNodes)
    FEType = eltype(FE)
    ncomponents = get_ncomponents(FEType)
    value = 0.0
    nneighbours = 0
    for node = 1 : num_sources(xCoords)
        for c = 1 : ncomponents
            value = 0.0
            nneighbours = num_targets(xNodeCells,node)
            for n = 1 : nneighbours
                value += Source[(xNodeCells[n,node]-1)*ncomponents+c]
            end
            value /= nneighbours
            Target[c,node] = value
        end
    end    
end

function get_basis(::Union{<:Type{ON_CELLS},<:Type{ON_BFACES}}, FEType::Type{<:H1P0}, ::Type{<:AbstractElementGeometry})
    ncomponents = get_ncomponents(FEType)
    function closure(refbasis, xref)
        for k = 1 : ncomponents
            refbasis[k,k] = 1.0
        end
    end
end