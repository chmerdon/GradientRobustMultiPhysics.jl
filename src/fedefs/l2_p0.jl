"""
$(TYPEDEF)

piecewise constant polynomials

allowed on every ElementGeometry
"""
abstract type L2P0{ncomponents} <: AbstractH1FiniteElement where {ncomponents<:Int} end


get_ncomponents(FEType::Type{<:L2P0}) = FEType.parameters[1]
get_ndofs_on_face(FEType::Type{<:L2P0}, EG::Type{<:AbstractElementGeometry}) = FEType.parameters[1]
get_ndofs_on_cell(FEType::Type{<:L2P0}, EG::Type{<:AbstractElementGeometry}) = FEType.parameters[1]

get_polynomialorder(::Type{<:L2P0}, ::Type{<:AbstractElementGeometry}) = 0;


function init!(FES::FESpace{FEType}) where {FEType <: L2P0}
    ncomponents = get_ncomponents(FEType)
    name = "P0"
    for n = 1 : ncomponents-1
        name = name * "xP0"
    end
    FES.name = name * " (L2)"   

    # count number of dofs
    xCellNodes = FES.xgrid[CellNodes]
    ncells = num_sources(xCellNodes) 
    FES.ndofs = ncells * ncomponents
end


function init_dofmap!(FES::FESpace{FEType}, ::Type{CellDofs}) where {FEType <: L2P0}
    ncomponents = get_ncomponents(FEType)
    ncells = num_sources(FES.xgrid[CellNodes]) 
    dof = 0
    colstart = Array{Int32,1}([1])
    for cell = 1 : ncells
        dof += ncomponents
        push!(colstart,dof+1)
    end
    #xCellDofs = VariableTargetAdjacency{Int32}(1:dof,colstart)
    xCellDofs = SerialVariableTargetAdjacency{Int32}(colstart)
    # save dofmap
    FES.dofmaps[CellDofs] = xCellDofs
end


function init_dofmap!(FES::FESpace{FEType}, ::Type{FaceDofs}) where {FEType <: L2P0}
    # not defined for L2 element
end

function init_dofmap!(FES::FESpace{FEType}, ::Type{BFaceDofs}) where {FEType <: L2P0}
    xBFaceNodes = FES.xgrid[BFaceNodes]
    nbfaces = num_sources(xBFaceNodes)
    xBFaces = FES.xgrid[BFaces]
    xFaceCells = FES.xgrid[FaceCells]
    xBFaceDofs = VariableTargetAdjacency(Int32)
    for bface = 1: nbfaces
        append!(xBFaceDofs,[xFaceCells[1,xBFaces[bface]]])
    end
    FES.dofmaps[BFaceDofs] = xBFaceDofs
end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{ON_CELLS}, exact_function!; items = [], time = time) where {FEType <: L2P0}
    xCoords = FE.xgrid[Coordinates]
    xCellVolumes = FE.xgrid[CellVolumes]
    ncells = num_sources(FE.xgrid[CellNodes])
    if items == []
        items = 1 : ncells
    end
    ncomponents = get_ncomponents(FEType)
    xdim = size(xCoords,1)
    integrals4cell = zeros(Float64,ncomponents,ncells)
    integrate!(integrals4cell, FE.xgrid, ON_CELLS, exact_function!; time = time)
    for cell in items
        if cell != 0
            for c = 1 : ncomponents
                Target[(cell-1)*ncomponents + c] = integrals4cell[c, cell] / xCellVolumes[cell]
            end
        end
    end    
end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{ON_FACES}, exact_function!; items = [], time = 0) where {FEType <: L2P0}
    # delegate to node cell interpolation
    subitems = slice(FE.xgrid[FaceCells], items)
    interpolate!(Target, FE, ON_CELLS, exact_function!; items = subitems, time = time)
end



function nodevalues!(Target::AbstractArray{<:Real,2}, Source::AbstractArray{<:Real,1}, FE::FESpace{<:L2P0})
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

function get_basis_on_cell(FEType::Type{<:L2P0}, ::Type{<:AbstractElementGeometry})
    ncomponents = get_ncomponents(FEType)
    function closure(refbasis, xref)
        for k = 1 : ncomponents
            refbasis[k,k] = 1.0
        end
    end
end

function get_basis_on_face(FE::Type{<:L2P0}, EG::Type{<:AbstractElementGeometry})
    refbasis_cell = get_basis_on_cell(FE, EG)
    function closure(refbasis, xref)
        return refbasis_cell(refbasis, xref[1:end-1])
    end    
end

