"""
$(TYPEDEF)

piecewise constant polynomials

allowed on every ElementGeometry
"""
abstract type L2P0{ncomponents} <: AbstractH1FiniteElement where {ncomponents<:Int} end


get_ncomponents(FEType::Type{<:L2P0}) = FEType.parameters[1]

get_polynomialorder(::Type{<:L2P0}, ::Type{<:AbstractElementGeometry}) = 0;


function init!(FES::FESpace{FEType}; dofmap_needed = true) where {FEType <: L2P0}
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


function init_dofmap!(FES::FESpace{FEType}, ::Type{AssemblyTypeCELL}) where {FEType <: L2P0}
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
    FES.CellDofs = xCellDofs
end


function init_dofmap!(FES::FESpace{FEType}, ::Type{AssemblyTypeFACE}) where {FEType <: L2P0}
    # not defined for L2 element
end

function init_dofmap!(FES::FESpace{FEType}, ::Type{AssemblyTypeBFACE}) where {FEType <: L2P0}
    xBFaceNodes = FES.xgrid[BFaceNodes]
    nbfaces = num_sources(xBFaceNodes)
    xBFaces = FES.xgrid[BFaces]
    xFaceCells = FES.xgrid[FaceCells]
    xBFaceDofs = VariableTargetAdjacency(Int32)
    for bface = 1: nbfaces
        append!(xBFaceDofs,[xFaceCells[1,xBFaces[bface]]])
    end
    FES.BFaceDofs = xBFaceDofs
end



function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{<:L2P0}, exact_function!::Function; dofs = [], bonus_quadorder::Int = 0)
    xCoords = FE.xgrid[Coordinates]
    xCellVolumes = FE.xgrid[CellVolumes]
    ncells = num_sources(FE.xgrid[CellNodes])
    FEType = eltype(FE)
    ncomponents = get_ncomponents(FEType)
    xdim = size(xCoords,1)
    if length(dofs) == 0 # interpolate at all dofs
        integrals4cell = zeros(Float64,ncells,ncomponents)
        integrate!(integrals4cell, FE.xgrid, AssemblyTypeCELL, exact_function!, bonus_quadorder, ncomponents)
        for cell = 1 : ncells
            for c = 1 : ncomponents
                Target[(cell-1)*ncomponents + c] = integrals4cell[cell,c] / xCellVolumes[cell]
            end
        end    
    else # todo: does not work for more than one component at the moment, also does not need to compute all integral means on all cells
        integrals4cell = zeros(Float64,ncells,ncomponents)
        integrate!(integrals4cell, FE.xgrid, AssemblyTypeCELL, exact_function!, bonus_quadorder, ncomponents)
        for dof in dofs
            cell = dof
            c = 1
            Target[(cell-1)*ncomponents + c] = integrals4cell[cell,c] / xCellVolumes[cell]
        end    
    end    
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

function get_basis_on_cell(::Type{L2P0{1}}, ::Type{<:AbstractElementGeometry})
    function closure(xref)
        return [1.0]
    end
end


function get_basis_on_cell(::Type{L2P0{2}}, ::Type{<:AbstractElementGeometry})
    function closure(xref)
        return [1.0 0.0;
                0.0 1.0]
    end
end

function get_basis_on_face(FE::Type{<:L2P0}, EG::Type{<:AbstractElementGeometry})
    function closure(xref)
        return get_basis_on_cell(FE, EG)(xref[1:end-1])
    end    
end

