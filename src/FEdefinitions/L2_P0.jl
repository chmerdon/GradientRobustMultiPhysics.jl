abstract type L2P0{ncomponents} <: AbstractH1FiniteElement where {ncomponents<:Int} end


get_ncomponents(::Type{L2P0{1}}) = 1
get_ncomponents(::Type{L2P0{2}}) = 2

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

    # generate dofmaps
    if dofmap_needed
        xCellDofs = VariableTargetAdjacency(Int32)
        dofs4item = zeros(Int32,ncomponents)
        nnodes4item = 0
        for cell = 1 : ncells
            dofs4item[1] = cell
            for n = 1 : ncomponents-1
                dofs4item[1+n] = n*ncells + cell
            end    
            append!(xCellDofs,dofs4item)
        end

        # save dofmaps
        FES.CellDofs = xCellDofs
    end

end


function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{<:L2P0}, exact_function!::Function; dofs = [], bonus_quadorder::Int = 0)
    xCoords = FE.xgrid[Coordinates]
    xCellVolumes = FE.xgrid[CellVolumes]
    ncells = num_sources(FE.xgrid[CellNodes])
    FEType = eltype(typeof(FE))
    ncomponents = get_ncomponents(FEType)
    xdim = size(xCoords,1)
    if length(dofs) == 0 # interpolate at all dofs
        integrals4cell = zeros(Float64,ncells,ncomponents)
        integrate!(integrals4cell, FE.xgrid, AbstractAssemblyTypeCELL, exact_function!, bonus_quadorder, ncomponents)
        for cell = 1 : ncells
            Target[cell] = integrals4cell[cell,1] / xCellVolumes[cell]
        end    
    else
        #TODO 
    end    
end


function nodevalues!(Target::AbstractArray{<:Real,2}, Source::AbstractArray{<:Real,1}, FE::FESpace{<:L2P0})
    xCoords = FE.xgrid[Coordinates]
    xCellNodes = FE.xgrid[CellNodes]
    xNodeCells = atranspose(xCellNodes)
    ncells = num_sources(xCellNodes)
    FEType = eltype(typeof(FE))
    ncomponents = get_ncomponents(FEType)
    value = 0.0
    nneighbours = 0
    offset4component = 0:ncells:ncomponents*ncells
    for node = 1 : num_sources(xCoords)
        for c = 1 : ncomponents
            value = 0.0
            nneighbours = num_targets(xNodeCells,node)
            for n = 1 : nneighbours
                value += Source[offset4component[c]+xNodeCells[n,node]]
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

