struct FEH1P0{ncomponents} <: AbstractH1FiniteElement where {ncomponents<:Int}
    name::String                         # full name of finite element (used in messages)
    xgrid::ExtendableGrid                # link to xgrid 
    CellDofs::VariableTargetAdjacency    # place to save cell dofs (filled by constructor)
    FaceDofs::VariableTargetAdjacency    # place to save face dofs (filled by constructor)
    BFaceDofs::VariableTargetAdjacency   # place to save bface dofs (filled by constructor)
    ndofs::Int32
end

function getP0FiniteElement(xgrid::ExtendableGrid, ncomponents::Int)
    name = "P0"
    for n = 1 : ncomponents-1
        name = name * "xP0"
    end
    name = name * " (L2)"    

    # generate celldofs
    dim = size(xgrid[Coordinates],1) 
    xCellNodes = xgrid[CellNodes]
    xFaceNodes = xgrid[FaceNodes]
    xCellGeometries = xgrid[CellGeometries]
    xBFaceNodes = xgrid[BFaceNodes]
    xBFaces = xgrid[BFaces]
    ncells = num_sources(xCellNodes)
    nfaces = num_sources(xFaceNodes)
    nbfaces = num_sources(xBFaceNodes)
    nnodes = num_sources(xgrid[Coordinates])

    # generate dofmaps
    xCellDofs = VariableTargetAdjacency(Int32)
    xFaceDofs = VariableTargetAdjacency(Int32)
    xBFaceDofs = VariableTargetAdjacency(Int32)
    dofs4item = zeros(Int32,ncomponents*max_num_targets_per_source(xCellNodes))
    nnodes4item = 0
    for cell = 1 : ncells
        append!(xCellDofs,cell*ones(Int32,ncomponents))
    end

    return FEH1P0{ncomponents}(name,xgrid,xCellDofs,xFaceDofs,xBFaceDofs,ncells * ncomponents)
end


get_ncomponents(::Type{FEH1P0{1}}) = 1
get_ncomponents(::Type{FEH1P0{2}}) = 2

get_polynomialorder(::Type{<:FEH1P0}, ::Type{<:AbstractElementGeometry}) = 0;


function interpolate!(Target::AbstractArray{<:Real,1}, FE::FEH1P0{1}, exact_function!::Function; dofs = [], bonus_quadorder::Int = 0)
    xCoords = FE.xgrid[Coordinates]
    xCellVolumes = FE.xgrid[CellVolumes]
    ncells = num_sources(FE.xgrid[CellNodes])
    ncomponents = get_ncomponents(typeof(FE))
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


function nodevalues!(Target::AbstractArray{<:Real,2}, Source::AbstractArray{<:Real,1}, FE::FEH1P0)
    xCoords = FE.xgrid[Coordinates]
    xCellNodes = FE.xgrid[CellNodes]
    xNodeCells = atranspose(xCellNodes)
    ncells = num_sources(xCellNodes)
    ncomponents = get_ncomponents(typeof(FE))
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

function get_basis_on_cell(::Type{FEH1P0{1}}, ::Type{<:AbstractElementGeometry})
    function closure(xref)
        return [1.0]
    end
end


function get_basis_on_cell(::Type{FEH1P0{2}}, ::Type{<:AbstractElementGeometry})
    function closure(xref)
        return [1.0 0.0;
                0.0 1.0]
    end
end

function get_basis_on_face(FE::Type{<:FEH1P0}, EG::Type{<:AbstractElementGeometry})
    function closure(xref)
        return get_basis_on_cell(FE, EG)(xref[1:end-1])
    end    
end

