"""
$(TYPEDEF)

discontinuous piecewise first-order polynomials
(same as H1P1 but with other dofmap)

allowed ElementGeometries:
- Edge1D (linear polynomials)
- Triangle2D (linear polynomials)
- Quadrilateral2D (Q1 space)
"""
abstract type L2P1{ncomponents} <: AbstractH1FiniteElement where {ncomponents<:Int} end

get_ncomponents(::Type{L2P1{1}}) = 1
get_ncomponents(::Type{L2P1{2}}) = 2

get_polynomialorder(::Type{<:L2P1}, ::Type{<:Edge1D}) = 1;
get_polynomialorder(::Type{<:L2P1}, ::Type{<:Triangle2D}) = 1;
get_polynomialorder(::Type{<:L2P1}, ::Type{<:Quadrilateral2D}) = 2;


function init!(FES::FESpace{FEType}; dofmap_needed = true) where {FEType <: L2P1}
    ncomponents = get_ncomponents(FEType)
    name = "P1"
    for n = 1 : ncomponents-1
        name = name * "xP1"
    end
    FES.name = name * " (L2)"   

    # count number of dofs
    xCellNodes = FES.xgrid[CellNodes]
    ncells = num_sources(xCellNodes) 
    ndofs4component = 0
    for cell = 1 : ncells
        ndofs4component += num_targets(xCellNodes,cell)
    end

    FES.ndofs = ndofs4component * ncomponents
end



function init_dofmap!(FES::FESpace{FEType}, ::Type{AssemblyTypeCELL}) where {FEType <: L2P1}
    xCellNodes = FES.xgrid[CellNodes]
    xCellGeometries = FES.xgrid[CellGeometries]
    xCellDofs = VariableTargetAdjacency(Int32)
    ncomponents = get_ncomponents(FEType)
    ncells = num_sources(xCellNodes) 
    dofs4item = zeros(Int32,ncomponents*max_num_targets_per_source(xCellNodes))
    nnodes4item = 0
    dof = 0
    ndofs4component = 0
    if ncomponents > 0
        for cell = 1 : ncells
            ndofs4component += num_targets(xCellNodes,cell)
        end
    end
    for cell = 1 : ncells
        nnodes4item = num_targets(xCellNodes,cell)
        for k = 1 : nnodes4item
            dof += 1
            dofs4item[k] = dof
            for n = 1 : ncomponents-1
                dofs4item[k+n*nnodes4item] = n*ndofs4component + dof
            end    
        end
        append!(xCellDofs,dofs4item[1:ncomponents*nnodes4item])
    end

    # save dofmap
    FES.CellDofs = xCellDofs
end

function init_dofmap!(FES::FESpace{FEType}, ::Type{AssemblyTypeFACE}) where {FEType <: L2P1}
    # not defined for L2 element
end

function init_dofmap!(FES::FESpace{FEType}, ::Type{AssemblyTypeBFACE}) where {FEType <: L2P1}
    xCellNodes = FES.xgrid[CellNodes]
    xCellGeometries = FES.xgrid[CellGeometries]
    xBFaceNodes = FES.xgrid[BFaceNodes]
    nbfaces = num_sources(xBFaceNodes)
    ncells = num_sources(xCellNodes) 
    xBFaceCellPos = FES.xgrid[BFaceCellPos]
    xBFaces = FES.xgrid[BFaces]
    xFaceCells = FES.xgrid[FaceCells]
    ncomponents = get_ncomponents(FEType)
    dofs4item = zeros(Int32,ncomponents*max_num_targets_per_source(xBFaceNodes))
    xBFaceDofs = VariableTargetAdjacency(Int32)
    ndofs4component = 0
    if ncomponents > 0
        for cell = 1 : ncells
            ndofs4component += num_targets(xCellNodes,cell)
        end
    end
    nnodes4item = 0
    cell = 0
    pos = 0
    for bface = 1: nbfaces
        face = xBFaces[bface]
        cell = xFaceCells[1,face]
        pos = xBFaceCellPos[bface]
        nnodes4item = num_targets(xBFaceNodes,bface)
        for k = 1 : nnodes4item
            dofs4item[k] = FES.CellDofs[1,cell] + face_enum_rule(xCellGeometries[cell])[pos,k] - 1
            for n = 1 : ncomponents-1
                dofs4item[k+n*nnodes4item] = n*ndofs4component + dofs4item[k]
            end    
        end
        append!(xBFaceDofs,dofs4item[1:ncomponents*nnodes4item])
    end
    # save dofmap
    FES.BFaceDofs = xBFaceDofs
end


function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{<:L2P1}, exact_function!::Function; dofs = [], bonus_quadorder::Int = 0)
    xCoords = FE.xgrid[Coordinates]
    xdim = size(xCoords,1)
    x = zeros(Float64,xdim)
    nnodes = num_sources(xCoords)
    xCellNodes = FE.xgrid[CellNodes]
    ncells = num_sources(xCellNodes)
    nnodes4item::Int = 0
    FEType = eltype(FE)
    ncomponents::Int = get_ncomponents(FEType)
    result = zeros(Float64,ncomponents)
    ndofs4component::Int = ceil(FE.ndofs / ncomponents)
    node::Int = 0
    if length(dofs) == 0 # interpolate at all dofs
        dof::Int = 0
        for cell = 1 : ncells
            nnodes4item = num_targets(xCellNodes,cell)
            for k = 1 : nnodes4item
                dof += 1
                node = xCellNodes[k,cell]
                for d=1:xdim
                    x[d] = xCoords[d,node]
                end    
                exact_function!(result,x)
                for c = 1 : ncomponents
                    Target[dof+(c-1)*ndofs4component] = result[c]
                end
            end
        end
    else
        #TODO 
    end    
end

function get_basis_on_cell(::Type{L2P1{1}}, EG::Type{<:AbstractElementGeometry}) 
    return get_basis_on_cell(H1P1{1}, EG)
end


function get_basis_on_cell(::Type{L2P1{2}}, EG::Type{<:AbstractElementGeometry}) 
    return get_basis_on_cell(H1P1{2}, EG)
end

# face functions are not continuous and should only be used on BFACES
function get_basis_on_face(::Type{L2P1{1}}, EG::Type{<:AbstractElementGeometry}) 
    return get_basis_on_face(H1P1{1}, EG)
end

function get_basis_on_face(::Type{L2P1{2}}, EG::Type{<:AbstractElementGeometry}) 
    return get_basis_on_face(H1P1{2}, EG)
end
