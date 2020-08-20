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

get_ncomponents(FEType::Type{<:L2P1}) = FEType.parameters[1]

get_polynomialorder(::Type{<:L2P1}, ::Type{<:Edge1D}) = 1;
get_polynomialorder(::Type{<:L2P1}, ::Type{<:Triangle2D}) = 1;
get_polynomialorder(::Type{<:L2P1}, ::Type{<:Quadrilateral2D}) = 2;
get_polynomialorder(::Type{<:L2P1}, ::Type{<:Tetrahedron3D}) = 1;
get_polynomialorder(::Type{<:L2P1}, ::Type{<:Hexahedron3D}) = 3;


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



function init_dofmap!(FES::FESpace{FEType}, ::Type{CellDofs}) where {FEType <: L2P1}
    xCellNodes = FES.xgrid[CellNodes]
    ncomponents = get_ncomponents(FEType)
    ncells = num_sources(xCellNodes) 
    dof = 0
    colstart = Array{Int32,1}([1])
    for cell = 1 : ncells
        dof += num_targets(xCellNodes,cell) * ncomponents
        push!(colstart,dof+1)
    end
    #xCellDofs = VariableTargetAdjacency{Int32}(1:dof,colstart)
    xCellDofs = SerialVariableTargetAdjacency{Int32}(colstart)
    # save dofmap
    FES.dofmaps[CellDofs] = xCellDofs
end

function init_dofmap!(FES::FESpace{FEType}, ::Type{FaceDofs}) where {FEType <: L2P1}
    # not defined for L2 element
end

function init_dofmap!(FES::FESpace{FEType}, ::Type{BFaceDofs}) where {FEType <: L2P1}
    xCellNodes = FES.xgrid[CellNodes]
    xCellGeometries = FES.xgrid[CellGeometries]
    xBFaceNodes = FES.xgrid[BFaceNodes]
    nbfaces = num_sources(xBFaceNodes)
    xBFaceCellPos = FES.xgrid[BFaceCellPos]
    xBFaces = FES.xgrid[BFaces]
    xFaceCells = FES.xgrid[FaceCells]
    xCellDofs = FES.dofmaps[CellDofs]
    ncomponents = get_ncomponents(FEType)
    dofs4item = zeros(Int32,ncomponents*max_num_targets_per_source(xBFaceNodes))
    nnodes4bface = 0
    nnodes4cell = 0
    cell = 0
    pos = 0
    pos_node = 0
    xBFaceDofs = VariableTargetAdjacency(Int32)
    for bface = 1: nbfaces
        face = xBFaces[bface]
        cell = xFaceCells[1,face]
        pos = xBFaceCellPos[bface]
        nnodes4bface = num_targets(xBFaceNodes,bface)
        nnodes4cell = num_targets(xCellNodes,cell)
        for k = 1 : nnodes4bface
            pos_node = findall(x->x==xBFaceNodes[k,bface], xCellNodes[:,cell])[1]
            dofs4item[k] = xCellDofs[1,cell] - 1 + pos_node
            for n = 1 : ncomponents-1
                dofs4item[k+n*nnodes4bface] = n*nnodes4cell + dofs4item[k]
            end    
        end
        append!(xBFaceDofs,dofs4item[1:ncomponents*nnodes4bface])
    end
    # save dofmap
    FES.dofmaps[BFaceDofs] = xBFaceDofs
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
    node::Int = 0
    if length(dofs) == 0 # interpolate at all dofs
        dof::Int = 0
        for cell = 1 : ncells
            nnodes4item = num_targets(xCellNodes,cell)
            for k = 1 : nnodes4item
                node = xCellNodes[k,cell]
                for d=1:xdim
                    x[d] = xCoords[d,node]
                end    
                exact_function!(result,x)
                for c = 1 : ncomponents
                    Target[dof+(c-1)*nnodes4item+k] = result[c]
                end
            end
            dof += nnodes4item*ncomponents
        end
    else
        #TODO 
    end    
end


# use same functions as P1
function get_basis_on_cell(FEType::Type{<:L2P1}, EG::Type{<:AbstractElementGeometry}) 
    return get_basis_on_cell(H1P1{get_ncomponents(FEType)}, EG)
end

# face functions are not continuous and should only be used on BFACES
function get_basis_on_face(FEType::Type{<:L2P1}, EG::Type{<:AbstractElementGeometry}) 
    return get_basis_on_face(H1P1{get_ncomponents(FEType)}, EG)
end
