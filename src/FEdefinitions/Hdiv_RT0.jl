struct FEHdivRT0{spacedim} <: AbstractHdivFiniteElement where {spacedim<:Int}
    name::String                         # full name of finite element (used in messages)
    xgrid::ExtendableGrid                # link to xgrid 
    CellDofs::VariableTargetAdjacency    # place to save cell dofs (filled by constructor)
    FaceDofs::VariableTargetAdjacency    # place to save face dofs (filled by constructor)
    BFaceDofs::VariableTargetAdjacency   # place to save bface dofs (filled by constructor)
    CellSigns::VariableTargetAdjacency   # place to save cell signumscell coefficients
    ndofs::Int32
end

function getHdivRT0FiniteElement(xgrid::ExtendableGrid)
    dim = size(xgrid[Coordinates],1) 
    name = "RT0 (H1, $(dim)d)"    

    # generate dofmaps
    xFaceNodes = xgrid[FaceNodes]
    xCellFaces = xgrid[CellFaces]
    xCellSigns = xgrid[CellSigns]
    xBFaceNodes = xgrid[BFaceNodes]
    xBFaces = xgrid[BFaces]
    ncells = num_sources(xCellFaces)
    nfaces = num_sources(xFaceNodes)
    nbfaces = num_sources(xBFaceNodes)
    xCellDofs = VariableTargetAdjacency(Int32)
    xFaceDofs = VariableTargetAdjacency(Int32)
    xBFaceDofs = VariableTargetAdjacency(Int32)
    dofs4item = zeros(Int32,max_num_targets_per_source(xCellFaces))
    nnodes4item = 0
    for cell = 1 : ncells
        nnodes4item = num_targets(xCellFaces,cell)
        for k = 1 : nnodes4item
            dofs4item[k] = xCellFaces[k,cell]
        end
        append!(xCellDofs,dofs4item[1:nnodes4item])
    end
    for face = 1 : nfaces
        append!(xFaceDofs,[face])
    end
    for bface = 1: nbfaces
        append!(xBFaceDofs,[xBFaces[bface]])
    end
    return FEHdivRT0{dim}(name,xgrid,xCellDofs,xFaceDofs,xBFaceDofs,xCellSigns,nfaces)
end


get_ncomponents(::Type{<:FEHdivRT0{2}}) = 2
get_ncomponents(::Type{<:FEHdivRT0{3}}) = 3

get_polynomialorder(::Type{<:FEHdivRT0{2}}, ::Type{<:Edge1D}) = 0;
get_polynomialorder(::Type{<:FEHdivRT0{2}}, ::Type{<:Triangle2D}) = 1;
get_polynomialorder(::Type{<:FEHdivRT0{2}}, ::Type{<:Quadrilateral2D}) = 2;


function interpolate!(Target::AbstractArray{<:Real,1}, FE::FEHdivRT0{2}, exact_function!::Function; dofs = [], bonus_quadorder::Int = 0)
    # todo
    # integrate normal flux of exact_function over edges
end


function get_basis_normalflux_on_face(::Type{FEHdivRT0{2}}, ::Type{<:Edge1D})
    function closure(xref)
        return [1.0]
    end
end

function get_basis_on_cell(::Type{FEHdivRT0{2}}, ::Type{<:Triangle2D})
    function closure(xref)
        return [xref[1] xref[2]-1.0;
                xref[1] xref[2];
                xref[1]-1.0 xref[2]]
    end
end


function get_basis_on_cell(::Type{FEHdivRT0{2}}, ::Type{<:Quadrilateral2D})
    function closure(xref)
        return [[0.0 xref[2]-1.0];
                [xref[1] 0.0];
                [0.0 xref[2]];
                [xref[1]-1.0 0.0]] 
    end
end


function get_coefficients_on_cell!(coefficients, FE::FEHdivRT0{2}, ::Type{<:Triangle2D}, cell::Int)
    # multiplication with normal vectors
    coefficients[1,1] = FE.CellSigns[1,cell];
    coefficients[2,1] = FE.CellSigns[1,cell];
    coefficients[1,2] = FE.CellSigns[2,cell];
    coefficients[2,2] = FE.CellSigns[2,cell];
    coefficients[1,3] = FE.CellSigns[3,cell];
    coefficients[2,3] = FE.CellSigns[3,cell];
end    
function get_coefficients_on_cell!(coefficients, FE::FEHdivRT0{2}, ::Type{<:Quadrilateral2D}, cell::Int,)
    # multiplication with normal vectors
    fill!(coefficients,1.0)
    coefficients[1,1] = FE.CellSigns[1,cell];
    coefficients[2,1] = FE.CellSigns[1,cell];
    coefficients[1,2] = FE.CellSigns[2,cell];
    coefficients[2,2] = FE.CellSigns[2,cell];
    coefficients[1,3] = FE.CellSigns[3,cell];
    coefficients[2,3] = FE.CellSigns[3,cell];
    coefficients[1,4] = FE.CellSigns[4,cell];
    coefficients[2,4] = FE.CellSigns[4,cell];
end   


function get_coefficients_on_face!(coefficients, FE::FEHdivRT0{2}, ::Type{<:Edge1D}, face::Int)
    # multiplication with normal vectors
    fill!(coefficients,1.0)
end    
