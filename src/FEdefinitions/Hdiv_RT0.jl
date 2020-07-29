"""
$(TYPEDEF)

Hdiv-conforming vector-valued (ncomponents = spacedim) lowest-order Raviart-Thomas space

allowed ElementGeometries:
- Triangle2D
- Quadrilateral2D
"""
abstract type HDIVRT0{spacedim} <: AbstractHdivFiniteElement where {spacedim<:Int} end

get_ncomponents(::Type{<:HDIVRT0{2}}) = 2
get_ncomponents(::Type{<:HDIVRT0{3}}) = 3

get_polynomialorder(::Type{<:HDIVRT0{2}}, ::Type{<:Edge1D}) = 0;
get_polynomialorder(::Type{<:HDIVRT0{2}}, ::Type{<:Triangle2D}) = 1;
get_polynomialorder(::Type{<:HDIVRT0{2}}, ::Type{<:Quadrilateral2D}) = 2;


function init!(FES::FESpace{FEType}; dofmap_needed = true) where {FEType <: HDIVRT0}
    ncomponents = get_ncomponents(FEType)
    FES.name = "RT0 (H1, $(ncomponents)d)"

    # count number of dofs
    nfaces = num_sources(FES.xgrid[FaceNodes])
    FES.ndofs = nfaces

    # register coefficients
    FES.xCellFaceSigns = FES.xgrid[CellFaceSigns]

end

function init_dofmap!(FES::FESpace{FEType}, ::Type{AssemblyTypeCELL}) where {FEType <: HDIVRT0}
    xCellFaces = FES.xgrid[CellFaces]
    xCellGeometries = FES.xgrid[CellGeometries]
    dofs4item = zeros(Int32,max_num_targets_per_source(xCellFaces))
    ncells = num_sources(xCellFaces)
    xCellDofs = VariableTargetAdjacency(Int32)
    nfaces4item = 0
    for cell = 1 : ncells
        nfaces4item = num_targets(xCellFaces,cell)
        for k = 1 : nfaces4item
            dofs4item[k] = xCellFaces[k,cell]
        end
        append!(xCellDofs,dofs4item[1:nfaces4item])
    end
    # save dofmap
    FES.CellDofs = xCellDofs
end

function init_dofmap!(FES::FESpace{FEType}, ::Type{AssemblyTypeFACE}) where {FEType <: HDIVRT0}
    nfaces = num_sources(FES.xgrid[FaceNodes])
    xFaceDofs = VariableTargetAdjacency(Int32)
    for face = 1 : nfaces
        append!(xFaceDofs,[face])
    end
    # save dofmap
    FES.FaceDofs = xFaceDofs
end

function init_dofmap!(FES::FESpace{FEType}, ::Type{AssemblyTypeBFACE}) where {FEType <: HDIVRT0}
    xBFaces = FES.xgrid[BFaces]
    nbfaces = length(xBFaces)
    xBFaceDofs = VariableTargetAdjacency(Int32)
    for bface = 1: nbfaces
        append!(xBFaceDofs,[xBFaces[bface]])
    end
    # save dofmap
    FES.BFaceDofs = xBFaceDofs
end


function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{HDIVRT0}, exact_function!::Function; dofs = [], bonus_quadorder::Int = 0)
    # todo
    # integrate normal flux of exact_function over edges
end


function get_basis_normalflux_on_face(::Type{HDIVRT0{2}}, ::Type{<:Edge1D})
    function closure(xref)
        return [1.0]
    end
end

function get_basis_on_cell(::Type{HDIVRT0{2}}, ::Type{<:Triangle2D})
    function closure(xref)
        return [xref[1] xref[2]-1.0;
                xref[1] xref[2];
                xref[1]-1.0 xref[2]]
    end
end


function get_basis_on_cell(::Type{HDIVRT0{2}}, ::Type{<:Quadrilateral2D})
    function closure(xref)
        return [[0.0 xref[2]-1.0];
                [xref[1] 0.0];
                [0.0 xref[2]];
                [xref[1]-1.0 0.0]] 
    end
end


function get_coefficients_on_cell!(coefficients, FE::FESpace{HDIVRT0{2}}, ::Type{<:Triangle2D}, cell::Int)
    # multiplication with normal vectors
    coefficients[1,1] = FE.xCellFaceSigns[1,cell];
    coefficients[2,1] = FE.xCellFaceSigns[1,cell];
    coefficients[1,2] = FE.xCellFaceSigns[2,cell];
    coefficients[2,2] = FE.xCellFaceSigns[2,cell];
    coefficients[1,3] = FE.xCellFaceSigns[3,cell];
    coefficients[2,3] = FE.xCellFaceSigns[3,cell];
end    
function get_coefficients_on_cell!(coefficients, FE::FESpace{HDIVRT0{2}}, ::Type{<:Quadrilateral2D}, cell::Int,)
    # multiplication with normal vectors
    fill!(coefficients,1.0)
    coefficients[1,1] = FE.xCellFaceSigns[1,cell];
    coefficients[2,1] = FE.xCellFaceSigns[1,cell];
    coefficients[1,2] = FE.xCellFaceSigns[2,cell];
    coefficients[2,2] = FE.xCellFaceSigns[2,cell];
    coefficients[1,3] = FE.xCellFaceSigns[3,cell];
    coefficients[2,3] = FE.xCellFaceSigns[3,cell];
    coefficients[1,4] = FE.xCellFaceSigns[4,cell];
    coefficients[2,4] = FE.xCellFaceSigns[4,cell];
end   


function get_coefficients_on_face!(coefficients, FE::FESpace{HDIVRT0{2}}, ::Type{<:Edge1D}, face::Int)
    # multiplication with normal vectors
    fill!(coefficients,1.0)
end    
