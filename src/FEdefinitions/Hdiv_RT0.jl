"""
$(TYPEDEF)

Hdiv-conforming vector-valued (ncomponents = spacedim) lowest-order Raviart-Thomas space

allowed ElementGeometries:
- Triangle2D
- Quadrilateral2D
- Tetrahedron3D
- Hexahedron3D
"""
abstract type HDIVRT0{spacedim} <: AbstractHdivFiniteElement where {spacedim<:Int} end

get_ncomponents(::Type{<:HDIVRT0{2}}) = 2
get_ncomponents(::Type{<:HDIVRT0{3}}) = 3

get_polynomialorder(::Type{<:HDIVRT0{2}}, ::Type{<:AbstractElementGeometry1D}) = 0;
get_polynomialorder(::Type{<:HDIVRT0{2}}, ::Type{<:AbstractElementGeometry2D}) = 1;
get_polynomialorder(::Type{<:HDIVRT0{3}}, ::Type{<:AbstractElementGeometry2D}) = 0;
get_polynomialorder(::Type{<:HDIVRT0{3}}, ::Type{<:AbstractElementGeometry3D}) = 1;


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


function get_basis_normalflux_on_face(::Type{<:HDIVRT0}, ::Type{<:AbstractElementGeometry})
    function closure(xref)
        return [1]
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
        return [0.0 xref[2]-1.0;
                xref[1] 0.0;
                0.0 xref[2];
                xref[1]-1.0 0.0] 
    end
end

function get_basis_on_cell(::Type{HDIVRT0{3}}, ::Type{<:Tetrahedron3D})
    function closure(xref)
        return 2*[xref[1] xref[2] xref[3]-1.0;
                xref[1] xref[2]-1.0 xref[3];
                xref[1] xref[2] xref[3];
                xref[1]-1.0 xref[2] xref[3]]
        # note: factor 2 is chosen, such that normal-flux integrated over faces is 1 again
    end
end

function get_basis_on_cell(::Type{HDIVRT0{3}}, ::Type{<:Hexahedron3D})
    function closure(xref)
        return [0.0 0.0 xref[3]-1.0;
                0.0 xref[2]-1.0 0.0;
                xref[1] 0.0 0.0;
                0.0 xref[2] 0.0;
                xref[1]-1.0 0.0 0.0;
                0.0 0.0 xref[3]] 
    end
end


function get_coefficients_on_cell!(coefficients, FE::FESpace{<:HDIVRT0}, EG::Type{<:AbstractElementGeometry}, cell::Int)
    # multiplication with normal vector signs
    for j = 1 : nfaces_for_geometry(EG),  k = 1 : size(coefficients,1)
        coefficients[k,j] = FE.xCellFaceSigns[j,cell];
    end
end    
