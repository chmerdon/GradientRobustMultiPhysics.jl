"""
$(TYPEDEF)

Hdiv-conforming vector-valued (ncomponents = edim) lowest-order Raviart-Thomas space

allowed ElementGeometries:
- Triangle2D
- Quadrilateral2D
- Tetrahedron3D
- Hexahedron3D
"""
abstract type HDIVRT0{edim} <: AbstractHdivFiniteElement where {edim<:Int} end

get_ncomponents(FEType::Type{<:HDIVRT0}) = FEType.parameters[1]
get_ndofs_on_face(FEType::Type{<:HDIVRT0}, EG::Type{<:AbstractElementGeometry}) = 1
get_ndofs_on_cell(FEType::Type{<:HDIVRT0}, EG::Type{<:AbstractElementGeometry}) = nfaces_for_geometry(EG)

get_polynomialorder(::Type{<:HDIVRT0{2}}, ::Type{<:AbstractElementGeometry1D}) = 0;
get_polynomialorder(::Type{<:HDIVRT0{2}}, ::Type{<:AbstractElementGeometry2D}) = 1;
get_polynomialorder(::Type{<:HDIVRT0{3}}, ::Type{<:AbstractElementGeometry2D}) = 0;
get_polynomialorder(::Type{<:HDIVRT0{3}}, ::Type{<:AbstractElementGeometry3D}) = 1;

get_dofmap_pattern(FEType::Type{<:HDIVRT0}, ::Type{CellDofs}, EG::Type{<:AbstractElementGeometry}) = "f1"
get_dofmap_pattern(FEType::Type{<:HDIVRT0}, ::Type{FaceDofs}, EG::Type{<:AbstractElementGeometry}) = "i1"
get_dofmap_pattern(FEType::Type{<:HDIVRT0}, ::Type{BFaceDofs}, EG::Type{<:AbstractElementGeometry}) = "i1"

function init!(FES::FESpace{FEType}) where {FEType <: HDIVRT0}
    ncomponents = get_ncomponents(FEType)
    FES.name = "RT0 (Hdiv, $(ncomponents)d)"

    # count number of dofs
    nfaces = num_sources(FES.xgrid[FaceNodes])
    FES.ndofs = nfaces

end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{ON_FACES}, exact_function!::Function; items = [], bonus_quadorder::Int = 0) where {FEType <: HDIVRT0}
    ncomponents = get_ncomponents(FEType)
    if items == []
        items = 1 : num_sources(FE.xgrid[FaceNodes])
    end

    # compute exact face means
    xFaceNormals = FE.xgrid[FaceNormals]
    function normalflux_eval()
        temp = zeros(Float64,ncomponents)
        function closure(result, x, face, xref)
            exact_function!(temp,x)
            result[1] = 0.0
            for j = 1 : ncomponents
                result[1] += temp[j] * xFaceNormals[j,face]
            end 
        end   
    end   
    integrate!(Target, FE.xgrid, ON_FACES, normalflux_eval(), bonus_quadorder, 1; items = items, item_dependent_integrand = true)

end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{ON_CELLS}, exact_function!::Function; items = [], bonus_quadorder::Int = 0) where {FEType <: HDIVRT0}
    # delegate cell faces to node interpolation
    subitems = slice(FE.xgrid[CellFaces], items)
    interpolate!(Target, FE, ON_FACES, exact_function!; items = subitems, bonus_quadorder = bonus_quadorder)
end


function get_basis_normalflux_on_face(::Type{<:HDIVRT0}, ::Type{<:AbstractElementGeometry})
    function closure(refbasis, xref)
        refbasis[1,1] = 1
    end
end

function get_basis_on_cell(::Type{HDIVRT0{2}}, ::Type{<:Triangle2D})
    function closure(refbasis, xref)
        refbasis[1,:] .= [xref[1], xref[2]-1.0]
        refbasis[2,:] .= [xref[1], xref[2]]
        refbasis[3,:] .= [xref[1]-1.0, xref[2]]
    end
end

function get_basis_on_cell(::Type{HDIVRT0{2}}, ::Type{<:Quadrilateral2D})
    function closure(refbasis, xref)
        refbasis[1,:] .= [0.0, xref[2]-1.0]
        refbasis[2,:] .= [xref[1], 0.0]
        refbasis[3,:] .= [0.0, xref[2]]
        refbasis[4,:] .= [xref[1]-1.0, 0.0]
    end
end

function get_basis_on_cell(::Type{HDIVRT0{3}}, ::Type{<:Tetrahedron3D})
    function closure(refbasis, xref)
        refbasis[1,:] .= 2*[xref[1], xref[2], xref[3]-1.0]
        refbasis[2,:] .= 2*[xref[1], xref[2]-1.0, xref[3]]
        refbasis[3,:] .= 2*[xref[1], xref[2], xref[3]]
        refbasis[4,:] .= 2*[xref[1]-1.0, xref[2], xref[3]]
    end
    # note: factor 2 is chosen, such that normal-flux integrated over faces is 1 again
end

function get_basis_on_cell(::Type{HDIVRT0{3}}, ::Type{<:Hexahedron3D})
    function closure(refbasis, xref)
        refbasis[1,:] .= [0.0, 0.0, xref[3]-1.0]
        refbasis[2,:] .= [0.0, xref[2]-1.0, 0.0]
        refbasis[3,:] .= [xref[1], 0.0, 0.0]
        refbasis[4,:] .= [0.0, xref[2], 0.0]
        refbasis[5,:] .= [xref[1]-1.0, 0.0, 0.0]
        refbasis[6,:] .= [0.0, 0.0, xref[3]]
    end
end


function get_coefficients_on_cell!(FE::FESpace{<:HDIVRT0}, EG::Type{<:AbstractElementGeometry})
    xCellFaceSigns = FE.xgrid[CellFaceSigns]
    nfaces = nfaces_for_geometry(EG)
    function closure(coefficients, cell)
        # multiplication with normal vector signs
        for j = 1 : nfaces,  k = 1 : size(coefficients,1)
            coefficients[k,j] = xCellFaceSigns[j,cell];
        end
        return nothing
    end
end    
