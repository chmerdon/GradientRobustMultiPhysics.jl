"""
$(TYPEDEF)

Hdiv-conforming vector-valued (ncomponents = edim) lowest-order Brezzi-Douglas-Marini space

allowed ElementGeometries:
- Triangle2D
- Quadrilateral2D
"""
abstract type HDIVBDM1{edim} <: AbstractHdivFiniteElement where {edim<:Int} end

get_ncomponents(FEType::Type{<:HDIVBDM1}) = FEType.parameters[1]
get_ndofs_on_face(FEType::Type{<:HDIVBDM1}, EG::Type{<:AbstractElementGeometry}) = 2
get_ndofs_on_cell(FEType::Type{<:HDIVBDM1}, EG::Type{<:AbstractElementGeometry}) = 2*nfaces_for_geometry(EG)

get_polynomialorder(::Type{<:HDIVBDM1{2}}, ::Type{<:Edge1D}) = 1;
get_polynomialorder(::Type{<:HDIVBDM1{2}}, ::Type{<:Triangle2D}) = 1;
get_polynomialorder(::Type{<:HDIVBDM1{2}}, ::Type{<:Quadrilateral2D}) = 2;

get_dofmap_pattern(FEType::Type{<:HDIVBDM1{2}}, ::Type{CellDofs}, EG::Type{<:AbstractElementGeometry}) = "f2"
get_dofmap_pattern(FEType::Type{<:HDIVBDM1{2}}, ::Type{FaceDofs}, EG::Type{<:AbstractElementGeometry}) = "i2"
get_dofmap_pattern(FEType::Type{<:HDIVBDM1{2}}, ::Type{BFaceDofs}, EG::Type{<:AbstractElementGeometry}) = "i2"


function init!(FES::FESpace{FEType}) where {FEType <: HDIVBDM1}
    ncomponents = get_ncomponents(FEType)
    FES.name = "BDM1 (Hdiv, $(ncomponents)d)"

    # count number of dofs
    nfaces = num_sources(FES.xgrid[FaceNodes])
    FES.ndofs = 2*nfaces
end



function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{ON_FACES}, exact_function!::Function; items = [], bonus_quadorder::Int = 0) where {FEType <: HDIVBDM1}
    ncomponents = get_ncomponents(FEType)
    if items == []
        items = 1 : num_sources(FE.xgrid[FaceNodes])
    end

    # integrate normal flux of exact_function over edges
    xFaceNormals = FE.xgrid[FaceNormals]
    nfaces = num_sources(xFaceNormals)
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

    # integrate normal flux with linear weight 2*(x - 1//2) of exact_function over edges
    integrate!(Target, FE.xgrid, ON_FACES, normalflux_eval(), bonus_quadorder, 1; items = items, item_dependent_integrand = true)
    function normalflux2_eval()
        temp = zeros(Float64,ncomponents)
        function closure(result, x, face, xref)
            exact_function!(temp,x)
            result[1] = 0.0
            for j = 1 : ncomponents
                result[1] += temp[j] * xFaceNormals[j,face]
            end
            result[1] *= 2*(xref[1] - 1//2)
        end   
    end   
    integrate!(Target, FE.xgrid, ON_FACES, normalflux2_eval(), bonus_quadorder + 1, 1; items = items, item_dependent_integrand = true, index_offset = nfaces)
end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{ON_CELLS}, exact_function!::Function; items = [], bonus_quadorder::Int = 0) where {FEType <: HDIVBDM1}
    # delegate cell faces to face interpolation
    subitems = slice(FE.xgrid[CellFaces], items)
    interpolate!(Target, FE, ON_FACES, exact_function!; items = subitems, bonus_quadorder = bonus_quadorder)
end


function get_basis_normalflux_on_face(::Type{<:HDIVBDM1}, ::Type{<:AbstractElementGeometry})
    function closure(refbasis,xref)
        refbasis[1,1] = 1
        refbasis[2,1] = 6*(xref[1]- 1//2) # linear normal-flux of BDM1 function
    end
end

function get_basis_on_cell(::Type{HDIVBDM1{2}}, ::Type{<:Triangle2D})
    function closure(refbasis, xref)
        refbasis[1,:] .= [xref[1], xref[2]-1.0]
        refbasis[2,:] .= [3*xref[1], 3-6*xref[1]-3*xref[2]]
        refbasis[3,:] .= [xref[1], xref[2]]
        refbasis[4,:] .= [-3*xref[1], 3*xref[2]]
        refbasis[5,:] .= [xref[1]-1.0, xref[2]]
        refbasis[6,:] .= [-3+3*xref[1]+6*xref[2], -3*xref[2]]
    end
end

function get_basis_on_cell(::Type{HDIVBDM1{2}}, ::Type{<:Quadrilateral2D})
    function closure(refbasis, xref)
        refbasis[1,:] .= [0.0, xref[2]-1.0]
        refbasis[2,:] .= -[3.0*xref[1]*xref[1]-3.0*xref[1], -6.0*xref[1]*xref[2]+6.0*xref[1]+3.0*xref[2]-3.0]
        refbasis[3,:] .= [xref[1], 0.0]
        refbasis[4,:] .= -[-6.0*xref[1]*xref[2]+3.0*xref[1], 3.0*xref[2]*xref[2]-3.0*xref[2]]
        refbasis[5,:] .= [0.0, xref[2]]
        refbasis[6,:] .= -[-3.0*xref[1]*xref[1]+3.0*xref[1], 6.0*xref[1]*xref[2]-3.0*xref[2]]
        refbasis[7,:] .= [xref[1]-1.0, 0.0]
        refbasis[8,:] .= -[6.0*xref[1]*xref[2]-3.0*xref[1]-6*xref[2]+3.0, -3.0*xref[2]*xref[2]+3.0*xref[2]]
    end
end


function get_coefficients_on_cell!(FE::FESpace{<:HDIVBDM1}, EG::Type{<:AbstractElementGeometry})
    xCellFaceSigns = FE.xgrid[CellFaceSigns]
    nfaces = nfaces_for_geometry(EG)
    function closure(coefficients, cell)
        fill!(coefficients,1.0)
        # multiplication with normal vector signs
        for j = 1 : nfaces,  k = 1 : size(coefficients,1)
            coefficients[k,2*j-1] = xCellFaceSigns[j,cell];
        end
        return nothing
    end
end  
