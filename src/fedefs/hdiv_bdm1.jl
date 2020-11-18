"""
$(TYPEDEF)

Hdiv-conforming vector-valued (ncomponents = edim) lowest-order Brezzi-Douglas-Marini space

allowed ElementGeometries:
- Triangle2D
- Quadrilateral2D
- Tetrahedron3D
"""
abstract type HDIVBDM1{edim} <: AbstractHdivFiniteElement where {edim<:Int} end

get_ncomponents(FEType::Type{<:HDIVBDM1}) = FEType.parameters[1]
get_ndofs_on_face(FEType::Type{<:HDIVBDM1}, EG::Type{<:AbstractElementGeometry1D}) = 2
get_ndofs_on_face(FEType::Type{<:HDIVBDM1}, EG::Type{<:AbstractElementGeometry2D}) = 3
get_ndofs_on_cell(FEType::Type{<:HDIVBDM1}, EG::Type{<:AbstractElementGeometry2D}) = 2*nfaces_for_geometry(EG)
get_ndofs_on_cell(FEType::Type{<:HDIVBDM1}, EG::Type{<:AbstractElementGeometry3D}) = 3*nfaces_for_geometry(EG)

get_polynomialorder(::Type{<:HDIVBDM1{2}}, ::Type{<:Edge1D}) = 1;
get_polynomialorder(::Type{<:HDIVBDM1{2}}, ::Type{<:Triangle2D}) = 1;
get_polynomialorder(::Type{<:HDIVBDM1{2}}, ::Type{<:Quadrilateral2D}) = 2;
get_polynomialorder(::Type{<:HDIVBDM1{3}}, ::Type{<:Triangle2D}) = 1;
get_polynomialorder(::Type{<:HDIVBDM1{3}}, ::Type{<:Tetrahedron3D}) = 1;

get_dofmap_pattern(FEType::Type{<:HDIVBDM1{2}}, ::Type{CellDofs}, EG::Type{<:AbstractElementGeometry}) = "f2"
get_dofmap_pattern(FEType::Type{<:HDIVBDM1{2}}, ::Type{FaceDofs}, EG::Type{<:AbstractElementGeometry}) = "i2"
get_dofmap_pattern(FEType::Type{<:HDIVBDM1{2}}, ::Type{BFaceDofs}, EG::Type{<:AbstractElementGeometry}) = "i2"

get_dofmap_pattern(FEType::Type{<:HDIVBDM1{3}}, ::Type{CellDofs}, EG::Type{<:AbstractElementGeometry}) = "f3"
get_dofmap_pattern(FEType::Type{<:HDIVBDM1{3}}, ::Type{FaceDofs}, EG::Type{<:AbstractElementGeometry}) = "i3"
get_dofmap_pattern(FEType::Type{<:HDIVBDM1{3}}, ::Type{BFaceDofs}, EG::Type{<:AbstractElementGeometry}) = "i3"


function init!(FES::FESpace{FEType}) where {FEType <: HDIVBDM1}
    ncomponents = get_ncomponents(FEType)
    FES.name = "BDM1 (Hdiv, $(ncomponents)d)"

    # count number of dofs
    nfaces = num_sources(FES.xgrid[FaceNodes])
    FES.ndofs = ncomponents*nfaces
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

    # integrate normal flux with linear weight (x[1] - 1//2) of exact_function over edges
    integrate!(Target, FE.xgrid, ON_FACES, normalflux_eval(), bonus_quadorder, 1; items = items, item_dependent_integrand = true)
    function normalflux2_eval()
        temp = zeros(Float64,ncomponents)
        function closure(result, x, face, xref)
            exact_function!(temp,x)
            result[1] = 0.0
            for j = 1 : ncomponents
                result[1] += temp[j] * xFaceNormals[j,face]
            end
            result[1] *= (xref[1] - 1//ncomponents)
        end   
    end   
    integrate!(Target, FE.xgrid, ON_FACES, normalflux2_eval(), bonus_quadorder + 1, 1; items = items, item_dependent_integrand = true, index_offset = nfaces)


    # integrate normal flux with linear weight (x[2] - 1//2) of exact_function over edges
    if ncomponents == 3
        integrate!(Target, FE.xgrid, ON_FACES, normalflux_eval(), bonus_quadorder, 1; items = items, item_dependent_integrand = true)
        function normalflux3_eval()
            temp = zeros(Float64,ncomponents)
            function closure(result, x, face, xref)
                exact_function!(temp,x)
                result[1] = 0.0
                for j = 1 : ncomponents
                    result[1] += temp[j] * xFaceNormals[j,face]
                end
                result[1] *= (xref[2] - 1//ncomponents)
            end   
        end   
        integrate!(Target, FE.xgrid, ON_FACES, normalflux3_eval(), bonus_quadorder + 1, 1; items = items, item_dependent_integrand = true, index_offset = 2*nfaces)
    end
end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{ON_CELLS}, exact_function!::Function; items = [], bonus_quadorder::Int = 0) where {FEType <: HDIVBDM1}
    # delegate cell faces to face interpolation
    subitems = slice(FE.xgrid[CellFaces], items)
    interpolate!(Target, FE, ON_FACES, exact_function!; items = subitems, bonus_quadorder = bonus_quadorder)
end


function get_basis_normalflux_on_face(::Type{<:HDIVBDM1}, ::Type{<:AbstractElementGeometry1D})
    function closure(refbasis,xref)
        refbasis[1,1] = 1
        refbasis[2,1] = 12*(xref[1] - 1//2) # linear normal-flux of BDM1 function
    end
end

function get_basis_on_cell(::Type{HDIVBDM1{2}}, ::Type{<:Triangle2D})
    function closure(refbasis, xref)
        # RT0 basis
        refbasis[1,:] .= [xref[1], xref[2]-1]
        refbasis[3,:] .= [xref[1], xref[2]]
        refbasis[5,:] .= [xref[1]-1, xref[2]]
        # additional BDM1 functions on faces
        refbasis[2,:] .= 2*[3*xref[1], 3-6*xref[1]-3*xref[2]]    # = 6*refbasis[1,:] + 12*[0,phi_1]       # phi2-weighted linear moment
        refbasis[4,:] .= 2*[-3*xref[1], 3*xref[2]]               # = 6*refbasis[3,:] + 12*[-phi_2,0]      # phi3-weighted linear moment
        refbasis[6,:] .= 2*[-3+3*xref[1]+6*xref[2], -3*xref[2]]  # = 6*refbasis[5,:] + 12*[phi_3,-phi_3]  # phi1-weighted linear moment
    end
end

function get_basis_on_cell(::Type{HDIVBDM1{2}}, ::Type{<:Quadrilateral2D})
    function closure(refbasis, xref)
        # RT0 basis
        refbasis[1,:] .= [0, xref[2]-1]
        refbasis[3,:] .= [xref[1], 0]
        refbasis[5,:] .= [0, xref[2]]
        refbasis[7,:] .= [xref[1]-1, 0]
        # additional BDM1 functions on faces
        refbasis[2,:] .= -2*[3*xref[1]*xref[1]-3*xref[1], -6*xref[1]*xref[2]+6*xref[1]+3*xref[2]-3]
        refbasis[4,:] .= -2*[-6*xref[1]*xref[2]+3*xref[1], 3*xref[2]*xref[2]-3*xref[2]]
        refbasis[6,:] .= -2*[-3*xref[1]*xref[1]+3*xref[1], 6*xref[1]*xref[2]-3*xref[2]]
        refbasis[8,:] .= -2*[6*xref[1]*xref[2]-3*xref[1]-6*xref[2]+3, -3*xref[2]*xref[2]+3*xref[2]]
    end
end

function get_basis_normalflux_on_face(::Type{<:HDIVBDM1}, ::Type{<:AbstractElementGeometry2D})
    function closure(refbasis,xref)
        refbasis[1,1] = 1
        refbasis[2,1] = 24*(xref[2] - 1//3) + 12*(xref[1] - 1//3) # linear normal-flux of first BDM1 function
        refbasis[3,1] = 24*(xref[1] - 1//3) + 12*(xref[2] - 1//3) # linear normal-flux of second BDM1 function
    end
end

function get_basis_on_cell(::Type{HDIVBDM1{3}}, ::Type{<:Tetrahedron3D})
    function closure(refbasis, xref)
        # RT0 basis
        refbasis[1,:] .= 2*[xref[1], xref[2], xref[3]-1]
        refbasis[4,:] .= 2*[xref[1], xref[2]-1, xref[3]]
        refbasis[7,:] .= 2*[xref[1], xref[2], xref[3]]
        refbasis[10,:] .= 2*[xref[1]-1, xref[2], xref[3]]
        # additional BDM1 functions on faces
        # face = [1 3 2]
        temp = 1 - xref[1] - xref[2] - xref[3]
        refbasis[2,:] .= 12*refbasis[1,:] .+ 48*[                      0,      -1//2 * xref[2],temp + 1//2 * xref[2]] # phi3-weighted linear moment
        refbasis[3,:] .= 12*refbasis[1,:] .+ 48*[        -1//2 * xref[1],                    0,temp + 1//2 * xref[1]] # phi2-weighted linear moment

        # face = [1 2 4]
        refbasis[5,:] .= 12*refbasis[4,:] .+ 48*[        -1//2 * xref[1],temp + 1//2 * xref[1],                    0] # phi2-weighted linear moment
        refbasis[6,:] .= 12*refbasis[4,:] .+ 48*[                      0,temp + 1//2 * xref[3],      -1//2 * xref[3]] # phi4-weighted linear moment

        # face = [2 3 4]
        refbasis[8,:] .= 12*refbasis[7,:] .- 48*[                xref[1],     + 1//2 * xref[2],                    0] # phi3-weighted linear moment
        refbasis[9,:] .= 12*refbasis[7,:] .- 48*[                xref[1],                    0,     + 1//2 * xref[3]] # phi4-weighted linear moment

        # face = [1,4,3]
        refbasis[11,:] .= 12*refbasis[10,:] .+ 48*[temp + 1//2 * xref[3],                    0,      -1//2 * xref[3]] # phi4-weighted linear moment
        refbasis[12,:] .= 12*refbasis[10,:] .+ 48*[temp + 1//2 * xref[2],      -1//2 * xref[2],                    0] # phi3-weighted linear moment
    end
end


function get_coefficients_on_cell!(FE::FESpace{<:HDIVBDM1}, EG::Type{<:AbstractElementGeometry})
    xCellFaceSigns = FE.xgrid[CellFaceSigns]
    nfaces = nfaces_for_geometry(EG)
    function closure(coefficients, cell)
        fill!(coefficients,1.0)
        # multiplication with normal vector signs (only RT0)
        for j = 1 : nfaces,  k = 1 : size(coefficients,1)
            coefficients[k,size(coefficients,1)*(j-1)+1] = xCellFaceSigns[j,cell];
        end
        return nothing
    end
end  
