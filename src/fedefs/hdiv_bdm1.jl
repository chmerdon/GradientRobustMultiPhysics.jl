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
get_ndofs_on_cell_all(FEType::Type{<:HDIVBDM1}, EG::Type{<:AbstractElementGeometry3D}) = 4*nfaces_for_geometry(EG)

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
    integrate!(Target, FE.xgrid, ON_FACES, normalflux_eval(), bonus_quadorder, 1; items = items, item_dependent_integrand = true)
   
    # integrate normal flux with linear weight (x[1] - 1//2) of exact_function over edges
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
        refbasis[2,1] = (24*(xref[1] - 1//3) + 12*(xref[2] - 1//3)) # linear normal-flux of first BDM1 function
        refbasis[3,1] = (24*(xref[2] - 1//3) + 12*(xref[1] - 1//3)) # linear normal-flux of second BDM1 function
    end
end

function get_basis_on_cell(::Type{HDIVBDM1{3}}, ::Type{<:Tetrahedron3D})
    function closure(refbasis, xref)
        # RT0 basis
        refbasis[1,:] .= 2*[xref[1], xref[2], xref[3]-1]
        refbasis[5,:] .= 2*[xref[1], xref[2]-1, xref[3]]
        refbasis[9,:] .= 2*[xref[1], xref[2], xref[3]]
        refbasis[13,:] .= 2*[xref[1]-1, xref[2], xref[3]]
        # additional BDM1 functions on faces
        # note: we define three additional functions per face
        #       and later select only two linear independent ones that match the local enumeration/orientation
        #       of the global/local face nodes + a possible sign change managed by coefficient_handler
        # face = [1 3 2]
        temp = 1 - xref[1] - xref[2] - xref[3]
        # FACE1 [1,3,2], normal = [0,0,-1], |E| = 1/2, xref[3] = 0
        # phi = [-gamma*phi2,-beta*phi3,alpha*phi1+beta*phi3+gamma*phi2]
        # [J1,J2,J3] = linear moments of normal flux weighted with (phi_1-1/3), (phi_3-1/3), (phi_2-1/3)
        refbasis[2,:] .= 2*[12*xref[1],0,12*temp-12*xref[1]]               # [1,0,-1]
        refbasis[3,:] .= 2*[0,-12*xref[2],-12*temp+12*xref[2]]             # [0,-1,1]
        refbasis[4,:] .= 2*[-12*xref[1],12*xref[2],-12*xref[2]+12*xref[1]] # [-1,1,0]
        
        # FACE2 [1 2 4], normal = [0,-1,0], |E| = 1/2, xref[2] = 0
        # phi = [-beta*phi2,alpha*phi1+beta*phi2+gamma*phi4,-gamma*phi4]
        # [J1,J2,J3] = linear moments of normal flux weighted with (phi_1-1/3), (phi_2-1/3), (phi_4-1/3)
        refbasis[6,:] .= 2*[0,12*temp-12*xref[3],12*xref[3]]      # [1,0,-1]
        refbasis[7,:] .= 2*[-12*xref[1],-12*temp+12*xref[1],0]    # [0,-1,1]
        refbasis[8,:] .= 2*[12*xref[1],-12*xref[1]+12*xref[3],-12*xref[3]] # [-1,1,0]

        # FACE3 [2 3 4], normal = [1,1,1]/sqrt(3), |E| = sqrt(3)/2, xref[1]+xref[2]+xref[3] = 1
        # phi = [alpha*phi2,beta*phi3,gamma*phi4]
        # [J1,J2,J3] = linear moments of normal flux weighted with (phi_2-1/3), (phi_3-1/3), (phi_4-1/3)
        refbasis[10,:] .= -2*[12*xref[1],0,-12*xref[3]]  # [1,0,-1]
        refbasis[11,:] .= -2*[-12*xref[1],12*xref[2],0]  # [0,-1,1]
        refbasis[12,:] .= -2*[0,-12*xref[2],12*xref[3]]  # [-1,1,0]
        
        # FACE4 [1 4 3], normal = [-1,0,0], |E| = 1/2, xref[1] = 0
        # phi = [alpha*phi1+beta*phi4+gamma*phi3,-gamma*phi3,-beta*phi4]
        # [J1,J2,J3] = linear moments of normal flux weighted with (phi_1-1/3), (phi_4-1/3), (phi_3-1/3)
        refbasis[14,:] .= 2*[12*temp-12*xref[2],12*xref[2],0]               # [1,0,-1]
        refbasis[15,:] .= 2*[-12*temp+12*xref[3],0,-12*xref[3]]             # [0,-1,1]
        refbasis[16,:] .= 2*[-12*xref[3]+12*xref[2],-12*xref[2],+12*xref[3]] # [-1,1,0]
    end

    # ##testing
    # refbasi = zeros(Rational,16,3)
    # closure(refbasi,[1//3 1//3 0])
    # println("refbasi normalflux at FACE 1 midpoint = $(-refbasi[:,3])")
    # closure(refbasi,[1//3 0 1//3])
    # println("refbasi normalflux at FACE 2 midpoint = $(-refbasi[:,2])")
    # closure(refbasi,[1//3 1//3 1//3])
    # println("refbasi normalflux at FACE 3 midpoint = $(refbasi[:,1] .+ refbasi[:,2] .+ refbasi[:,3])")
    # closure(refbasi,[0 1//3 1//3])
    # println("refbasi normalflux at FACE 4 midpoint = $(-refbasi[:,1])")

    # closure(refbasi,[2//3 1//3 0])
    # println("refbasi normalflux at FACE 1 node 1 = $(-refbasi[:,3])")
    # closure(refbasi,[0 2//3 0])
    # println("refbasi normalflux at FACE 1 node 2 = $(-refbasi[:,3])")
    # closure(refbasi,[1//3 0 0])
    # println("refbasi normalflux at FACE 1 node 3 = $(-refbasi[:,3])")

    # closure(refbasi,[1//3 0 2//3])
    # println("refbasi normalflux at FACE 2 node 1 = $(-refbasi[:,2])")
    # closure(refbasi,[0 0 1//3])
    # println("refbasi normalflux at FACE 2 node 2 = $(-refbasi[:,2])")
    # closure(refbasi,[2//3 0 0])
    # println("refbasi normalflux at FACE 2 node 3 = $(-refbasi[:,2])")

    # closure(refbasi,[0 1//3 2//3])
    # println("refbasi normalflux at FACE 3 node 1 = $(refbasi[:,1] .+ refbasi[:,2] .+ refbasi[:,3])")
    # closure(refbasi,[2//3 0 1//3])
    # println("refbasi normalflux at FACE 3 node 2 = $(refbasi[:,1] .+ refbasi[:,2] .+ refbasi[:,3])")
    # closure(refbasi,[1//3 2//3 0])
    # println("refbasi normalflux at FACE 3 node 3 = $(refbasi[:,1] .+ refbasi[:,2] .+ refbasi[:,3])")

    # closure(refbasi,[0 1//3 0])
    # println("refbasi normalflux at FACE 4 node 1 = $(-refbasi[:,1])")
    # closure(refbasi,[0 2//3 1//3])
    # println("refbasi normalflux at FACE 4 node 2 = $(-refbasi[:,1])")
    # closure(refbasi,[0 0 2//3])
    # println("refbasi normalflux at FACE 4 node 3 = $(-refbasi[:,1])")

    return closure
end


function get_coefficients_on_cell!(FE::FESpace{<:HDIVBDM1}, EG::Type{<:AbstractElementGeometry2D})
    xCellFaceSigns = FE.xgrid[CellFaceSigns]
    nfaces = nfaces_for_geometry(EG)
    function closure(coefficients, cell)
        fill!(coefficients,1.0)
        # multiplication with normal vector signs (only RT0)
        for j = 1 : nfaces,  k = 1 : size(coefficients,1)
            coefficients[k,2*j-1] = xCellFaceSigns[j,cell];
        end
        return nothing
    end
end  


function get_coefficients_on_cell!(FE::FESpace{<:HDIVBDM1}, EG::Type{<:AbstractElementGeometry3D})
    xCellFaceSigns = FE.xgrid[CellFaceSigns]
    nfaces = nfaces_for_geometry(EG)
    function closure(coefficients, cell)
        fill!(coefficients,1.0)
        for j = 1 : nfaces,  k = 1 : size(coefficients,1)
            coefficients[k,3*j-2] = xCellFaceSigns[j,cell]; # RT0
            coefficients[k,3*j-1] = -1;
            coefficients[k,3*j] = 1;
        end
        return nothing
    end
end  

function get_basissubset_on_cell!(FE::FESpace{<:HDIVBDM1}, EG::Type{<:AbstractElementGeometry3D})
    xCellFaceOrientations = FE.xgrid[CellFaceOrientations]
    xCellFaces = FE.xgrid[CellFaces]
    xFaceNodes = FE.xgrid[FaceNodes]
    xCellNodes = FE.xgrid[CellNodes]
    nfaces = nfaces_for_geometry(EG)
    orientation::Int32 = 0
    face::Int32 = 0
    face_rule = face_enum_rule(EG)
    function closure(subset_ids, cell)
        for j = 1 : nfaces
            subset_ids[3*j-2] = 4*j-3; # always take the RT0 function
            orientation = xCellFaceOrientations[j,cell]
            face = xCellFaces[j,cell]
            if orientation > 0 # cellface nodes and face nodes coincide
                subset_ids[3*j-1] = 4*j-1; # second BDM1 function
                subset_ids[3*j] = 4*j-2; # first BDM1 function
            elseif orientation == -1 # first cellface node and face node coincide (otherwise reversed order)
                subset_ids[3*j-1] = 4*j-2; # third
                subset_ids[3*j] = 4*j-1; # second
            elseif orientation == -2 # second cellface node and face node coincide (otherwise reversed order)
                subset_ids[3*j-1] = 4*j;
                subset_ids[3*j] = 4*j-2;
            elseif orientation == -3 # third cellface node and face node coincide (otherwise reversed order)
                subset_ids[3*j-1] = 4*j-1; # first
                subset_ids[3*j] = 4*j; # second
            end

        end
        return nothing
    end
end  
 