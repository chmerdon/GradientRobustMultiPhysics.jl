"""
$(TYPEDEF)

Hdiv-conforming vector-valued (ncomponents = edim) lowest-order Brezzi-Douglas-Marini space

allowed ElementGeometries:
- Triangle2D
- Quadrilateral2D
"""
abstract type HDIVBDM1{edim} <: AbstractHdivFiniteElement where {edim<:Int} end

get_ncomponents(FEType::Type{<:HDIVBDM1}) = FEType.parameters[1]
get_polynomialorder(::Type{<:HDIVBDM1{2}}, ::Type{<:Edge1D}) = 1;
get_polynomialorder(::Type{<:HDIVBDM1{2}}, ::Type{<:Triangle2D}) = 1;
get_polynomialorder(::Type{<:HDIVBDM1{2}}, ::Type{<:Quadrilateral2D}) = 2;


function init!(FES::FESpace{FEType}) where {FEType <: HDIVBDM1}
    ncomponents = get_ncomponents(FEType)
    FES.name = "BDM1 (H1, $(ncomponents)d)"

    # count number of dofs
    nfaces = num_sources(FES.xgrid[FaceNodes])
    FES.ndofs = 2*nfaces
end

function init_dofmap!(FES::FESpace{FEType}, ::Type{CellDofs}) where {FEType <: HDIVBDM1}
    xCellFaces = FES.xgrid[CellFaces]
    xCellGeometries = FES.xgrid[CellGeometries]
    dofs4item = zeros(Int32,2*max_num_targets_per_source(xCellFaces))
    ncells = num_sources(xCellFaces)
    xCellDofs = VariableTargetAdjacency(Int32)
    nfaces = num_sources(FES.xgrid[FaceNodes])
    nfaces4item = 0
    for cell = 1 : ncells
        nfaces4item = num_targets(xCellFaces,cell)
        for k = 1 : nfaces4item
            dofs4item[2*k-1] = xCellFaces[k,cell]
            dofs4item[2*k] = xCellFaces[k,cell] + nfaces
        end
        append!(xCellDofs,dofs4item[1:2*nfaces4item])
    end
    # save dofmap
    FES.dofmaps[CellDofs] = xCellDofs
end

function init_dofmap!(FES::FESpace{FEType}, ::Type{FaceDofs}) where {FEType <: HDIVBDM1}
    nfaces = num_sources(FES.xgrid[FaceNodes])
    xFaceDofs = VariableTargetAdjacency(Int32)
    for face = 1 : nfaces
        append!(xFaceDofs,[face nfaces+face])
    end
    # save dofmap
    FES.dofmaps[FaceDofs] = xFaceDofs
end

function init_dofmap!(FES::FESpace{FEType}, ::Type{BFaceDofs}) where {FEType <: HDIVBDM1}
    nfaces = num_sources(FES.xgrid[FaceNodes])
    xBFaces = FES.xgrid[BFaces]
    nbfaces = length(xBFaces)
    xBFaceDofs = VariableTargetAdjacency(Int32)
    for bface = 1: nbfaces
        append!(xBFaceDofs,[xBFaces[bface] nfaces + xBFaces[bface]])
    end
    # save dofmap
    FES.dofmaps[BFaceDofs] = xBFaceDofs
end


function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{<:HDIVBDM1}, exact_function!::Function; dofs = [], bonus_quadorder::Int = 1)
   
    # integrate normal flux of exact_function over edges
    ncomponents = get_ncomponents(eltype(FE))
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

    indicesRT0 = Array{Int32,1}(1:2:2*nfaces)
    integrate!(Target, FE.xgrid, ON_FACES, normalflux_eval(), bonus_quadorder, 1; item_dependent_integrand = true)
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
    integrate!(Target, FE.xgrid, ON_FACES, normalflux2_eval(), bonus_quadorder + 1, 1; item_dependent_integrand = true, index_offset = nfaces)
end


function get_basis_normalflux_on_face(::Type{<:HDIVBDM1}, ::Type{<:AbstractElementGeometry})
    function closure(xref)
        return [1;
                6*(xref[1]- 1//2)]; # linear normal-flux of BDM1 function
    end
end

function get_basis_on_cell(::Type{HDIVBDM1{2}}, ::Type{<:Triangle2D})
    function closure(xref)
        return [[xref[1] xref[2]-1.0];
                [3*xref[1] 3-6*xref[1]-3*xref[2]];
                [xref[1] xref[2]];
                [-3*xref[1] 3*xref[2]];
                [xref[1]-1.0 xref[2]];
                [-3+3*xref[1]+6*xref[2] -3*xref[2]]]
    end
end

function get_basis_on_cell(::Type{HDIVBDM1{2}}, ::Type{<:Quadrilateral2D})
    function closure(xref)
        return [[0.0 xref[2]-1.0];
                -[3.0*xref[1]*xref[1]-3.0*xref[1] -6.0*xref[1]*xref[2]+6.0*xref[1]+3.0*xref[2]-3.0];
                [xref[1] 0.0];
                -[-6.0*xref[1]*xref[2]+3.0*xref[1]  3.0*xref[2]*xref[2]-3.0*xref[2]];
                [0.0 xref[2]];
                -[-3.0*xref[1]*xref[1]+3.0*xref[1] 6.0*xref[1]*xref[2]-3.0*xref[2]];
                [xref[1]-1.0 0.0];
                -[6.0*xref[1]*xref[2]-3.0*xref[1]-6*xref[2]+3.0 -3.0*xref[2]*xref[2]+3.0*xref[2]]]
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
