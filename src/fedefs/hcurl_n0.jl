"""
$(TYPEDEF)

Hcurl-conforming vector-valued (ncomponents = edim) lowest-order Nedelec space

allowed ElementGeometries:
- Triangle2D
- Quadrilateral2D
- Tetrahedron3D
"""
abstract type HCURLN0{edim} <: AbstractHcurlFiniteElement where {edim<:Int} end

get_ncomponents(FEType::Type{<:HCURLN0}) = FEType.parameters[1]

get_polynomialorder(::Type{<:HCURLN0{2}}, ::Type{<:AbstractElementGeometry1D}) = 0;
get_polynomialorder(::Type{<:HCURLN0{2}}, ::Type{<:AbstractElementGeometry2D}) = 1;
get_polynomialorder(::Type{<:HCURLN0{3}}, ::Type{<:AbstractElementGeometry1D}) = 0;
get_polynomialorder(::Type{<:HCURLN0{3}}, ::Type{<:AbstractElementGeometry3D}) = 1;


function init!(FES::FESpace{FEType}) where {FEType <: HCURLN0}
    ncomponents = get_ncomponents(FEType)
    FES.name = "N0 (H1, $(ncomponents)d)"

    # count number of dofs
    edim = get_ncomponents(FEType)
    if edim == 2
        nfaces = num_sources(FES.xgrid[FaceNodes])
        FES.ndofs = nfaces
    elseif edim == 3
        nedges = num_sources(FES.xgrid[EdgeNodes])
        FES.ndofs = nedges
    end
end

function init_dofmap!(FES::FESpace{FEType}, ::Type{CellDofs}) where {FEType <: HCURLN0}
    edim = get_ncomponents(FEType)
    if edim == 2
        FES.dofmaps[CellDofs] = FES.xgrid[CellFaces]
    elseif edim == 3
        FES.dofmaps[CellDofs] = FES.xgrid[CellEdges]
    end
end

function init_dofmap!(FES::FESpace{FEType}, ::Type{FaceDofs}) where {FEType <: HCURLN0}
    edim = get_ncomponents(FEType)
    if edim == 2
        nfaces = num_sources(FES.xgrid[FaceNodes])
        xFaceDofs = VariableTargetAdjacency(Int32)
        for face = 1 : nfaces
            append!(xFaceDofs,[face])
        end
        # save dofmap
        FES.dofmaps[FaceDofs] = xFaceDofs
    elseif edim == 3
        FES.dofmaps[FaceDofs] = FES.xgrid[FaceEdges]
    end
end

function init_dofmap!(FES::FESpace{FEType}, ::Type{BFaceDofs}) where {FEType <: HCURLN0}
    edim = get_ncomponents(FEType)
    if edim == 2
        xBFaces = FES.xgrid[BFaces]
        nbfaces = length(xBFaces)
        xBFaceDofs = VariableTargetAdjacency(Int32)
        for bface = 1: nbfaces
            append!(xBFaceDofs,[xBFaces[bface]])
        end
        # save dofmap
        FES.dofmaps[BFaceDofs] = xBFaceDofs
    elseif edim == 3
        xFaceEdges = FES.xgrid[FaceEdges]
        xBFaces = FES.xgrid[BFaces]
        nbfaces = length(xBFaces)
        xBFaceDofs = VariableTargetAdjacency(Int32)
        for bface = 1: nbfaces
            append!(xBFaceDofs,xFaceEdges[:,xBFaces[bface]])
        end
        # save dofmap
        FES.dofmaps[BFaceDofs] = xBFaceDofs
    end
end


function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, exact_function!::Function; dofs = [], bonus_quadorder::Int = 1) where FEType<:HCURLN0
    # integrate normal flux of exact_function over edges
    edim = get_ncomponents(FEType)
    if edim == 2
        ncomponents = get_ncomponents(eltype(FE))
        xFaceNormals = FE.xgrid[FaceNormals]
        function tangentflux_eval2d()
            temp = zeros(Float64,ncomponents)
            function closure(result, x, face, xref)
                exact_function!(temp,x) 
                result[1] = - temp[1] * xFaceNormals[2,face] # rotated normal = tangent
                result[1] += temp[2] * xFaceNormals[1,face]
            end   
        end   
        integrate!(Target, FE.xgrid, ON_FACES, tangentflux_eval2d(), bonus_quadorder, 1; item_dependent_integrand = true)
    elseif edim == 3
        ncomponents = get_ncomponents(eltype(FE))
        xEdgeTangents = FE.xgrid[EdgeTangents]
        function tangentflux_eval3d()
            temp = zeros(Float64,ncomponents)
            function closure(result, x, edge, xref)
                exact_function!(temp,x) 
                result[1] = temp[1] * xEdgeTangents[1,edge]
                result[1] += temp[2] * xEdgeTangents[2,edge]
                result[1] += temp[3] * xEdgeTangents[3,edge]
            end   
        end   
        integrate!(Target, FE.xgrid, ON_EDGES, tangentflux_eval3d(), bonus_quadorder, 1; item_dependent_integrand = true)
    end
end


function get_basis_tangentflux_on_edge(::Type{<:HCURLN0}, ::Type{<:AbstractElementGeometry})
    function closure(xref)
        return [1]
    end
end

function get_basis_on_cell(::Type{HCURLN0{2}}, ::Type{<:Triangle2D})
    function closure(xref)
        return [1.0 - xref[2] xref[1];
                -xref[2] xref[1];
                -xref[2] xref[1]-1.0]
    end
end

function get_basis_on_cell(::Type{HCURLN0{2}}, ::Type{<:Quadrilateral2D})
    function closure(xref)
        return [1 - xref[2] 0.0;
                0.0 xref[1];
                -xref[2] 0.0;
                0.0 xref[1]-1.0] 
    end
end

function get_basis_on_cell(::Type{HCURLN0{3}}, ::Type{<:Tetrahedron3D})
    function closure(xref)
        return [1.0-xref[2]-xref[3] xref[1] xref[1]; # edge 1 = [1,2]
                xref[2] 1-xref[3]-xref[1] xref[2]; # edge 2 = [1,3]
                xref[3] xref[3] 1-xref[1]-xref[2];
                -xref[2] xref[1] 0;
                -xref[3] 0 xref[1];
                0 -xref[3] xref[2]]
    end
end

function get_coefficients_on_cell!(FE::FESpace{<:HCURLN0}, EG::Type{<:AbstractElementGeometry2D})
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

function get_coefficients_on_cell!(FE::FESpace{<:HCURLN0}, EG::Type{<:AbstractElementGeometry3D})
    xCellEdgeSigns = FE.xgrid[CellEdgeSigns]
    nedges = nedges_for_geometry(EG)
    function closure(coefficients, cell)
        # multiplication with normal vector signs
        for j = 1 : nedges,  k = 1 : size(coefficients,1)
            coefficients[k,j] = xCellEdgeSigns[j,cell];
        end
        return nothing
    end
end     
