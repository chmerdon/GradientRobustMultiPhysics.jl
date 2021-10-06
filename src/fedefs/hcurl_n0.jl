"""
````
abstract type HCURLN0{edim} <: AbstractHcurlFiniteElement where {edim<:Int}
````

Hcurl-conforming vector-valued (ncomponents = edim) lowest-order Nedelec space.

allowed ElementGeometries:
- Triangle2D
- Quadrilateral2D
- Tetrahedron3D
"""
abstract type HCURLN0{edim} <: AbstractHcurlFiniteElement where {edim<:Int} end

function Base.show(io::Core.IO, ::Type{<:HCURLN0{edim}}) where {edim}
    print(io,"HCURLN0{$edim}")
end

get_ncomponents(FEType::Type{<:HCURLN0}) = FEType.parameters[1]
get_ndofs(::Union{Type{<:ON_EDGES}, Type{<:ON_BEDGES}, Type{<:ON_FACES}, Type{<:ON_BFACES}}, FEType::Type{<:HCURLN0}, EG::Type{<:AbstractElementGeometry}) = 1
get_ndofs(::Type{ON_CELLS}, FEType::Type{HCURLN0{2}}, EG::Type{<:AbstractElementGeometry}) = num_faces(EG)
get_ndofs(::Type{ON_CELLS}, FEType::Type{HCURLN0{3}}, EG::Type{<:AbstractElementGeometry}) = num_edges(EG)

get_polynomialorder(::Type{<:HCURLN0{2}}, ::Type{<:AbstractElementGeometry1D}) = 0;
get_polynomialorder(::Type{<:HCURLN0{2}}, ::Type{<:AbstractElementGeometry2D}) = 1;
get_polynomialorder(::Type{<:HCURLN0{3}}, ::Type{<:AbstractElementGeometry1D}) = 0;
get_polynomialorder(::Type{<:HCURLN0{3}}, ::Type{<:AbstractElementGeometry3D}) = 1;

get_dofmap_pattern(FEType::Type{<:HCURLN0{2}}, ::Type{CellDofs}, EG::Type{<:AbstractElementGeometry2D}) = "f1"
get_dofmap_pattern(FEType::Type{<:HCURLN0{2}}, ::Type{FaceDofs}, EG::Type{<:AbstractElementGeometry1D}) = "i1"
get_dofmap_pattern(FEType::Type{<:HCURLN0{2}}, ::Type{BFaceDofs}, EG::Type{<:AbstractElementGeometry1D}) = "i1"

get_dofmap_pattern(FEType::Type{<:HCURLN0{3}}, ::Type{CellDofs}, EG::Type{<:AbstractElementGeometry3D}) = "e1"
get_dofmap_pattern(FEType::Type{<:HCURLN0{3}}, ::Type{FaceDofs}, EG::Type{<:AbstractElementGeometry2D}) = "e1"
get_dofmap_pattern(FEType::Type{<:HCURLN0{3}}, ::Type{EdgeDofs}, EG::Type{<:AbstractElementGeometry1D}) = "i1"
get_dofmap_pattern(FEType::Type{<:HCURLN0{3}}, ::Type{BFaceDofs}, EG::Type{<:AbstractElementGeometry2D}) = "e1"
get_dofmap_pattern(FEType::Type{<:HCURLN0{3}}, ::Type{BEdgeDofs}, EG::Type{<:AbstractElementGeometry1D}) = "i1"

isdefined(FEType::Type{<:HCURLN0}, ::Type{<:Triangle2D}) = true
isdefined(FEType::Type{<:HCURLN0}, ::Type{<:Quadrilateral2D}) = true
isdefined(FEType::Type{<:HCURLN0}, ::Type{<:Tetrahedron3D}) = true

function interpolate!(Target::AbstractArray{T,1}, FE::FESpace{Tv,Ti,FEType,APT}, ::Type{ON_EDGES}, exact_function!; items = [], time = 0) where {T,Tv,Ti,FEType <: HCURLN0,APT}
    edim = get_ncomponents(FEType)
    if edim == 3
        if items == []
            items = 1 : nitems
        end
        ncomponents = get_ncomponents(eltype(FE))
        xEdgeTangents = FE.xgrid[EdgeTangents]
        function tangentflux_eval3d()
            temp = zeros(T,ncomponents)
            function closure(result, x, edge)
                eval_data!(temp, exact_function!, x, time) 
                result[1] = temp[1] * xEdgeTangents[1,edge]
                result[1] += temp[2] * xEdgeTangents[2,edge]
                result[1] += temp[3] * xEdgeTangents[3,edge]
            end   
        end   
        edata_function = ExtendedDataFunction(tangentflux_eval3d(), [1, ncomponents]; dependencies = "XI", quadorder = exact_function!.quadorder)
        integrate!(Target, FE.xgrid, ON_EDGES, edata_function; items = items, time = time)
    end
end

function interpolate!(Target::AbstractArray{T,1}, FE::FESpace{Tv,Ti,FEType,APT}, ::Type{ON_FACES}, exact_function!; items = [], time = 0) where {T,Tv,Ti,FEType <: HCURLN0,APT}
    edim = get_ncomponents(FEType)
    if edim == 2
        if items == []
            items = 1 : nitems
        end
        ncomponents = get_ncomponents(eltype(FE))
        xFaceNormals = FE.xgrid[FaceNormals]
        function tangentflux_eval2d()
            temp = zeros(T,ncomponents)
            function closure(result, x, face)
                eval_data!(temp, exact_function!, x, time) 
                result[1] = - temp[1] * xFaceNormals[2,face] # rotated normal = tangent
                result[1] += temp[2] * xFaceNormals[1,face]
            end   
        end   
        edata_function = ExtendedDataFunction(tangentflux_eval2d(), [1, ncomponents]; dependencies = "XI", quadorder = exact_function!.quadorder)
        integrate!(Target, FE.xgrid, ON_FACES, edata_function; items = items, time = time)
    elseif edim == 3
        # delegate face edges to edge interpolation
        subitems = slice(FE.xgrid[FaceEdges], items)
        interpolate!(Target, FE, ON_EDGES, exact_function!; items = subitems, time = time)
    end
end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{Tv,Ti,FEType,APT}, ::Type{ON_CELLS}, exact_function!; items = [], time = 0) where {Tv,Ti,FEType <: HCURLN0,APT}
    edim = get_ncomponents(FEType)
    if edim == 2
        # delegate cell faces to face interpolation
        subitems = slice(FE.xgrid[CellFaces], items)
        interpolate!(Target, FE, ON_FACES, exact_function!; items = subitems, time = time)
    elseif edim == 3
        # delegate cell edges to edge interpolation
        subitems = slice(FE.xgrid[CellEdges], items)
        interpolate!(Target, FE, ON_EDGES, exact_function!; items = subitems, time = time)
    end
end

# on faces dofs are only tangential fluxes
function get_basis(::Union{Type{<:ON_EDGES}, Type{<:ON_BEDGES}, Type{<:ON_BFACES}, Type{<:ON_FACES}}, ::Type{<:HCURLN0}, ::Type{<:AbstractElementGeometry})
    function closure(refbasis,xref)
        refbasis[1,1] = 1
    end
end

function get_basis(::Type{ON_CELLS}, ::Type{HCURLN0{2}}, ::Type{<:Triangle2D})
    function closure(refbasis,xref)
        refbasis[1,:] .= [1.0-xref[2], xref[1]]
        refbasis[2,:] .= [-xref[2], xref[1]]
        refbasis[3,:] .= [-xref[2], xref[1]-1.0]
    end
end

function get_basis(::Type{ON_CELLS}, ::Type{HCURLN0{2}}, ::Type{<:Quadrilateral2D})
    function closure(refbasis,xref)
        refbasis[1,:] .= [1 - xref[2], 0.0]
        refbasis[2,:] .= [0.0, xref[1]]
        refbasis[3,:] .= [-xref[2], 0.0]
        refbasis[4,:] .= [0.0, xref[1]-1.0]
    end
end

function get_basis(::Type{ON_CELLS}, ::Type{HCURLN0{3}}, ::Type{<:Tetrahedron3D})
    function closure(refbasis,xref)
        refbasis[1,:] .= [1.0-xref[2]-xref[3], xref[1], xref[1]] # edge 1 = [1,2]
        refbasis[2,:] .= [xref[2], 1-xref[3]-xref[1], xref[2]]   # edge 2 = [1,3]
        refbasis[3,:] .= [xref[3], xref[3], 1-xref[1]-xref[2]]
        refbasis[4,:] .= [-xref[2], xref[1], 0]
        refbasis[5,:] .= [-xref[3], 0, xref[1]]
        refbasis[6,:] .= [0, -xref[3], xref[2]]
    end
end

function get_coefficients(::Type{ON_CELLS}, FE::FESpace{Tv,Ti,<:HCURLN0,APT}, EG::Type{<:AbstractElementGeometry2D}) where {Tv,Ti,APT}
    xCellFaceSigns = FE.xgrid[CellFaceSigns]
    nfaces = num_faces(EG)
    function closure(coefficients, cell)
        # multiplication with normal vector signs
        for j = 1 : nfaces,  k = 1 : size(coefficients,1)
            coefficients[k,j] = xCellFaceSigns[j,cell];
        end
        return nothing
    end
end   

function get_coefficients(::Type{ON_CELLS}, FE::FESpace{Tv,Ti,<:HCURLN0,APT}, EG::Type{<:AbstractElementGeometry3D}) where {Tv,Ti,APT}
    xCellEdgeSigns = FE.xgrid[CellEdgeSigns]
    nedges = num_edges(EG)
    function closure(coefficients, cell)
        # multiplication with normal vector signs
        for j = 1 : nedges,  k = 1 : size(coefficients,1)
            coefficients[k,j] = xCellEdgeSigns[j,cell];
        end
        return nothing
    end
end     
