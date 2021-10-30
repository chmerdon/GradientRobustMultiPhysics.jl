
"""
````
abstract type H1BR{edim} <: AbstractH1FiniteElementWithCoefficients where {edim<:Int}
````

vector-valued (ncomponents = edim) Bernardi--Raugel element
(first-order polynomials + normal-weighted face bubbles)

allowed ElementGeometries:
- Triangle2D (piecewise linear + normal-weighted face bubbles)
- Quadrilateral2D (Q1 space + normal-weighted face bubbles)
- Tetrahedron3D (piecewise linear + normal-weighted face bubbles)
"""
abstract type H1BR{edim} <: AbstractH1FiniteElementWithCoefficients where {edim<:Int} end

function Base.show(io::Core.IO, ::Type{<:H1BR{edim}}) where {edim}
    print(io,"H1BR{$edim}")
end

get_ncomponents(FEType::Type{<:H1BR}) = FEType.parameters[1]
get_ndofs(::Union{Type{<:ON_FACES}, Type{<:ON_BFACES}}, FEType::Type{<:H1BR}, EG::Type{<:AbstractElementGeometry}) = 1 + num_nodes(EG) * FEType.parameters[1]
get_ndofs(::Type{ON_CELLS}, FEType::Type{<:H1BR}, EG::Type{<:AbstractElementGeometry}) = num_faces(EG) + num_nodes(EG) * FEType.parameters[1]

get_polynomialorder(::Type{<:H1BR{2}}, ::Type{<:Edge1D}) = 2;
get_polynomialorder(::Type{<:H1BR{2}}, ::Type{<:Triangle2D}) = 2;
get_polynomialorder(::Type{<:H1BR{2}}, ::Type{<:Quadrilateral2D}) = 3;
get_polynomialorder(::Type{<:H1BR{3}}, ::Type{<:Triangle2D}) = 3;
get_polynomialorder(::Type{<:H1BR{3}}, ::Type{<:Tetrahedron3D}) = 3;
get_polynomialorder(::Type{<:H1BR{3}}, ::Type{<:Parallelogram2D}) = 4;
get_polynomialorder(::Type{<:H1BR{3}}, ::Type{<:Hexahedron3D}) = 5;

get_dofmap_pattern(FEType::Type{<:H1BR}, ::Type{CellDofs}, EG::Type{<:AbstractElementGeometry}) = "N1f1"
get_dofmap_pattern(FEType::Type{<:H1BR}, ::Type{FaceDofs}, EG::Type{<:AbstractElementGeometry}) = "N1i1"
get_dofmap_pattern(FEType::Type{<:H1BR}, ::Type{BFaceDofs}, EG::Type{<:AbstractElementGeometry}) = "N1i1"

isdefined(FEType::Type{<:H1BR}, ::Type{<:Triangle2D}) = true
isdefined(FEType::Type{<:H1BR}, ::Type{<:Quadrilateral2D}) = true
isdefined(FEType::Type{<:H1BR}, ::Type{<:Tetrahedron3D}) = true


function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{Tv,Ti,FEType,APT}, ::Type{AT_NODES}, exact_function!; items = [], time = 0) where {Tv,Ti,FEType <: H1BR,APT}
    nnodes = size(FE.xgrid[Coordinates],2)
    point_evaluation!(Target, FE, AT_NODES, exact_function!; items = items, component_offset = nnodes, time = time)
end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{Tv,Ti,FEType,APT}, ::Type{ON_EDGES}, exact_function!; items = [], time = 0) where {Tv,Ti,FEType <: H1BR,APT}
    # delegate edge nodes to node interpolation
    subitems = slice(FE.xgrid[EdgeNodes], items)
    interpolate!(Target, FE, AT_NODES, exact_function!; items = subitems, time = time)
end

function interpolate!(Target::AbstractArray{T,1}, FE::FESpace{Tv,Ti,FEType,APT}, ::Type{ON_FACES}, exact_function!; items = [], time = 0) where {T,Tv,Ti, FEType <: H1BR,APT}
    # delegate face nodes to node interpolation
    subitems = slice(FE.xgrid[FaceNodes], items)
    interpolate!(Target, FE, AT_NODES, exact_function!; items = subitems, time = time)

    # preserve face means in normal direction
    xItemVolumes = FE.xgrid[FaceVolumes]
    xItemNodes = FE.xgrid[FaceNodes]
    xItemGeometries = FE.xgrid[FaceGeometries]
    xFaceNormals = FE.xgrid[FaceNormals]
    xItemDofs = FE[FaceDofs]
    nnodes = size(FE.xgrid[Coordinates],2)
    nitems = num_sources(xItemNodes)
    ncomponents = get_ncomponents(FEType)
    offset = ncomponents*nnodes
    if items == []
        items = 1 : nitems
    end

    # compute exact face means
    facemeans = zeros(T,ncomponents,nitems)
    integrate!(facemeans, FE.xgrid, ON_FACES, exact_function!; items = items, time = time)
    P1flux::T = 0
    value::T = 0
    itemEG = Edge1D
    nitemnodes::Int = 0
    for item in items
        itemEG = xItemGeometries[item]
        nitemnodes = num_nodes(itemEG)
        # compute normal flux (minus linear part)
        value = 0
        for c = 1 : ncomponents
            P1flux = 0
            for dof = 1 : nitemnodes
                P1flux += Target[xItemDofs[(c-1)*nitemnodes + dof,item]] * xItemVolumes[item] / nitemnodes
            end
            value += (facemeans[c,item] - P1flux)*xFaceNormals[c,item]
        end
        # set face bubble value
        Target[offset+item] = value / xItemVolumes[item]
    end
end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{Tv,Ti,FEType,APT}, ::Type{ON_CELLS}, exact_function!; items = [], time = 0) where {Tv,Ti,FEType <: H1BR,APT}
    # delegate cell faces to face interpolation
    subitems = slice(FE.xgrid[CellFaces], items)
    interpolate!(Target, FE, ON_FACES, exact_function!; items = subitems, time = time)
end


############
# 2D basis #
############

function get_basis(AT::Union{Type{<:ON_FACES}, Type{<:ON_BFACES}}, FEType::Type{H1BR{2}}, EG::Type{<:Edge1D})
    ncomponents = get_ncomponents(FEType)
    refbasis_P1 = get_basis(AT, H1P1{ncomponents}, EG)
    offset = get_ndofs(AT, H1P1{ncomponents}, EG)
    function closure(refbasis, xref)
        refbasis_P1(refbasis, xref)
        # add face bubble to P1 basis
        refbasis[offset+1,1] = 6 * xref[1] * refbasis[1,1]
        refbasis[offset+1,2] = refbasis[offset+1,1]
    end
end

function get_basis(AT::Type{ON_CELLS}, FEType::Type{H1BR{2}}, EG::Type{<:Triangle2D})
    ncomponents = get_ncomponents(FEType)
    refbasis_P1 = get_basis(AT, H1P1{ncomponents}, EG)
    offset = get_ndofs(AT, H1P1{ncomponents}, EG)
    function closure(refbasis, xref)
        refbasis_P1(refbasis, xref)
        # add face bubbles to P1 basis
        refbasis[offset+1,1] = 6 * xref[1] * refbasis[1,1]
        refbasis[offset+2,1] = 6 * xref[2] * xref[1]
        refbasis[offset+3,1] = 6 * refbasis[1,1] * xref[2]
        refbasis[offset+1,2] = refbasis[offset+1,1]
        refbasis[offset+2,2] = refbasis[offset+2,1]
        refbasis[offset+3,2] = refbasis[offset+3,1]
    end
end


function get_basis(AT::Type{ON_CELLS}, FEType::Type{H1BR{2}}, EG::Type{<:Quadrilateral2D})
    ncomponents = get_ncomponents(FEType)
    refbasis_P1 = get_basis(AT, H1P1{ncomponents}, EG)
    offset = get_ndofs(AT, H1P1{ncomponents}, EG)
    a = 0.0
    b = 0.0
    function closure(refbasis, xref)
        refbasis_P1(refbasis, xref)
        # add face bubbles to Q1 basis
        a = 1 - xref[1]
        b = 1 - xref[2]
        refbasis[offset+1,1] = 6*xref[1]*a*b
        refbasis[offset+2,1] = 6*xref[2]*xref[1]*b
        refbasis[offset+3,1] = 6*xref[1]*xref[2]*a
        refbasis[offset+4,1] = 6*xref[2]*a*b
        refbasis[offset+1,2] = refbasis[offset+1,1]
        refbasis[offset+2,2] = refbasis[offset+2,1]
        refbasis[offset+3,2] = refbasis[offset+3,1]
        refbasis[offset+4,2] = refbasis[offset+4,1]
    end
end

function get_coefficients(::Type{ON_CELLS}, FE::FESpace{Tv,Ti,H1BR{2},APT}, ::Type{<:Triangle2D}) where {Tv,Ti,APT}
    xFaceNormals::Array{Tv,2} = FE.xgrid[FaceNormals]
    xCellFaces = FE.xgrid[CellFaces]
    function closure(coefficients::Array{<:Real,2}, cell)
        fill!(coefficients,1.0)
        coefficients[1,7] = xFaceNormals[1, xCellFaces[1,cell]];
        coefficients[2,7] = xFaceNormals[2, xCellFaces[1,cell]];
        coefficients[1,8] = xFaceNormals[1, xCellFaces[2,cell]];
        coefficients[2,8] = xFaceNormals[2, xCellFaces[2,cell]];
        coefficients[1,9] = xFaceNormals[1, xCellFaces[3,cell]];
        coefficients[2,9] = xFaceNormals[2, xCellFaces[3,cell]];
    end
end

function get_coefficients(::Type{ON_CELLS}, FE::FESpace{Tv,Ti,H1BR{2},APT}, ::Type{<:Quadrilateral2D}) where {Tv,Ti,APT}
    xFaceNormals::Array{Tv,2} = FE.xgrid[FaceNormals]
    xCellFaces = FE.xgrid[CellFaces]
    function closure(coefficients::Array{<:Real,2}, cell)
        fill!(coefficients,1.0)
        coefficients[1,9] = xFaceNormals[1, xCellFaces[1,cell]];
        coefficients[2,9] = xFaceNormals[2, xCellFaces[1,cell]];
        coefficients[1,10] = xFaceNormals[1, xCellFaces[2,cell]];
        coefficients[2,10] = xFaceNormals[2, xCellFaces[2,cell]];
        coefficients[1,11] = xFaceNormals[1, xCellFaces[3,cell]];
        coefficients[2,11] = xFaceNormals[2, xCellFaces[3,cell]];
        coefficients[1,12] = xFaceNormals[1, xCellFaces[4,cell]];
        coefficients[2,12] = xFaceNormals[2, xCellFaces[4,cell]];
    end
end

function get_coefficients(::Type{<:ON_FACES}, FE::FESpace{Tv,Ti,H1BR{2},APT}, ::Type{<:Edge1D}) where {Tv,Ti,APT}
    xFaceNormals::Array{Tv,2} = FE.xgrid[FaceNormals]
    function closure(coefficients::Array{<:Real,2}, face)
        # multiplication of face bubble with normal vector of face
        fill!(coefficients,1.0)
        coefficients[1,5] = xFaceNormals[1, face];
        coefficients[2,5] = xFaceNormals[2, face];
    end
end    

############
# 3D basis #
############

function get_basis(AT::Union{Type{<:ON_FACES}, Type{<:ON_BFACES}}, FEType::Type{H1BR{3}}, EG::Type{<:Triangle2D})
    ncomponents = get_ncomponents(FEType)
    refbasis_P1 = get_basis(AT, H1P1{ncomponents}, EG)
    offset = get_ndofs(AT, H1P1{ncomponents}, EG)
    function closure(refbasis, xref)
        refbasis_P1(refbasis, xref)
        # add face bubbles to P1 basis
        refbasis[offset+1,1] = 60 * xref[1] * refbasis[1,1] * xref[2]
        refbasis[offset+1,2] = refbasis[offset+1,1]
        refbasis[offset+1,3] = refbasis[offset+1,1]
    end
end

function get_basis(AT::Union{Type{<:ON_FACES}, Type{<:ON_BFACES}}, FEType::Type{H1BR{3}}, EG::Type{<:Quadrilateral2D})
    ncomponents = get_ncomponents(FEType)
    refbasis_P1 = get_basis(AT, H1P1{ncomponents}, EG)
    offset = get_ndofs(AT, H1P1{ncomponents}, EG)
    function closure(refbasis, xref)
        refbasis_P1(refbasis, xref)
        # add face bubbles to P1 basis
        refbasis[offset+1,1] = 36 * xref[1] * (1 - xref[1]) * (1 - xref[2]) * xref[2]
        refbasis[offset+1,2] = refbasis[offset+1,1]
        refbasis[offset+1,3] = refbasis[offset+1,1]
    end
end

function get_basis(AT::Type{ON_CELLS}, FEType::Type{H1BR{3}}, EG::Type{<:Tetrahedron3D})
    ncomponents = get_ncomponents(FEType)
    refbasis_P1 = get_basis(AT, H1P1{ncomponents}, EG)
    offset = get_ndofs(AT, H1P1{ncomponents}, EG)
    function closure(refbasis, xref)
        refbasis_P1(refbasis, xref)
        # add face bubbles to P1 basis
        refbasis[offset+1,1] = 60 * xref[1] * refbasis[1,1] * xref[2]
        refbasis[offset+2,1] = 60 * refbasis[1,1] * xref[1] * xref[3]
        refbasis[offset+3,1] = 60 * xref[1] * xref[2] * xref[3]
        refbasis[offset+4,1] = 60 * refbasis[1,1] * xref[2] * xref[3]
        for j = 1 : 4, k = 2 : 3
            refbasis[offset+j,k] = refbasis[offset+j,1]
        end
    end
end

function get_basis(AT::Type{ON_CELLS}, FEType::Type{H1BR{3}}, EG::Type{<:Hexahedron3D})
    ncomponents = get_ncomponents(FEType)
    refbasis_P1 = get_basis(AT, H1P1{ncomponents}, EG)
    offset = get_ndofs(AT, H1P1{ncomponents}, EG)
    a = 0.0
    b = 0.0
    c = 0.0
    function closure(refbasis, xref)
        refbasis_P1(refbasis, xref)
        # add face bubbles to Q1 basis
        a = 1 - xref[1]
        b = 1 - xref[2]
        c = 1 - xref[3]
        refbasis[offset+1,1] = 36*a*b*xref[1]*xref[2]*c # bottom
        refbasis[offset+2,1] = 36*a*xref[1]*c*xref[3]*b # front
        refbasis[offset+3,1] = 36*a*b*c*xref[2]*xref[3] # left
        refbasis[offset+4,1] = 36*a*xref[1]*c*xref[3]*xref[2] # back
        refbasis[offset+5,1] = 36*xref[1]*b*c*xref[2]*xref[3] # right
        refbasis[offset+6,1] = 36*a*b*xref[1]*xref[2]*xref[3] # top
        for j = 1 : 6, k = 2 : 3
            refbasis[offset+j,k] = refbasis[offset+j,1]
        end
    end
end


function get_coefficients(::Type{ON_CELLS},FE::FESpace{Tv,Ti,H1BR{3},APT}, ::Type{<:Tetrahedron3D}) where {Tv,Ti,APT}
    xFaceNormals::Array{Tv,2} = FE.xgrid[FaceNormals]
    xCellFaces::Adjacency{Ti} = FE.xgrid[CellFaces]
    function closure(coefficients::Array{<:Real,2}, cell)
        # multiplication with normal vectors
        fill!(coefficients,1.0)
        coefficients[1,13] = xFaceNormals[1, xCellFaces[1,cell]];
        coefficients[2,13] = xFaceNormals[2, xCellFaces[1,cell]];
        coefficients[3,13] = xFaceNormals[3, xCellFaces[1,cell]];
        coefficients[1,14] = xFaceNormals[1, xCellFaces[2,cell]];
        coefficients[2,14] = xFaceNormals[2, xCellFaces[2,cell]];
        coefficients[3,14] = xFaceNormals[3, xCellFaces[2,cell]];
        coefficients[1,15] = xFaceNormals[1, xCellFaces[3,cell]];
        coefficients[2,15] = xFaceNormals[2, xCellFaces[3,cell]];
        coefficients[3,15] = xFaceNormals[3, xCellFaces[3,cell]];
        coefficients[1,16] = xFaceNormals[1, xCellFaces[4,cell]];
        coefficients[2,16] = xFaceNormals[2, xCellFaces[4,cell]];
        coefficients[3,16] = xFaceNormals[3, xCellFaces[4,cell]];
        return nothing
    end
end    


function get_coefficients(::Type{<:ON_FACES},FE::FESpace{Tv,Ti,H1BR{3},APT}, ::Type{<:Triangle2D}) where {Tv,Ti,APT}
    xFaceNormals::Array{Tv,2} = FE.xgrid[FaceNormals]
    function closure(coefficients::Array{<:Real,2}, face)
        # multiplication of face bubble with normal vector of face
        fill!(coefficients,1.0)
        coefficients[1,10] = xFaceNormals[1, face];
        coefficients[2,10] = xFaceNormals[2, face];
        coefficients[3,10] = xFaceNormals[3, face];
    end
end    

function get_coefficients(::Type{ON_CELLS},FE::FESpace{Tv,Ti,H1BR{3},APT}, ::Type{<:Hexahedron3D}) where {Tv,Ti,APT}
    xFaceNormals::Array{Tv,2} = FE.xgrid[FaceNormals]
    xCellFaces::Adjacency{Ti} = FE.xgrid[CellFaces]
    function closure(coefficients::Array{<:Real,2}, cell)
        # multiplication with normal vectors
        fill!(coefficients,1.0)
        coefficients[1,25] = xFaceNormals[1, xCellFaces[1,cell]];
        coefficients[2,25] = xFaceNormals[2, xCellFaces[1,cell]];
        coefficients[3,25] = xFaceNormals[3, xCellFaces[1,cell]];
        coefficients[1,26] = xFaceNormals[1, xCellFaces[2,cell]];
        coefficients[2,26] = xFaceNormals[2, xCellFaces[2,cell]];
        coefficients[3,26] = xFaceNormals[3, xCellFaces[2,cell]];
        coefficients[1,27] = xFaceNormals[1, xCellFaces[3,cell]];
        coefficients[2,27] = xFaceNormals[2, xCellFaces[3,cell]];
        coefficients[3,27] = xFaceNormals[3, xCellFaces[3,cell]];
        coefficients[1,28] = xFaceNormals[1, xCellFaces[4,cell]];
        coefficients[2,28] = xFaceNormals[2, xCellFaces[4,cell]];
        coefficients[3,28] = xFaceNormals[3, xCellFaces[4,cell]];
        coefficients[1,29] = xFaceNormals[1, xCellFaces[5,cell]];
        coefficients[2,29] = xFaceNormals[2, xCellFaces[5,cell]];
        coefficients[3,29] = xFaceNormals[3, xCellFaces[5,cell]];
        coefficients[1,30] = xFaceNormals[1, xCellFaces[6,cell]];
        coefficients[2,30] = xFaceNormals[2, xCellFaces[6,cell]];
        coefficients[3,30] = xFaceNormals[3, xCellFaces[6,cell]];
        return nothing
    end
end    


function get_coefficients(::Type{<:ON_FACES}, FE::FESpace{Tv,Ti,H1BR{3},APT}, ::Type{<:Quadrilateral2D}) where {Tv,Ti,APT}
    xFaceNormals::Array{Tv,2} = FE.xgrid[FaceNormals]
    function closure(coefficients::Array{<:Real,2}, face)
        # multiplication of face bubble with normal vector of face
        fill!(coefficients,1.0)
        coefficients[1,13] = xFaceNormals[1, face];
        coefficients[2,13] = xFaceNormals[2, face];
        coefficients[3,13] = xFaceNormals[3, face];
        return nothing
    end
end    


###########################
# RT0/BDM1 Reconstruction #
###########################


function get_reconstruction_coefficients!(xgrid::ExtendableGrid{Tv,Ti}, ::Union{Type{<:ON_FACES}, Type{<:ON_BFACES}}, FE::Type{<:H1BR{2}}, FER::Type{<:HDIVRT0{2}}, ::Type{<:Edge1D}) where {Tv,Ti}
    xFaceVolumes::Array{<:Real,1} = xgrid[FaceVolumes]
    xFaceNormals::Array{<:Real,2} = xgrid[FaceNormals]
    function closure(coefficients::Array{<:Real,2}, face::Int) 
        coefficients[1,1] = 1 // 2 * xFaceVolumes[face] * xFaceNormals[1, face]
        coefficients[2,1] = 1 // 2 * xFaceVolumes[face] * xFaceNormals[1, face]
        coefficients[3,1] = 1 // 2 * xFaceVolumes[face] * xFaceNormals[2, face]
        coefficients[4,1] = 1 // 2 * xFaceVolumes[face] * xFaceNormals[2, face]
        coefficients[5,1] = xFaceVolumes[face]
        return nothing
    end
end


function get_reconstruction_coefficients!(xgrid::ExtendableGrid{Tv,Ti}, ::Union{Type{<:ON_FACES}, Type{<:ON_BFACES}}, FE::Type{<:H1BR{2}}, FER::Type{<:HDIVBDM1{2}}, ::Type{<:Edge1D}) where {Tv,Ti}
    xFaceVolumes::Array{Tv,1} = xgrid[FaceVolumes]
    xFaceNormals::Array{Tv,2} = xgrid[FaceNormals]
    function closure(coefficients::Array{<:Real,2}, face) 

        coefficients[1,1] = 1 // 2 * xFaceVolumes[face] * xFaceNormals[1, face]
        coefficients[1,2] = 1 // 12 * xFaceVolumes[face] * xFaceNormals[1, face]
        
        coefficients[2,1] = 1 // 2 * xFaceVolumes[face] * xFaceNormals[1, face]
        coefficients[2,2] = -1 // 12 * xFaceVolumes[face] * xFaceNormals[1, face]

        coefficients[3,1] = 1 // 2 * xFaceVolumes[face] * xFaceNormals[2, face]
        coefficients[3,2] = 1 // 12 * xFaceVolumes[face] * xFaceNormals[2, face]
        
        coefficients[4,1] = 1 // 2 * xFaceVolumes[face] * xFaceNormals[2, face]
        coefficients[4,2] = -1 // 12 * xFaceVolumes[face] * xFaceNormals[2, face]

        coefficients[5,1] = xFaceVolumes[face]
        return nothing
    end
end


function get_reconstruction_coefficients!(xgrid::ExtendableGrid{Tv,Ti},::Union{Type{<:ON_FACES}, Type{<:ON_BFACES}}, FE::Type{<:H1BR{3}}, FER::Type{<:HDIVBDM1{3}}, EG::Type{<:Triangle2D}) where {Tv,Ti}
    xFaceVolumes::Array{Tv,1} = xgrid[FaceVolumes]
    xFaceNormals::Array{Tv,2} = xgrid[FaceNormals]
    nfacenodes::Int = num_nodes(EG)
    function closure(coefficients::Array{<:Real,2}, face::Int) 
        for j = 1 : nfacenodes, k = 1 : 3
            coefficients[(j-1)*3 + j,1] = 1 // nfacenodes * xFaceVolumes[face] * xFaceNormals[k, face]
            coefficients[(j-1)*3 + j,2] = - 1 // 36 * xFaceVolumes[face] * xFaceNormals[k, face]
            coefficients[(j-1)*3 + j,3] = - 1 // 36 * xFaceVolumes[face] * xFaceNormals[k, face]
        end
        coefficients[nfacenodes*3 + 1,1] = xFaceVolumes[face]
        return nothing
    end
end

function get_reconstruction_coefficients!(xgrid::ExtendableGrid{Tv,Ti},::Type{ON_CELLS}, FE::Type{<:H1BR{2}}, FER::Type{<:HDIVRT0{2}}, EG::Union{Type{<:Triangle2D},Type{<:Parallelogram2D}}) where {Tv,Ti}
    xFaceVolumes::Array{Tv,1} = xgrid[FaceVolumes]
    xFaceNormals::Array{Tv,2} = xgrid[FaceNormals]
    xCellFaces::Adjacency{Ti} = xgrid[CellFaces]
    face_rule::Array{Int,2} = local_cellfacenodes(EG)
    nnodes::Int = size(face_rule,1)
    nfaces::Int = size(face_rule,2)
    node::Int = 0
    face::Int = 0
    function closure(coefficients::Array{<:Real,2}, cell::Int) 
        # fill!(coefficients,0.0)
        for f = 1 : nfaces
            face = xCellFaces[f,cell]
            # reconstruction coefficients for P1 functions on reference element
            for n = 1 : nnodes
                node = face_rule[n,f]
                for k = 1 : 2
                    coefficients[nfaces*(k-1)+node,f] = 1 // 2 * xFaceVolumes[face] * xFaceNormals[k, face]
                end
            end
            # reconstruction coefficients for face bubbles on reference element
            coefficients[2*nfaces+f,f] = xFaceVolumes[face]
        end
        return nothing
    end
end

function get_reconstruction_coefficients!(xgrid::ExtendableGrid{Tv,Ti}, ::Type{ON_CELLS}, FE::Type{<:H1BR{3}}, FER::Type{<:HDIVRT0{3}}, EG::Type{<:Tetrahedron3D}) where {Tv,Ti}
    xFaceVolumes::Array{Tv,1} = xgrid[FaceVolumes]
    xFaceNormals::Array{Tv,2} = xgrid[FaceNormals]
    xCellFaces::Adjacency{Ti} = xgrid[CellFaces]
    face_rule::Array{Int,2} = local_cellfacenodes(EG)
    node::Int = 0
    face::Int = 0
    function closure(coefficients::Array{<:Real,2}, cell::Int) 
        # fill!(coefficients,0.0)
        for f = 1 : 4
            face = xCellFaces[f,cell]
            # reconstruction coefficients for P1 functions on reference element
            for n = 1 : 3
                node = face_rule[n,f]
                for k = 1 : 3
                    coefficients[4*(k-1)+node,f] = 1 // 3 * xFaceVolumes[face] * xFaceNormals[k, face]
                end
            end
            # reconstruction coefficients for face bubbles on reference element
            coefficients[12+f,f] = xFaceVolumes[face]
        end
        return nothing
    end
end

function get_reconstruction_coefficients!(xgrid::ExtendableGrid{Tv,Ti}, ::Type{ON_CELLS}, FE::Type{<:H1BR{2}}, FER::Type{<:HDIVBDM1{2}}, EG::Union{Type{<:Triangle2D}, Type{<:Parallelogram2D}}) where {Tv,Ti}
    xFaceVolumes::Array{<:Real,1} = xgrid[FaceVolumes]
    xFaceNormals::Array{<:Real,2} = xgrid[FaceNormals]
    xCellFaceSigns::Adjacency{Ti} = xgrid[CellFaceSigns]
    xCellFaces::Adjacency{Ti} = xgrid[CellFaces]
    face_rule::Array{Int,2} = local_cellfacenodes(EG)
    nfaces::Int = size(face_rule,2)
    node::Int = 0
    face::Int = 0
    BDM1_coeffs::Array{<:Real,1} = [-1//12, 1//12]
    function closure(coefficients::Array{<:Real,2}, cell::Int) 
        # fill!(coefficients,0.0)
        for f = 1 : nfaces
            face = xCellFaces[f,cell]
            for n = 1 : 2
                node = face_rule[n,f]
                for k = 1 : 2
                    # RT0 reconstruction coefficients for P1 functions on reference element
                    coefficients[nfaces*(k-1)+node,2*(f-1)+1] = 1 // 2 * xFaceVolumes[face] * xFaceNormals[k, face]

                    # BDM1 reconstruction coefficients for P1 functions on reference element
                    coefficients[nfaces*(k-1)+node,2*(f-1)+2] = BDM1_coeffs[n] * xFaceVolumes[face] * xFaceNormals[k, face] * xCellFaceSigns[f, cell]
                end
            end
            # RT0 reconstruction coefficients for face bubbles on reference element
            coefficients[nfaces*2+f,2*(f-1)+1] = xFaceVolumes[face]
        end
        return nothing
    end
end


function get_reconstruction_coefficients!(xgrid::ExtendableGrid{Tv,Ti}, ::Type{ON_CELLS}, FE::Type{<:H1BR{3}}, FER::Type{<:HDIVBDM1{3}}, EG::Type{<:Tetrahedron3D}) where {Tv,Ti}
    xFaceVolumes::Array{Tv,1} = xgrid[FaceVolumes]
    xFaceNormals::Array{Tv,2} = xgrid[FaceNormals]
    xCellFaceOrientations::Adjacency{Ti} = xgrid[CellFaceOrientations]
    xCellFaces::Adjacency{Ti} = xgrid[CellFaces]
    face_rule::Array{Int,2} = local_cellfacenodes(EG)
    node::Int = face_rule[1,1]
    face::Ti = 0
    BDM1_coeffs::Array{Tv,2} = [-1//36 -1//36 1//18;
                   -1//36 1//18 -1//36;
                    1//18 -1//36 -1//36]
    orientation::Int = 0
    index1::Int = 0
    index2::Int = 0
    row4orientation1::Array{Int,1} = [2,2,3,1]
    row4orientation2::Array{Int,1} = [1,3,1,2]
    function closure(coefficients::Array{<:Real,2}, cell::Int) 
        # fill!(coefficients,0.0)
        index2 = 0
        for f = 1 : 4
            face = xCellFaces[f,cell]
            index1 = 0
            for k = 1 : 3
                for n = 1 : 3
                    node = face_rule[n,f]
                    # RT0 reconstruction coefficients for P1 functions on reference element
                    coefficients[index1+node,index2+1] = 1 // 3 * xFaceNormals[k, face] * xFaceVolumes[face] 
                    orientation = xCellFaceOrientations[f,cell]
                    # BDM1 reconstruction coefficients for P1 functions on reference element
                    coefficients[index1+node,index2+2] = BDM1_coeffs[n, row4orientation1[orientation]] * xFaceNormals[k, face] * xFaceVolumes[face] 
                    coefficients[index1+node,index2+3] = BDM1_coeffs[n, row4orientation2[orientation]] * xFaceNormals[k, face] * xFaceVolumes[face] 
                end
                index1 += 4
            end
            # RT0 reconstruction coefficients for face bubbles on reference element
            coefficients[index1+f,index2+1] = xFaceVolumes[face]
            index2 += 3
        end
        return nothing
    end
end
