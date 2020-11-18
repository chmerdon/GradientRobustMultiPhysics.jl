
"""
$(TYPEDEF)

vector-valued (ncomponents = edim) Bernardi--Raugel element
(first-order polynomials + normal-weighted face bubbles)

allowed ElementGeometries:
- Triangle2D (piecewise linear + normal-weighted face bubbles)
- Quadrilateral2D (Q1 space + normal-weighted face bubbles)
- Tetrahedron3D (piecewise linear + normal-weighted face bubbles)
"""
abstract type H1BR{edim} <: AbstractH1FiniteElementWithCoefficients where {edim<:Int} end

get_ncomponents(FEType::Type{<:H1BR}) = FEType.parameters[1]
get_ndofs_on_face(FEType::Type{<:H1BR}, EG::Type{<:AbstractElementGeometry}) = 1 + nnodes_for_geometry(EG) * FEType.parameters[1]
get_ndofs_on_cell(FEType::Type{<:H1BR}, EG::Type{<:AbstractElementGeometry}) = nfaces_for_geometry(EG) + nnodes_for_geometry(EG) * FEType.parameters[1]


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


function init!(FES::FESpace{FEType}) where {FEType <: H1BR}
    ncomponents = get_ncomponents(FEType)
    FES.name = "BR (H1, $(ncomponents)d)"

    # count number of dofs
    nnodes = num_sources(FES.xgrid[Coordinates])
    nfaces = num_sources(FES.xgrid[FaceNodes])
    FES.ndofs = nnodes * ncomponents + nfaces
end


function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{AT_NODES}, exact_function!::Function; items = [], bonus_quadorder::Int = 0) where {FEType <: H1BR}
    nnodes = size(FE.xgrid[Coordinates],2)
    point_evaluation!(Target, FE, AT_NODES, exact_function!; items = items, component_offset = nnodes)
end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{ON_EDGES}, exact_function!::Function; items = [], bonus_quadorder::Int = 0) where {FEType <: H1BR}
    # delegate edge nodes to node interpolation
    subitems = slice(FE.xgrid[EdgeNodes], items)
    interpolate!(Target, FE, AT_NODES, exact_function!; items = subitems, bonus_quadorder = bonus_quadorder)
end

function interpolate!(Target::AbstractArray{T,1}, FE::FESpace{FEType}, ::Type{ON_FACES}, exact_function!::Function; items = [], bonus_quadorder::Int = 0) where {T<:Real, FEType <: H1BR}
    # delegate face nodes to node interpolation
    subitems = slice(FE.xgrid[FaceNodes], items)
    interpolate!(Target, FE, AT_NODES, exact_function!; items = subitems, bonus_quadorder = bonus_quadorder)

    # preserve face means in normal direction
    xItemVolumes = FE.xgrid[FaceVolumes]
    xItemNodes = FE.xgrid[FaceNodes]
    xItemGeometries = FE.xgrid[FaceGeometries]
    xFaceNormals = FE.xgrid[FaceNormals]
    xItemDofs = FE.dofmaps[FaceDofs]
    nnodes = size(FE.xgrid[Coordinates],2)
    nitems = num_sources(xItemNodes)
    ncomponents = get_ncomponents(FEType)
    offset = ncomponents*nnodes
    if items == []
        items = 1 : nitems
    end

    # compute exact face means
    facemeans = zeros(Float64,ncomponents,nitems)
    integrate!(facemeans, FE.xgrid, ON_FACES, exact_function!, bonus_quadorder, ncomponents; items = items)
    P1flux::T = 0
    value::T = 0
    itemEG = Edge1D
    nitemnodes::Int = 0
    for item in items
        itemEG = xItemGeometries[item]
        nitemnodes = nnodes_for_geometry(itemEG)
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

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{ON_CELLS}, exact_function!::Function; items = [], bonus_quadorder::Int = 0) where {FEType <: H1BR}
    # delegate cell faces to face interpolation
    subitems = slice(FE.xgrid[CellFaces], items)
    interpolate!(Target, FE, ON_FACES, exact_function!; items = subitems, bonus_quadorder = bonus_quadorder)
end


function nodevalues!(Target::AbstractArray{<:Real,2}, Source::AbstractArray{<:Real,1}, FE::FESpace{<:H1BR})
    nnodes = num_sources(FE.xgrid[Coordinates])
    nfaces = num_sources(FE.xgrid[FaceNodes])
    FEType = eltype(FE)
    ncomponents = get_ncomponents(FEType)
    offset4component = 0:nnodes:ncomponents*nnodes
    for node = 1 : nnodes
        for c = 1 : ncomponents
            Target[c,node] = Source[offset4component[c]+node]
        end    
    end    
end


############
# 2D basis #
############

function get_basis_on_face(FEType::Type{H1BR{2}}, EG::Type{<:Edge1D})
    ncomponents = get_ncomponents(FEType)
    refbasis_P1 = get_basis_on_face(H1P1{ncomponents}, EG)
    offset = get_ndofs_on_face(H1P1{ncomponents}, EG)
    function closure(refbasis, xref)
        refbasis_P1(refbasis, xref)
        # add face bubble to P1 basis
        refbasis[offset+1,:] .= 6 * xref[1] * refbasis[1,1]
    end
end

function get_basis_on_cell(FEType::Type{H1BR{2}}, EG::Type{<:Triangle2D})
    ncomponents = get_ncomponents(FEType)
    refbasis_P1 = get_basis_on_cell(H1P1{ncomponents}, EG)
    offset = get_ndofs_on_cell(H1P1{ncomponents}, EG)
    function closure(refbasis, xref)
        refbasis_P1(refbasis, xref)
        # add face bubbles to P1 basis
        refbasis[offset+1,:] .= 6 * xref[1] * refbasis[1,1]
        refbasis[offset+2,:] .= 6 * xref[2] * xref[1]
        refbasis[offset+3,:] .= 6 * refbasis[1,1] * xref[2]
    end
end


function get_basis_on_cell(FEType::Type{H1BR{2}}, EG::Type{<:Quadrilateral2D})
    ncomponents = get_ncomponents(FEType)
    refbasis_P1 = get_basis_on_cell(H1P1{ncomponents}, EG)
    offset = get_ndofs_on_cell(H1P1{ncomponents}, EG)
    a = 0.0
    b = 0.0
    function closure(refbasis, xref)
        refbasis_P1(refbasis, xref)
        # add face bubbles to Q1 basis
        a = 1 - xref[1]
        b = 1 - xref[2]
        refbasis[offset+1,:] .= 6*xref[1]*a*b
        refbasis[offset+2,:] .= 6*xref[2]*xref[1]*b
        refbasis[offset+3,:] .= 6*xref[1]*xref[2]*a
        refbasis[offset+4,:] .= 6*xref[2]*a*b
    end
end

function get_coefficients_on_cell!(FE::FESpace{H1BR{2}}, ::Type{<:Triangle2D})
    xFaceNormals::Array{Float64,2} = FE.xgrid[FaceNormals]
    xCellFaces = FE.xgrid[CellFaces]
    function closure(coefficients, cell)
        fill!(coefficients,1.0)
        coefficients[1,7] = xFaceNormals[1, xCellFaces[1,cell]];
        coefficients[2,7] = xFaceNormals[2, xCellFaces[1,cell]];
        coefficients[1,8] = xFaceNormals[1, xCellFaces[2,cell]];
        coefficients[2,8] = xFaceNormals[2, xCellFaces[2,cell]];
        coefficients[1,9] = xFaceNormals[1, xCellFaces[3,cell]];
        coefficients[2,9] = xFaceNormals[2, xCellFaces[3,cell]];
    end
end

function get_coefficients_on_cell!(FE::FESpace{H1BR{2}}, ::Type{<:Quadrilateral2D})
    xFaceNormals::Array{Float64,2} = FE.xgrid[FaceNormals]
    xCellFaces = FE.xgrid[CellFaces]
    function closure(coefficients, cell)
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

function get_coefficients_on_face!(FE::FESpace{H1BR{2}}, ::Type{<:Edge1D})
    xFaceNormals::Array{Float64,2} = FE.xgrid[FaceNormals]
    function closure(coefficients, face)
        # multiplication of face bubble with normal vector of face
        fill!(coefficients,1.0)
        coefficients[1,5] = xFaceNormals[1, face];
        coefficients[2,5] = xFaceNormals[2, face];
    end
end    


############
# 3D basis #
############

function get_basis_on_face(FEType::Type{H1BR{3}}, EG::Type{<:Triangle2D})
    ncomponents = get_ncomponents(FEType)
    refbasis_P1 = get_basis_on_face(H1P1{ncomponents}, EG)
    offset = get_ndofs_on_face(H1P1{ncomponents}, EG)
    function closure(refbasis, xref)
        refbasis_P1(refbasis, xref)
        # add face bubbles to P1 basis
        refbasis[offset+1,:] .= 60 * xref[1] * refbasis[1,1] * xref[2]
    end
end

function get_basis_on_face(FEType::Type{H1BR{3}}, EG::Type{<:Quadrilateral2D})
    ncomponents = get_ncomponents(FEType)
    refbasis_P1 = get_basis_on_face(H1P1{ncomponents}, EG)
    offset = get_ndofs_on_face(H1P1{ncomponents}, EG)
    function closure(refbasis, xref)
        refbasis_P1(refbasis, xref)
        # add face bubbles to P1 basis
        refbasis[offset+1,:] .= 36 * xref[1] * (1 - xref[1]) * (1 - xref[2]) * xref[2]
    end
end

function get_basis_on_cell(FEType::Type{H1BR{3}}, EG::Type{<:Tetrahedron3D})
    ncomponents = get_ncomponents(FEType)
    refbasis_P1 = get_basis_on_cell(H1P1{ncomponents}, EG)
    offset = get_ndofs_on_cell(H1P1{ncomponents}, EG)
    function closure(refbasis, xref)
        refbasis_P1(refbasis, xref)
        # add face bubbles to P1 basis
        refbasis[offset+1,:] .= 60 * xref[1] * refbasis[1,1] * xref[2]
        refbasis[offset+2,:] .= 60 * refbasis[1,1] * xref[1] * xref[3]
        refbasis[offset+3,:] .= 60 * xref[1] * xref[2] * xref[3]
        refbasis[offset+4,:] .= 60 * refbasis[1,1] * xref[2] * xref[3]
    end
end

function get_basis_on_cell(FEType::Type{H1BR{3}}, EG::Type{<:Hexahedron3D})
    ncomponents = get_ncomponents(FEType)
    refbasis_P1 = get_basis_on_cell(H1P1{ncomponents}, EG)
    offset = get_ndofs_on_cell(H1P1{ncomponents}, EG)
    a = 0.0
    b = 0.0
    c = 0.0
    function closure(refbasis, xref)
        refbasis_P1(refbasis, xref)
        # add face bubbles to Q1 basis
        a = 1 - xref[1]
        b = 1 - xref[2]
        c = 1 - xref[3]
        refbasis[offset+1,:] .= 36*a*b*xref[1]*xref[2]*c # bottom
        refbasis[offset+2,:] .= 36*a*xref[1]*c*xref[3]*b # front
        refbasis[offset+3,:] .= 36*a*b*c*xref[2]*xref[3] # left
        refbasis[offset+4,:] .= 36*a*xref[1]*c*xref[3]*xref[2] # back
        refbasis[offset+5,:] .= 36*xref[1]*b*c*xref[2]*xref[3] # right
        refbasis[offset+6,:] .= 36*a*b*xref[1]*xref[2]*xref[3] # top
    end
end


function get_coefficients_on_cell!(FE::FESpace{H1BR{3}}, ::Type{<:Tetrahedron3D})
    xFaceNormals::Array{Float64,2} = FE.xgrid[FaceNormals]
    xCellFaces = FE.xgrid[CellFaces]
    function closure(coefficients, cell)
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


function get_coefficients_on_face!(FE::FESpace{H1BR{3}}, ::Type{<:Triangle2D})
    xFaceNormals::Array{Float64,2} = FE.xgrid[FaceNormals]
    function closure(coefficients, face)
        # multiplication of face bubble with normal vector of face
        fill!(coefficients,1.0)
        coefficients[1,10] = xFaceNormals[1, face];
        coefficients[2,10] = xFaceNormals[2, face];
        coefficients[3,10] = xFaceNormals[3, face];
    end
end    


function get_coefficients_on_cell!(FE::FESpace{H1BR{3}}, ::Type{<:Hexahedron3D})
    xFaceNormals::Array{Float64,2} = FE.xgrid[FaceNormals]
    xCellFaces = FE.xgrid[CellFaces]
    function closure(coefficients, cell)
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


function get_coefficients_on_face!(FE::FESpace{H1BR{3}}, ::Type{<:Quadrilateral2D})
    xFaceNormals::Array{Float64,2} = FE.xgrid[FaceNormals]
    function closure(coefficients, face)
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


function get_reconstruction_coefficients_on_face!(FE::FESpace{H1BR{2}}, FER::FESpace{HDIVRT0{2}}, ::Type{<:Edge1D})
    xFaceVolumes::Array{Float64,1} = FE.xgrid[FaceVolumes]
    xFaceNormals::Array{Float64,2} = FE.xgrid[FaceNormals]
    xCellFaces = FE.xgrid[CellFaces]
    function closure(coefficients, face::Int) 

        coefficients[1,1] = 1 // 2 * xFaceVolumes[face] * xFaceNormals[1, face]
        coefficients[2,1] = 1 // 2 * xFaceVolumes[face] * xFaceNormals[1, face]
        coefficients[3,1] = 1 // 2 * xFaceVolumes[face] * xFaceNormals[2, face]
        coefficients[4,1] = 1 // 2 * xFaceVolumes[face] * xFaceNormals[2, face]
        coefficients[5,1] = xFaceVolumes[face]
        return nothing
    end
end


function get_reconstruction_coefficients_on_face!(FE::FESpace{H1BR{2}}, FER::FESpace{HDIVBDM1{2}}, ::Type{<:Edge1D})
    xFaceVolumes::Array{Float64,1} = FE.xgrid[FaceVolumes]
    xFaceNormals::Array{Float64,2} = FE.xgrid[FaceNormals]
    xCellFaces = FE.xgrid[CellFaces]
    function closure(coefficients, face::Int) 

        coefficients[1,1] = 1 // 2 * xFaceVolumes[face] * xFaceNormals[1, face]
        coefficients[1,2] = 1 // 12 * xFaceVolumes[face] * xFaceNormals[1, face]
        
        coefficients[2,1] = 1 // 2 * xFaceVolumes[face] * xFaceNormals[1, face]
        coefficients[2,2] = -1 // 12 * xFaceVolumes[face] * xFaceNormals[1, face]

        coefficients[3,1] = 1 // 2 * xFaceVolumes[face] * xFaceNormals[2, face]
        coefficients[3,2] = 1 // 12 * xFaceVolumes[face] * xFaceNormals[2, face]
        
        coefficients[4,1] = 1 // 2 * xFaceVolumes[face] * xFaceNormals[2, face]
        coefficients[4,2] = -1 // 12 * xFaceVolumes[face] * xFaceNormals[2, face]

        coefficients[3,1] = xFaceVolumes[face]
        return nothing
    end
end

function get_reconstruction_coefficients_on_cell!(FE::FESpace{H1BR{2}}, FER::FESpace{HDIVRT0{2}}, ::Type{<:Triangle2D})
    xFaceVolumes::Array{Float64,1} = FE.xgrid[FaceVolumes]
    xFaceNormals::Array{Float64,2} = FE.xgrid[FaceNormals]
    xCellFaces = FE.xgrid[CellFaces]
    faces::Array{Int32,1} = [1,2,3]
    function closure(coefficients, cell::Int) 
        
        # fill!(coefficients,0.0) # not needed if coefficients is initialized with zeros

        # get faces of cell
        faces[1] = xCellFaces[1,cell]
        faces[2] = xCellFaces[2,cell]
        faces[3] = xCellFaces[3,cell]

        # P1 first component RT0
        coefficients[1,3] = 1 // 2 * xFaceVolumes[faces[3]] * xFaceNormals[1, faces[3]]
        coefficients[1,1] = 1 // 2 * xFaceVolumes[faces[1]] * xFaceNormals[1, faces[1]]
        coefficients[2,1] = 1 // 2 * xFaceVolumes[faces[1]] * xFaceNormals[1, faces[1]]
        coefficients[2,2] = 1 // 2 * xFaceVolumes[faces[2]] * xFaceNormals[1, faces[2]]
        coefficients[3,2] = 1 // 2 * xFaceVolumes[faces[2]] * xFaceNormals[1, faces[2]]
        coefficients[3,3] = 1 // 2 * xFaceVolumes[faces[3]] * xFaceNormals[1, faces[3]]
    
        # P2 second component RT0
        coefficients[4,3] = 1 // 2 * xFaceVolumes[faces[3]] * xFaceNormals[2, faces[3]]
        coefficients[4,1] = 1 // 2 * xFaceVolumes[faces[1]] * xFaceNormals[2, faces[1]]
        coefficients[5,1] = 1 // 2 * xFaceVolumes[faces[1]] * xFaceNormals[2, faces[1]]
        coefficients[5,2] = 1 // 2 * xFaceVolumes[faces[2]] * xFaceNormals[2, faces[2]]
        coefficients[6,2] = 1 // 2 * xFaceVolumes[faces[2]] * xFaceNormals[2, faces[2]]
        coefficients[6,3] = 1 // 2 * xFaceVolumes[faces[3]] * xFaceNormals[2, faces[3]]
    
        # reconstruction coefficients for face bubbles on reference element (same as RT0, BDM1 coefficients are zero)
        coefficients[7,1] = xFaceVolumes[faces[1]]
        coefficients[8,2] = xFaceVolumes[faces[2]]
        coefficients[9,3] = xFaceVolumes[faces[3]]
        return nothing
    end
end

function get_reconstruction_coefficients_on_cell!(FE::FESpace{H1BR{2}}, FER::FESpace{HDIVBDM1{2}}, ::Type{<:Triangle2D})
    xFaceVolumes::Array{Float64,1} = FE.xgrid[FaceVolumes]
    xFaceNormals::Array{Float64,2} = FE.xgrid[FaceNormals]
    xCellFaceSigns = FER.xgrid[CellFaceSigns]
    xCellFaces = FE.xgrid[CellFaces]
    faces::Array{Int32,1} = [1,2,3]
    function closure(coefficients, cell::Int) 
        
        # fill!(coefficients,0.0) # not needed if coefficients is initialized with zeros

        # get faces of cell
        faces[1] = xCellFaces[1,cell]
        faces[2] = xCellFaces[2,cell]
        faces[3] = xCellFaces[3,cell]

        # P1 first component RT0
        coefficients[1,5] = 1 // 2 * xFaceVolumes[faces[3]] * xFaceNormals[1, faces[3]]
        coefficients[1,1] = 1 // 2 * xFaceVolumes[faces[1]] * xFaceNormals[1, faces[1]]
        coefficients[2,1] = 1 // 2 * xFaceVolumes[faces[1]] * xFaceNormals[1, faces[1]]
        coefficients[2,3] = 1 // 2 * xFaceVolumes[faces[2]] * xFaceNormals[1, faces[2]]
        coefficients[3,3] = 1 // 2 * xFaceVolumes[faces[2]] * xFaceNormals[1, faces[2]]
        coefficients[3,5] = 1 // 2 * xFaceVolumes[faces[3]] * xFaceNormals[1, faces[3]]
    
        # P1 first component BDM1/RT0
        coefficients[1,6] =  1 // 12 * xFaceVolumes[faces[3]] * xFaceNormals[1, faces[3]] * xCellFaceSigns[3,cell];
        coefficients[1,2] = -1 // 12 * xFaceVolumes[faces[1]] * xFaceNormals[1, faces[1]] * xCellFaceSigns[1,cell]
        coefficients[2,2] =  1 // 12 * xFaceVolumes[faces[1]] * xFaceNormals[1, faces[1]] * xCellFaceSigns[1,cell]
        coefficients[2,4] = -1 // 12 * xFaceVolumes[faces[2]] * xFaceNormals[1, faces[2]] * xCellFaceSigns[2,cell]
        coefficients[3,4] =  1 // 12 * xFaceVolumes[faces[2]] * xFaceNormals[1, faces[2]] * xCellFaceSigns[2,cell]
        coefficients[3,6] = -1 // 12 * xFaceVolumes[faces[3]] * xFaceNormals[1, faces[3]] * xCellFaceSigns[3,cell]
    
        # P2 second component RT0
        coefficients[4,5] = 1 // 2 * xFaceVolumes[faces[3]] * xFaceNormals[2, faces[3]]
        coefficients[4,1] = 1 // 2 * xFaceVolumes[faces[1]] * xFaceNormals[2, faces[1]]
        coefficients[5,1] = 1 // 2 * xFaceVolumes[faces[1]] * xFaceNormals[2, faces[1]]
        coefficients[5,3] = 1 // 2 * xFaceVolumes[faces[2]] * xFaceNormals[2, faces[2]]
        coefficients[6,3] = 1 // 2 * xFaceVolumes[faces[2]] * xFaceNormals[2, faces[2]]
        coefficients[6,5] = 1 // 2 * xFaceVolumes[faces[3]] * xFaceNormals[2, faces[3]]
    
        # P1 second component BDM1/RT0
        coefficients[4,6] =  1 // 12 * xFaceVolumes[faces[3]] * xFaceNormals[2, faces[3]] * xCellFaceSigns[3,cell]
        coefficients[4,2] = -1 // 12 * xFaceVolumes[faces[1]] * xFaceNormals[2, faces[1]] * xCellFaceSigns[1,cell]
        coefficients[5,2] =  1 // 12 * xFaceVolumes[faces[1]] * xFaceNormals[2, faces[1]] * xCellFaceSigns[1,cell]
        coefficients[5,4] = -1 // 12 * xFaceVolumes[faces[2]] * xFaceNormals[2, faces[2]] * xCellFaceSigns[2,cell]
        coefficients[6,4] =  1 // 12 * xFaceVolumes[faces[2]] * xFaceNormals[2, faces[2]] * xCellFaceSigns[2,cell]
        coefficients[6,6] = -1 // 12 * xFaceVolumes[faces[3]] * xFaceNormals[2, faces[3]] * xCellFaceSigns[3,cell]
    
        # reconstruction coefficients for face bubbles on reference element (same as RT0, BDM1 coefficients are zero)
        coefficients[7,1] = xFaceVolumes[faces[1]]
        coefficients[8,3] = xFaceVolumes[faces[2]]
        coefficients[9,5] = xFaceVolumes[faces[3]]
        return nothing
    end
end



function get_reconstruction_coefficients_on_cell!(FE::FESpace{H1BR{2}}, FER::FESpace{HDIVRT0{2}}, ::Type{<:Parallelogram2D})
    xFaceVolumes::Array{Float64,1} = FE.xgrid[FaceVolumes]
    xFaceNormals::Array{Float64,2} = FE.xgrid[FaceNormals]
    xCellFaces = FE.xgrid[CellFaces]
    faces::Array{Int32,1} = [1,2,3,4]
    function closure(coefficients, cell::Int) 
        
        # fill!(coefficients,0.0) # not needed if coefficients is initialized with zeros

        # get faces of cell
        faces[1] = xCellFaces[1,cell]
        faces[2] = xCellFaces[2,cell]
        faces[3] = xCellFaces[3,cell]
        faces[4] = xCellFaces[4,cell]

        # reconstruction coefficients for P1 basis functions on reference element
        fill!(coefficients,0.0)
        coefficients[1,4] = 1 // 2 * xFaceVolumes[faces[4]] * xFaceNormals[1, faces[4]]
        coefficients[1,1] = 1 // 2 * xFaceVolumes[faces[1]] * xFaceNormals[1, faces[1]]
        coefficients[2,1] = 1 // 2 * xFaceVolumes[faces[1]] * xFaceNormals[1, faces[1]]
        coefficients[2,2] = 1 // 2 * xFaceVolumes[faces[2]] * xFaceNormals[1, faces[2]]
        coefficients[3,2] = 1 // 2 * xFaceVolumes[faces[2]] * xFaceNormals[1, faces[2]]
        coefficients[3,3] = 1 // 2 * xFaceVolumes[faces[3]] * xFaceNormals[1, faces[3]]
        coefficients[4,3] = 1 // 2 * xFaceVolumes[faces[3]] * xFaceNormals[1, faces[3]]
        coefficients[4,4] = 1 // 2 * xFaceVolumes[faces[4]] * xFaceNormals[1, faces[4]]
        coefficients[5,4] = 1 // 2 * xFaceVolumes[faces[4]] * xFaceNormals[2, faces[4]]
        coefficients[5,1] = 1 // 2 * xFaceVolumes[faces[1]] * xFaceNormals[2, faces[1]]
        coefficients[6,1] = 1 // 2 * xFaceVolumes[faces[1]] * xFaceNormals[2, faces[1]]
        coefficients[6,2] = 1 // 2 * xFaceVolumes[faces[2]] * xFaceNormals[2, faces[2]]
        coefficients[7,2] = 1 // 2 * xFaceVolumes[faces[2]] * xFaceNormals[2, faces[2]]
        coefficients[7,3] = 1 // 2 * xFaceVolumes[faces[3]] * xFaceNormals[2, faces[3]]
        coefficients[8,3] = 1 // 2 * xFaceVolumes[faces[3]] * xFaceNormals[2, faces[3]]
        coefficients[8,4] = 1 // 2 * xFaceVolumes[faces[4]] * xFaceNormals[2, faces[4]]

        coefficients[ 9,1] = xFaceVolumes[faces[1]]
        coefficients[10,2] = xFaceVolumes[faces[2]]
        coefficients[11,3] = xFaceVolumes[faces[3]]
        coefficients[12,4] = xFaceVolumes[faces[4]]
        return nothing
    end
end



function get_reconstruction_coefficients_on_cell!(FE::FESpace{H1BR{2}}, FER::FESpace{HDIVBDM1{2}}, ::Type{<:Parallelogram2D})
    xFaceVolumes::Array{Float64,1} = FE.xgrid[FaceVolumes]
    xFaceNormals::Array{Float64,2} = FE.xgrid[FaceNormals]
    xCellFaceSigns = FER.xgrid[CellFaceSigns]
    xCellFaces = FE.xgrid[CellFaces]
    faces::Array{Int32,1} = [1,2,3,4]
    function closure(coefficients, cell::Int) 
        
        # fill!(coefficients,0.0) # not needed if coefficients is initialized with zeros

        # get faces of cell
        faces[1] = xCellFaces[1,cell]
        faces[2] = xCellFaces[2,cell]
        faces[3] = xCellFaces[3,cell]
        faces[4] = xCellFaces[4,cell]

        # reconstruction coefficients for P1 basis functions on reference element
        coefficients[1,7] = 1 // 2 * xFaceVolumes[faces[4]] * xFaceNormals[1, faces[4]]
        coefficients[1,1] = 1 // 2 * xFaceVolumes[faces[1]] * xFaceNormals[1, faces[1]]
        coefficients[2,1] = 1 // 2 * xFaceVolumes[faces[1]] * xFaceNormals[1, faces[1]]
        coefficients[2,3] = 1 // 2 * xFaceVolumes[faces[2]] * xFaceNormals[1, faces[2]]
        coefficients[3,3] = 1 // 2 * xFaceVolumes[faces[2]] * xFaceNormals[1, faces[2]]
        coefficients[3,5] = 1 // 2 * xFaceVolumes[faces[3]] * xFaceNormals[1, faces[3]]
        coefficients[4,5] = 1 // 2 * xFaceVolumes[faces[3]] * xFaceNormals[1, faces[3]]
        coefficients[4,7] = 1 // 2 * xFaceVolumes[faces[4]] * xFaceNormals[1, faces[4]]
        coefficients[5,7] = 1 // 2 * xFaceVolumes[faces[4]] * xFaceNormals[2, faces[4]]
        coefficients[5,1] = 1 // 2 * xFaceVolumes[faces[1]] * xFaceNormals[2, faces[1]]
        coefficients[6,1] = 1 // 2 * xFaceVolumes[faces[1]] * xFaceNormals[2, faces[1]]
        coefficients[6,3] = 1 // 2 * xFaceVolumes[faces[2]] * xFaceNormals[2, faces[2]]
        coefficients[7,3] = 1 // 2 * xFaceVolumes[faces[2]] * xFaceNormals[2, faces[2]]
        coefficients[7,5] = 1 // 2 * xFaceVolumes[faces[3]] * xFaceNormals[2, faces[3]]
        coefficients[8,5] = 1 // 2 * xFaceVolumes[faces[3]] * xFaceNormals[2, faces[3]]
        coefficients[8,7] = 1 // 2 * xFaceVolumes[faces[4]] * xFaceNormals[2, faces[4]]

        # face bubbles (only RT0)
        coefficients[ 9,1] = xFaceVolumes[faces[1]]
        coefficients[10,3] = xFaceVolumes[faces[2]]
        coefficients[11,5] = xFaceVolumes[faces[3]]
        coefficients[12,7] = xFaceVolumes[faces[4]]

        # higher-order BDM1
        coefficients[1,8] =  1 // 12 * xFaceVolumes[faces[4]] * xFaceNormals[1, faces[4]] * xCellFaceSigns[4,cell]
        coefficients[1,2] = -1 // 12 * xFaceVolumes[faces[1]] * xFaceNormals[1, faces[1]] * xCellFaceSigns[1,cell]
        coefficients[2,2] =  1 // 12 * xFaceVolumes[faces[1]] * xFaceNormals[1, faces[1]] * xCellFaceSigns[1,cell]
        coefficients[2,4] = -1 // 12 * xFaceVolumes[faces[2]] * xFaceNormals[1, faces[2]] * xCellFaceSigns[2,cell]
        coefficients[3,4] =  1 // 12 * xFaceVolumes[faces[2]] * xFaceNormals[1, faces[2]] * xCellFaceSigns[2,cell]
        coefficients[3,6] = -1 // 12 * xFaceVolumes[faces[3]] * xFaceNormals[1, faces[3]] * xCellFaceSigns[3,cell]
        coefficients[4,6] =  1 // 12 * xFaceVolumes[faces[3]] * xFaceNormals[1, faces[3]] * xCellFaceSigns[3,cell]
        coefficients[4,8] = -1 // 12 * xFaceVolumes[faces[4]] * xFaceNormals[1, faces[4]] * xCellFaceSigns[4,cell]
        coefficients[5,8] =  1 // 12 * xFaceVolumes[faces[4]] * xFaceNormals[2, faces[4]] * xCellFaceSigns[4,cell]
        coefficients[5,2] = -1 // 12 * xFaceVolumes[faces[1]] * xFaceNormals[2, faces[1]] * xCellFaceSigns[1,cell]
        coefficients[6,2] =  1 // 12 * xFaceVolumes[faces[1]] * xFaceNormals[2, faces[1]] * xCellFaceSigns[1,cell]
        coefficients[6,4] = -1 // 12 * xFaceVolumes[faces[2]] * xFaceNormals[2, faces[2]] * xCellFaceSigns[2,cell]
        coefficients[7,4] =  1 // 12 * xFaceVolumes[faces[2]] * xFaceNormals[2, faces[2]] * xCellFaceSigns[2,cell]
        coefficients[7,6] = -1 // 12 * xFaceVolumes[faces[3]] * xFaceNormals[2, faces[3]] * xCellFaceSigns[3,cell]
        coefficients[8,6] =  1 // 12 * xFaceVolumes[faces[3]] * xFaceNormals[2, faces[3]] * xCellFaceSigns[3,cell]
        coefficients[8,8] = -1 // 12 * xFaceVolumes[faces[4]] * xFaceNormals[2, faces[4]] * xCellFaceSigns[4,cell]

        return nothing
    end
end


function get_reconstruction_coefficients_on_cell!(FE::FESpace{H1BR{3}}, FER::FESpace{HDIVRT0{3}}, ::Type{<:Tetrahedron3D})
    xFaceVolumes::Array{Float64,1} = FE.xgrid[FaceVolumes]
    xFaceNormals::Array{Float64,2} = FE.xgrid[FaceNormals]
    xCellFaces = FE.xgrid[CellFaces]
    faces::Array{Int32,1} = [1,2,3,4]
    function closure(coefficients, cell::Int) 
        # reconstruction coefficients for P1 basis functions on reference element
        # fill!(coefficients,0.0)
        
        faces[1] = xCellFaces[1,cell]
        faces[2] = xCellFaces[2,cell]
        faces[3] = xCellFaces[3,cell]
        faces[4] = xCellFaces[4,cell]

        for k = 1 : 3
            # node 1
            coefficients[4*k-3,4] = 1 // 3 * xFaceVolumes[faces[4]] * xFaceNormals[k, faces[4]]
            coefficients[4*k-3,1] = 1 // 3 * xFaceVolumes[faces[1]] * xFaceNormals[k, faces[1]]
            coefficients[4*k-3,2] = 1 // 3 * xFaceVolumes[faces[2]] * xFaceNormals[k, faces[2]]

            # node 2
            coefficients[4*k-2,1] = 1 // 3 * xFaceVolumes[faces[1]] * xFaceNormals[k, faces[1]]
            coefficients[4*k-2,2] = 1 // 3 * xFaceVolumes[faces[2]] * xFaceNormals[k, faces[2]]
            coefficients[4*k-2,3] = 1 // 3 * xFaceVolumes[faces[3]] * xFaceNormals[k, faces[3]]

            # node 3
            coefficients[4*k-1,1] = 1 // 3 * xFaceVolumes[faces[1]] * xFaceNormals[k, faces[1]]
            coefficients[4*k-1,3] = 1 // 3 * xFaceVolumes[faces[3]] * xFaceNormals[k, faces[3]]
            coefficients[4*k-1,4] = 1 // 3 * xFaceVolumes[faces[4]] * xFaceNormals[k, faces[4]]

            # node 4
            coefficients[4*k,2] = 1 // 3 * xFaceVolumes[faces[2]] * xFaceNormals[k, faces[2]]
            coefficients[4*k,3] = 1 // 3 * xFaceVolumes[faces[3]] * xFaceNormals[k, faces[3]]
            coefficients[4*k,4] = 1 // 3 * xFaceVolumes[faces[4]] * xFaceNormals[k, faces[4]]
        end

        # reconstruction coefficients for face bubbles on reference element
        coefficients[13,1] = xFaceVolumes[faces[1]]
        coefficients[14,2] = xFaceVolumes[faces[2]]
        coefficients[15,3] = xFaceVolumes[faces[3]]
        coefficients[16,4] = xFaceVolumes[faces[4]]
        return nothing
    end
end
