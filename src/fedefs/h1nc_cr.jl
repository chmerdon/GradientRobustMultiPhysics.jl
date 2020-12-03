
"""
$(TYPEDEF)

Crouzeix-Raviart element (only continuous at face centers)

allowed ElementGeometries:
- Triangle2D (piecewise linear, similar to P1)
- Quadrilateral2D (similar to Q1 space)
- Tetrahedron3D (piecewise linear, similar to P1)
"""
abstract type H1CR{ncomponents} <: AbstractH1FiniteElement where {ncomponents<:Int} end


get_ncomponents(FEType::Type{<:H1CR}) = FEType.parameters[1]
get_ndofs_on_face(FEType::Type{<:H1CR}, EG::Type{<:AbstractElementGeometry}) = FEType.parameters[1]
get_ndofs_on_cell(FEType::Type{<:H1CR}, EG::Type{<:AbstractElementGeometry}) = nfaces_for_geometry(EG) * FEType.parameters[1]

get_polynomialorder(::Type{<:H1CR}, ::Type{<:Edge1D}) = 1; # 0 on continuous edges, but = 1 on edges with jumps
get_polynomialorder(::Type{<:H1CR}, ::Type{<:Triangle2D}) = 1;
get_polynomialorder(::Type{<:H1CR}, ::Type{<:Quadrilateral2D}) = 2;
get_polynomialorder(::Type{<:H1CR}, ::Type{<:Tetrahedron3D}) = 1;

get_dofmap_pattern(FEType::Type{<:H1CR}, ::Type{CellDofs}, EG::Type{<:AbstractElementGeometry}) = "F1"
get_dofmap_pattern(FEType::Type{<:H1CR}, ::Type{FaceDofs}, EG::Type{<:AbstractElementGeometry}) = "I1"
get_dofmap_pattern(FEType::Type{<:H1CR}, ::Type{BFaceDofs}, EG::Type{<:AbstractElementGeometry}) = "I1"


function init!(FES::FESpace{FEType}) where {FEType <: H1CR}
    ncomponents = get_ncomponents(FEType)
    name = "CR"
    for n = 1 : ncomponents-1
        name = name * "xCR"
    end
    FES.name = name * " (H1nc)"   

    # count number of dofs
    nfaces = num_sources(FES.xgrid[FaceNodes])
    FES.ndofs = nfaces * ncomponents
end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{ON_FACES}, exact_function!; items = [], time = 0) where {FEType <: H1CR}
    # preserve face means
    xItemVolumes = FE.xgrid[FaceVolumes]
    xItemNodes = FE.xgrid[FaceNodes]
    nitems = num_sources(xItemNodes)
    ncomponents = get_ncomponents(FEType)
    offset4component = 0:nitems:ncomponents*nitems
    if items == []
        items = 1 : nitems
    end

    # compute exact face means
    facemeans = zeros(Float64,ncomponents,nitems)
    integrate!(facemeans, FE.xgrid, ON_FACES, exact_function!; items = items, time = time)
    for item in items
        for c = 1 : ncomponents
            Target[offset4component[c]+item] = facemeans[c,item] / xItemVolumes[item]
        end
    end
end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{ON_CELLS}, exact_function!; items = [], time = 0) where {FEType <: H1CR}
    # delegate cell faces to face interpolation
    subitems = slice(FE.xgrid[CellFaces], items)
    interpolate!(Target, FE, ON_FACES, exact_function!; items = subitems, time = time)
end


# BEWARE
#
# all basis functions on a cell are nonzero on all edges,
# but only the dof associated to the face is eveluated
# when using get_basis_on_face
# this leads to lumped bestapproximations along the boundary for example

function get_basis_on_face(FEType::Type{<:H1CR}, ::Type{<:AbstractElementGeometry})
    ncomponents = get_ncomponents(FEType)
    function closure(refbasis, xref)
        for k = 1 : ncomponents
            refbasis[k,k] = 1
        end
    end
end

function get_basis_on_cell(FEType::Type{<:H1CR}, ET::Type{<:Triangle2D})
    ncomponents = get_ncomponents(FEType)
    temp = 0.0
    function closure(refbasis, xref)
        temp = 2*(xref[1]+xref[2]) - 1
        for k = 1 : ncomponents
            refbasis[3*k-2,k] = 1 - 2*xref[2]
            refbasis[3*k-1,k] = temp
            refbasis[3*k,k] = 1 - 2*xref[1]
        end
    end
end

function get_basis_on_cell(FEType::Type{<:H1CR}, ET::Type{<:Tetrahedron3D})
    ncomponents = get_ncomponents(FEType)
    temp = 0.0
    function closure(refbasis, xref)
        temp = 3*(xref[1]+xref[2]+xref[3]) - 2
        for k = 1 : ncomponents
            refbasis[4*k-3,k] = 1 - 3*xref[3]
            refbasis[4*k-2,k] = 1 - 3*xref[2]
            refbasis[4*k-1,k] = temp
            refbasis[4*k,k] = 1 - 3*xref[1]
        end
    end
end

function get_basis_on_cell(FEType::Type{<:H1CR}, ET::Type{<:Quadrilateral2D})
    ncomponents = get_ncomponents(FEType)
    a = 0.0
    b = 0.0
    temp = 0.0
    temp2 = 0.0
    temp3 = 0.0
    temp4 = 0.0
    function closure(refbasis, xref)
        a = 1 - xref[1]
        b = 1 - xref[2]
        temp = xref[1]*a + b*b - 1//4
        temp2 = xref[2]*b + xref[1]*xref[1] - 1//4
        temp3 = xref[1]*a + xref[2]*xref[2] - 1//4
        temp4 = xref[2]*b + a*a - 1//4
        for k = 1 : ncomponents
            refbasis[4*k-3,k] = temp
            refbasis[4*k-2,k] = temp2
            refbasis[4*k-1,k] = temp3
            refbasis[4*k,k] = temp4
        end
    end
end

function get_reconstruction_coefficients_on_cell!(FE::FESpace{FEType}, FER::FESpace{<:HDIVRT0}, EG::Type{<:AbstractElementGeometry}) where {FEType<:H1CR}
    xFaceVolumes::Array{Float64,1} = FE.xgrid[FaceVolumes]
    xFaceNormals::Array{Float64,2} = FE.xgrid[FaceNormals]
    xCellFaces = FE.xgrid[CellFaces]
    ncomponents = get_ncomponents(FEType)
    nf::Int = nfaces_for_geometry(EG)
    face::Int = 0
    function closure(coefficients, cell::Int) 
        # fill!(coefficients,0.0)
        for f = 1 : nf
            face = xCellFaces[f,cell]
            for k = 1 : ncomponents
                coefficients[nf*(k-1)+f,f] = xFaceVolumes[face] * xFaceNormals[k, face]
            end
        end
        return nothing
    end
end


