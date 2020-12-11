
"""
$(TYPEDEF)

Continuous piecewise first-order polynomials with arbitrary number of components.

allowed ElementGeometries:
- Edge1D (linear polynomials)
- Triangle2D (linear polynomials)
- Quadrilateral2D (Q1 space)
- Tetrahedron3D (linear polynomials)
- Hexahedron3D (Q1 space)
"""
abstract type H1P1{ncomponents} <: AbstractH1FiniteElement where {ncomponents<:Int} end

get_ncomponents(FEType::Type{<:H1P1}) = FEType.parameters[1] # is this okay?
get_ndofs_on_face(FEType::Type{<:H1P1}, EG::Type{<:AbstractElementGeometry}) = nnodes_for_geometry(EG) * FEType.parameters[1]
get_ndofs_on_cell(FEType::Type{<:H1P1}, EG::Type{<:AbstractElementGeometry}) = nnodes_for_geometry(EG) * FEType.parameters[1]

get_polynomialorder(::Type{<:H1P1}, ::Type{<:Edge1D}) = 1;
get_polynomialorder(::Type{<:H1P1}, ::Type{<:Triangle2D}) = 1;
get_polynomialorder(::Type{<:H1P1}, ::Type{<:Tetrahedron3D}) = 1;
get_polynomialorder(::Type{<:H1P1}, ::Type{<:Quadrilateral2D}) = 2;
get_polynomialorder(::Type{<:H1P1}, ::Type{<:Hexahedron3D}) = 3;

get_dofmap_pattern(FEType::Type{<:H1P1}, ::Type{CellDofs}, EG::Type{<:AbstractElementGeometry}) = "N1"
get_dofmap_pattern(FEType::Type{<:H1P1}, ::Type{FaceDofs}, EG::Type{<:AbstractElementGeometry}) = "N1"
get_dofmap_pattern(FEType::Type{<:H1P1}, ::Type{BFaceDofs}, EG::Type{<:AbstractElementGeometry}) = "N1"

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{AT_NODES}, exact_function; items = [], bonus_quadorder::Int = 0, time = 0) where {FEType <: H1P1}
    nnodes = size(FE.xgrid[Coordinates],2)
    point_evaluation!(Target, FE, AT_NODES, exact_function; items = items, component_offset = nnodes, time = time)
end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{ON_EDGES}, exact_function; items = [], bonus_quadorder::Int = 0, time = 0) where {FEType <: H1P1}
    # delegate edge nodes to node interpolation
    subitems = slice(FE.xgrid[EdgeNodes], items)
    interpolate!(Target, FE, AT_NODES, exact_function; items = subitems, bonus_quadorder = bonus_quadorder, time = time)
end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{ON_FACES}, exact_function; items = [], bonus_quadorder::Int = 0, time = 0) where {FEType <: H1P1}
    # delegate face nodes to node interpolation
    subitems = slice(FE.xgrid[FaceNodes], items)
    interpolate!(Target, FE, AT_NODES, exact_function; items = subitems, bonus_quadorder = bonus_quadorder, time = time)
end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{ON_CELLS}, exact_function; items = [], bonus_quadorder::Int = 0, time = 0) where {FEType <: H1P1}
    if FE.broken == true
        # broken interpolation
        point_evaluation_broken!(Target, FE, ON_CELLS, exact_function; items = items, time = time)
    else
        # delegate cell nodes to node interpolation
        subitems = slice(FE.xgrid[CellNodes], items)
        interpolate!(Target, FE, AT_NODES, exact_function; items = subitems, bonus_quadorder = bonus_quadorder, time = time)
    end
end

function get_basis_on_cell(FEType::Type{<:H1P1}, ET::Type{<:Union{Vertex0D,AbstractElementGeometry1D,Triangle2D,Tetrahedron3D}})
    ncomponents = get_ncomponents(FEType)
    function closure(refbasis, xref)
        edim = dim_element(ET)
        for k = 1 : ncomponents
            refbasis[(edim+1)*k-edim,k] = 1
            for j = 1 : edim
                refbasis[(edim+1)*k-edim,k] -= xref[j]
                refbasis[(edim+1)*k-edim+j,k] = xref[j]
            end
        end
    end
end

function get_basis_on_cell(FEType::Type{<:H1P1}, ::Type{<:Quadrilateral2D})
    ncomponents = get_ncomponents(FEType)
    function closure(refbasis, xref)
        a = 1 - xref[1]
        b = 1 - xref[2]
        for k = 1 : ncomponents
            refbasis[4*k-3,k] = a*b
            refbasis[4*k-2,k] = xref[1]*b
            refbasis[4*k-1,k] = xref[1]*xref[2]
            refbasis[4*k,k]   = xref[2]*a
        end
    end
end


function get_basis_on_cell(FEType::Type{<:H1P1}, ::Type{<:Hexahedron3D})
    ncomponents = get_ncomponents(FEType)
    function closure(refbasis, xref)
        a = 1 - xref[1]
        b = 1 - xref[2]
        c = 1 - xref[3]
        for k = 1 : ncomponents
            refbasis[8*k-7,k] = a*b*c
            refbasis[8*k-6,k] = xref[1]*b*c 
            refbasis[8*k-5,k] = xref[1]*xref[2]*c
            refbasis[8*k-4,k] = xref[2]*a*c
            refbasis[8*k-3,k] = xref[3]*a*b
            refbasis[8*k-2,k] = xref[1]*b*xref[3]
            refbasis[8*k-1,k] = xref[1]*xref[2]*xref[3]
            refbasis[8*k,k] = a*xref[2]*xref[3]
        end
    end
end

function get_basis_on_face(FE::Type{<:H1P1}, EG::Type{<:AbstractElementGeometry})
    cell_basis = get_basis_on_cell(FE, EG)
    function closure(refbasis,xref)
        return cell_basis(refbasis,xref)
    end    
end
