
"""
$(TYPEDEF)

Mini finite element (continuous piecewise linear + cell bubbles)

allowed element geometries:
- Triangle2D (linear polynomials + cubic cell bubble)
- Quadrilateral2D (Q1 space + quartic cell bubble)
- Tetrahedron3D (linear polynomials + cubic cell bubble)
"""
abstract type H1MINI{ncomponents,edim} <: AbstractH1FiniteElement where {ncomponents<:Int,edim<:Int} end

get_ncomponents(FEType::Type{<:H1MINI}) = FEType.parameters[1]
get_edim(FEType::Type{<:H1MINI}) = FEType.parameters[2]
get_ndofs_on_face(FEType::Type{<:H1MINI}, EG::Type{<:AbstractElementGeometry}) = nnodes_for_geometry(EG) * FEType.parameters[1]
get_ndofs_on_cell(FEType::Type{<:H1MINI}, EG::Type{<:AbstractElementGeometry}) = (1+nnodes_for_geometry(EG)) * FEType.parameters[1]

get_polynomialorder(::Type{<:H1MINI{2,2}}, ::Type{<:Edge1D}) = 1
get_polynomialorder(::Type{<:H1MINI{2,2}}, ::Type{<:Triangle2D}) = 3;
get_polynomialorder(::Type{<:H1MINI{2,2}}, ::Type{<:Quadrilateral2D}) = 4;

get_polynomialorder(::Type{<:H1MINI{3,3}}, ::Type{<:Triangle2D}) = 1;
get_polynomialorder(::Type{<:H1MINI{3,3}}, ::Type{<:Tetrahedron3D}) = 4;

function init!(FES::FESpace{FEType}) where {FEType <: H1MINI}
    ncomponents = get_ncomponents(FEType)
    name = "MINI"
    for n = 1 : ncomponents-1
        name = name * "xMINI"
    end
    FES.name = name * " (H1)"   

    # count number of dofs
    nnodes = num_sources(FES.xgrid[Coordinates]) 
    ncells = num_sources(FES.xgrid[CellNodes])
    FES.ndofs = (nnodes + ncells) * ncomponents

end

get_dofmap_pattern(FEType::Type{<:H1MINI}, ::Type{CellDofs}, EG::Type{<:AbstractElementGeometry}) = "N1I1"
get_dofmap_pattern(FEType::Type{<:H1MINI}, ::Type{FaceDofs}, EG::Type{<:AbstractElementGeometry}) = "N1C1" # quick and dirty: C1 is ignored on faces, but need to calculate offset
get_dofmap_pattern(FEType::Type{<:H1MINI}, ::Type{BFaceDofs}, EG::Type{<:AbstractElementGeometry}) = "N1C1" # quick and dirty: C1 is ignored on faces, but need to calculate offset


function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{<:H1MINI}, exact_function!::Function; dofs = [], bonus_quadorder::Int = 0)
    xCoords = FE.xgrid[Coordinates]
    xdim = size(xCoords,1)
    x = zeros(Float64,xdim)
    nnodes = num_sources(xCoords)
    xCellNodes = FE.xgrid[CellNodes]
    ncells = num_sources(xCellNodes)
    nnodes4item::Int = 0
    FEType = eltype(FE)
    ncomponents = get_ncomponents(FEType)
    result = zeros(Float64,ncomponents)
    linpart = 0.0
    if length(dofs) == 0 # interpolate at all dofs
        for j = 1 : num_sources(xCoords)
            for k=1:xdim
                x[k] = xCoords[k,j]
            end    
            exact_function!(result,x)
            for c = 1 : ncomponents
                Target[j+(c-1)*(nnodes+ncells)] = result[c]
            end
        end

        for cell=1:ncells
            nnodes4item = num_targets(xCellNodes,cell)
            fill!(x,0.0)
            for j=1:xdim
                for k=1:nnodes4item
                    x[j] += xCoords[j,xCellNodes[k,cell]]
                end
                x[j] /= nnodes4item
            end
            exact_function!(result,x)
            for c = 1 : ncomponents
                linpart = 0.0
                for k=1:nnodes4item
                    linpart += Target[xCellNodes[k,cell]+(c-1)*(nnodes+ncells)]
                end
                Target[(c-1)*(nnodes+ncells)+nnodes+cell] = result[c] - linpart / nnodes4item
            end
        end
    else
        item = 0
        for j in dofs 
            item = mod(j-1,nnodes+ncells)+1
            c = Int(ceil(j/(nnodes+ncells)))
            if item <= nnodes
                for k=1:xdim
                    x[k] = xCoords[k,item]
                end    
                exact_function!(result,x)
                Target[j] = result[c]
            else # cell bubble
                nnodes4item = num_targets(xCellNodes,cell)
                fill!(x,0.0)
                for j=1:xdim
                    for k=1:nnodes4item
                        x[j] += xCoords[j,xCellNodes[k,cell]]
                    end
                    x[j] /= nnodes4item
                end
                exact_function!(result,x)
                linpart = 0.0
                for k=1:nnodes4item
                    linpart += Target[xCellNodes[k,cell]+(c-1)*(nnodes+ncells)]
                end
                Target[(c-1)*(nnodes+ncells)+nnodes+cell] = result[c] - linpart / nnodes4item    
            end    
        end    
    end    
end

function nodevalues!(Target::AbstractArray{<:Real,2}, Source::AbstractArray{<:Real,1}, FE::FESpace{<:H1MINI})
    nnodes = num_sources(FE.xgrid[Coordinates])
    ncells = num_sources(FE.xgrid[CellNodes])
    FEType = eltype(FE)
    ncomponents = get_ncomponents(FEType)
    offset4component = 0:(nnodes+ncells):ncomponents*(nnodes+ncells)
    for node = 1 : nnodes
        for c = 1 : ncomponents
            Target[c,node] = Source[offset4component[c]+node]
        end    
    end    
end


function get_basis_on_face(FEType::Type{<:H1MINI}, EG::Type{<:AbstractElementGeometry})
    ncomponents = get_ncomponents(FEType)
    # same as P1
    return get_basis_on_face(H1P1{ncomponents}, EG)
end

function get_basis_on_cell(FEType::Type{<:H1MINI}, EG::Type{<:Triangle2D})
    ncomponents = get_ncomponents(FEType)
    refbasis_P1 = get_basis_on_cell(H1P1{1}, EG)
    offset = get_ndofs_on_cell(H1P1{1}, EG) + 1
    function closure(refbasis, xref)
        refbasis_P1(refbasis, xref)
        # add cell bubbles to P1 basis
        refbasis[offset,1] = 27*(1-xref[1]-xref[2])*xref[1]*xref[2]
        for k = 1 : ncomponents-1, j = 1 : offset
            refbasis[k*offset+j,k+1] = refbasis[j,1]
        end
    end
end

function get_basis_on_cell(FEType::Type{<:H1MINI}, EG::Type{<:Quadrilateral2D})
    ncomponents = get_ncomponents(FEType)
    refbasis_P1 = get_basis_on_cell(H1P1{1}, EG)
    offset = get_ndofs_on_cell(H1P1{1}, EG) + 1
    function closure(refbasis, xref)
        refbasis_P1(refbasis, xref)
        # add cell bubbles to P1 basis
        refbasis[offset,1] = 16*(1-xref[1])*(1-xref[2])*xref[1]*xref[2]
        for k = 1 : ncomponents-1, j = 1 : offset
            refbasis[k*offset+j,k+1] = refbasis[j,1]
        end
    end
end

function get_basis_on_cell(FEType::Type{<:H1MINI}, EG::Type{<:Tetrahedron3D})
    ncomponents = get_ncomponents(FEType)
    refbasis_P1 = get_basis_on_cell(H1P1{1}, EG)
    offset = get_ndofs_on_cell(H1P1{1}, EG) + 1
    function closure(refbasis, xref)
        refbasis_P1(refbasis, xref)
        # add cell bubbles to P1 basis
        refbasis[offset,1] = 81*(1-xref[1]-xref[2]-xref[3])*xref[1]*xref[2]*xref[3]
        for k = 1 : ncomponents-1, j = 1 : offset
            refbasis[k*offset+j,k+1] = refbasis[j,1]
        end
    end
end