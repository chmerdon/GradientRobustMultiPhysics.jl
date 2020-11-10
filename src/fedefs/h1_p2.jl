
"""
$(TYPEDEF)

Continuous piecewise second-order polynomials

allowed ElementGeometries:
- Edge1D (quadratic polynomials)
- Triangle2D (quadratic polynomials)
- Quadrilateral2D (Q2 space)
- Tetrahedron3D (quadratic polynomials)
"""
abstract type H1P2{ncomponents,edim} <: AbstractH1FiniteElement where {ncomponents<:Int,edim<:Int} end

get_ncomponents(FEType::Type{<:H1P2}) = FEType.parameters[1]
get_edim(FEType::Type{<:H1P2}) = FEType.parameters[2]

get_ndofs_on_face(FEType::Type{<:H1P2}, EG::Type{<:Union{AbstractElementGeometry1D, Triangle2D, Tetrahedron3D}}) = Int((FEType.parameters[2])*(FEType.parameters[2]+1)/2*FEType.parameters[1])
get_ndofs_on_cell(FEType::Type{<:H1P2}, EG::Type{<:Union{AbstractElementGeometry1D, Triangle2D, Tetrahedron3D}}) = Int((FEType.parameters[2]+1)*(FEType.parameters[2]+2)/2*FEType.parameters[1])
get_ndofs_on_cell(FEType::Type{<:H1P2}, EG::Type{<:Quadrilateral2D}) = 8*FEType.parameters[1]


get_polynomialorder(::Type{<:H1P2}, ::Type{<:Edge1D}) = 2;
get_polynomialorder(::Type{<:H1P2}, ::Type{<:Triangle2D}) = 2;
get_polynomialorder(::Type{<:H1P2}, ::Type{<:Quadrilateral2D}) = 3;
get_polynomialorder(::Type{<:H1P2}, ::Type{<:Tetrahedron3D}) = 2;


get_dofmap_pattern(FEType::Type{<:H1P2}, ::Type{CellDofs}, EG::Type{<:AbstractElementGeometry1D}) = "N1I1"
get_dofmap_pattern(FEType::Type{<:H1P2}, ::Type{CellDofs}, EG::Type{<:AbstractElementGeometry2D}) = "N1F1"
get_dofmap_pattern(FEType::Type{<:H1P2}, ::Type{CellDofs}, EG::Type{<:AbstractElementGeometry3D}) = "N1E1"
get_dofmap_pattern(FEType::Type{<:H1P2}, ::Type{FaceDofs}, EG::Type{<:AbstractElementGeometry0D}) = "N1"
get_dofmap_pattern(FEType::Type{<:H1P2}, ::Type{FaceDofs}, EG::Type{<:AbstractElementGeometry1D}) = "N1I1"
get_dofmap_pattern(FEType::Type{<:H1P2}, ::Type{FaceDofs}, EG::Type{<:AbstractElementGeometry2D}) = "N1E1"
get_dofmap_pattern(FEType::Type{<:H1P2}, ::Type{BFaceDofs}, EG::Type{<:AbstractElementGeometry0D}) = "N1"
get_dofmap_pattern(FEType::Type{<:H1P2}, ::Type{BFaceDofs}, EG::Type{<:AbstractElementGeometry1D}) = "N1I1"
get_dofmap_pattern(FEType::Type{<:H1P2}, ::Type{BFaceDofs}, EG::Type{<:AbstractElementGeometry2D}) = "N1E1"


function init!(FES::FESpace{FEType}) where {FEType <: H1P2}
    ncomponents = get_ncomponents(FEType)
    edim = get_edim(FEType)
    name = "P2"
    for n = 1 : ncomponents-1
        name = name * "xP2"
    end
    FES.name = name * " (H1, edim=$edim)"   

    # total number of dofs
    ndofs = 0
    nnodes = num_sources(FES.xgrid[Coordinates]) 
    ndofs4component = 0
    if edim == 1
        ncells = num_sources(FES.xgrid[CellNodes])
        ndofs4component = nnodes + ncells
    elseif edim == 2
        nfaces = num_sources(FES.xgrid[FaceNodes])
        ndofs4component = nnodes + nfaces
    elseif edim == 3
        nedges = num_sources(FES.xgrid[EdgeNodes])
        ndofs4component = nnodes + nedges
    end    
    FES.ndofs = ndofs4component * ncomponents

end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{<:H1P2}, exact_function!::Function; dofs = [], bonus_quadorder::Int = 0)
    xCoords = FE.xgrid[Coordinates]
    xCellNodes = FE.xgrid[CellNodes]
    xFaceNodes = FE.xgrid[FaceNodes]
    nnodes = num_sources(xCoords)
    ncells = num_sources(xCellNodes)
    nfaces = num_sources(xFaceNodes)
    FEType = eltype(FE)
    ncomponents::Int = get_ncomponents(FEType)
    edim = get_edim(FEType)

    result = zeros(Float64,ncomponents)
    xdim = size(xCoords,1)
    x = zeros(Float64,xdim)

    nnodes4item = 0
    offset4component = [0, nnodes+nfaces]
    face = 0
    if length(dofs) == 0 # interpolate at all dofs
        # interpolate at nodes
        for j = 1 : nnodes
            for k=1:xdim
                x[k] = xCoords[k,j]
            end    
            exact_function!(result,x)
            for k = 1 : ncomponents
                Target[j+offset4component[k]] = result[k]
            end    
        end

        # preserve edge integrals 
        if edim == 1 # edges are cells
            xItemNodes = FE.xgrid[CellNodes]
            xItemVolumes = FE.xgrid[CellVolumes]
            xItemDofs = FE.dofmaps[CellDofs]
            AT = ON_CELLS
        elseif edim == 2 # edges are faces
            xItemNodes = FE.xgrid[FaceNodes]
            xItemVolumes = FE.xgrid[FaceVolumes]
            xItemDofs = FE.dofmaps[FaceDofs]
            AT = ON_FACES
        elseif edgim == 3 # edges are edges
            #todo
        end

        # compute exact edge means
        nitems = num_sources(xItemNodes)
        edgemeans = zeros(Float64,ncomponents,nitems)
        integrate!(edgemeans, FE.xgrid, AT, exact_function!, bonus_quadorder, ncomponents)

        for item = 1 : nitems
            for c = 1 : ncomponents
                # subtract edge mean value of P1 part
                for dof = 1 : 2
                    edgemeans[c,item] -= Target[xItemDofs[(c-1)*3 + dof,item]] * xItemVolumes[item] / 6
                end
                # set P2 edge bubble such that edge mean is preserved
                Target[xItemDofs[3*c,item]] = 3 // 2 * edgemeans[c,item] / xItemVolumes[item]
            end
        end
    else
        item = 0
        if edim == 2
            for j in dofs 
                item = mod(j-1,nnodes+nfaces)+1
                c = Int(ceil(j/(nnodes+nfaces)))
                if item <= nnodes
                    for k=1:xdim
                        x[k] = xCoords[k,item]
                    end    
                    exact_function!(result,x)
                    Target[j] = result[c]
                elseif item > nnodes && item <= nnodes+nfaces
                    item = j - nnodes
                    nnodes4item = num_targets(xFaceNodes,item)
                    for k=1:xdim
                        x[k] = 0
                        for n=1:nnodes4item
                            x[k] += xCoords[k,xFaceNodes[n,item]]
                        end
                        x[k] /= nnodes4item    
                    end 
                    exact_function!(result,x)
                    Target[j] = result[c]
                end
            end
        elseif edim == 1
            for j in dofs 
                item = mod(j-1,nnodes+ncells)+1
                c = Int(ceil(j/(nnodes+ncells)))
                if item <= nnodes
                    for k=1:xdim
                        x[k] = xCoords[k,item]
                    end    
                    exact_function!(result,x)
                    Target[j] = result[c]
                elseif item > nnodes && item <= nnodes+ncells
                    item = j - nnodes
                    nnodes4item = num_targets(xCellNodes,item)
                    for k=1:xdim
                        x[k] = 0
                        for n=1:nnodes4item
                            x[k] += xCoords[k,xCellNodes[n,item]]
                        end
                        x[k] /= nnodes4item    
                    end 
                    exact_function!(result,x)
                    Target[j] = result[c]
                end
            end
        end
    end
end

function nodevalues!(Target::AbstractArray{<:Real,2}, Source::AbstractArray{<:Real,1}, FE::FESpace{<:H1P2})
    nnodes = num_sources(FE.xgrid[Coordinates])
    FEType = eltype(FE)
    ncomponents::Int = get_ncomponents(FEType)
    edim = get_edim(FEType)
    if edim == 1
        ncells = num_sources(FE.xgrid[CellNodes])
        offset4component = 0:(nnodes+ncells):ncomponents*(nnodes+ncells)
    elseif edim == 2
        nfaces = num_sources(FE.xgrid[FaceNodes])
        offset4component = 0:(nnodes+nfaces):ncomponents*(nnodes+nfaces)
    elseif edim == 3
        nedges = num_sources(FE.xgrid[EdgeNodes])
        offset4component = 0:(nnodes+nedges):ncomponents*(nnodes+nedges)
    end
    for node = 1 : nnodes
        for c = 1 : ncomponents
            Target[c,node] = Source[offset4component[c]+node]
        end    
    end    
end


function get_basis_on_cell(::Type{<:H1P2}, ::Type{<:Vertex0D})
    ncomponents = get_ncomponents(FEType)
    function closure(refbasis,xref)
        for k = 1 : ncomponents
            refbasis[k,k] = 1
        end
    end
end

function get_basis_on_cell(FEType::Type{<:H1P2}, ::Type{<:Edge1D})
    ncomponents = get_ncomponents(FEType)
    function closure(refbasis, xref)
        temp = 1 - xref[1]
        for k = 1 : ncomponents
            refbasis[3*k-2,k] = 2*temp*(temp - 1//2)            # node 1
            refbasis[3*k-1,k] = 2*xref[1]*(xref[1] - 1//2)      # node 2
            refbasis[3*k,k] = 4*temp*xref[1]                    # face 1
        end
    end
end

function get_basis_on_cell(FEType::Type{<:H1P2}, ::Type{<:Triangle2D})
    ncomponents = get_ncomponents(FEType)
    function closure(refbasis, xref)
        temp = 1 - xref[1] - xref[2]
        for k = 1 : ncomponents
            refbasis[6*k-5,k] = 2*temp*(temp - 1//2)            # node 1
            refbasis[6*k-4,k] = 2*xref[1]*(xref[1] - 1//2)      # node 2
            refbasis[6*k-3,k] = 2*xref[2]*(xref[2] - 1//2)      # node 3
            refbasis[6*k-2,k] = 4*temp*xref[1]                  # face 1
            refbasis[6*k-1,k] = 4*xref[1]*xref[2]               # face 2
            refbasis[6*k,k] = 4*xref[2]*temp                    # face 3
        end
    end
end


function get_basis_on_cell(FEType::Type{<:H1P2}, ::Type{<:Tetrahedron3D})
    ncomponents = get_ncomponents(FEType)
    function closure(refbasis, xref)
        temp = 1 - xref[1] - xref[2] - xref[3]
        for k = 1 : ncomponents
            refbasis[10*k-9,k] = 2*temp*(temp - 1//2)            # node 1
            refbasis[10*k-8,k] = 2*xref[1]*(xref[1] - 1//2)      # node 2
            refbasis[10*k-7,k] = 2*xref[2]*(xref[2] - 1//2)      # node 3
            refbasis[10*k-6,k] = 2*xref[3]*(xref[3] - 1//2)      # node 4
            refbasis[10*k-5,k] = 4*temp*xref[1]                  # edge 1
            refbasis[10*k-4,k] = 4*temp*xref[2]                  # edge 2
            refbasis[10*k-3,k] = 4*temp*xref[3]                  # edge 3
            refbasis[10*k-2,k] = 4*xref[1]*xref[2]               # edge 4
            refbasis[10*k-1,k] = 4*xref[1]*xref[3]               # edge 5
            refbasis[10*k  ,k] = 4*xref[2]*xref[3]               # edge 6
        end
    end
end


function get_basis_on_cell(FEType::Type{<:H1P2}, ::Type{<:Quadrilateral2D})
    ncomponents = get_ncomponents(FEType)
    function closure(refbasis, xref)
        refbasis[1,1] = 1 - xref[1]
        refbasis[2,1] = 1 - xref[2]
        refbasis[3,1] = 2*xref[1]*xref[2]*(xref[1]+xref[2]-3//2);
        refbasis[4,1] = -2*xref[2]*refbasis[1,1]*(xref[1]-xref[2]+1//2);
        refbasis[5,1] = 4*xref[1]*refbasis[1,1]*refbasis[2,1]
        refbasis[6,1] = 4*xref[2]*xref[1]*refbasis[2,1]
        refbasis[7,1] = 4*xref[1]*xref[2]*refbasis[1,1]
        refbasis[8,1] = 4*xref[2]*refbasis[1,1]*refbasis[2,1]
        refbasis[1,1] = -2*refbasis[1,1]*refbasis[2,1]*(xref[1]+xref[2]-1//2);
        refbasis[2,1] = -2*xref[1]*refbasis[2,1]*(xref[2]-xref[1]+1//2);
        for k = 2 : ncomponents
            refbasis[8*k-7,k] = refbasis[1,1] # node 1
            refbasis[8*k-6,k] = refbasis[2,1] # node 2
            refbasis[8*k-5,k] = refbasis[3,1] # node 3
            refbasis[8*k-4,k] = refbasis[4,1] # node 4
            refbasis[8*k-3,k] = refbasis[5,1] # face 1
            refbasis[8*k-2,k] = refbasis[6,1] # face 2
            refbasis[8*k-1,k] = refbasis[7,1] # face 3
            refbasis[8*k,k] = refbasis[8,1]  # face 4
        end
    end
end


function get_basis_on_face(FE::Type{<:H1P2}, EG::Type{<:AbstractElementGeometry})
    return get_basis_on_cell(FE, EG)
end
