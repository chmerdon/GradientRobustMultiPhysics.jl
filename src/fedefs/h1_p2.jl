
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

get_polynomialorder(::Type{<:H1P2}, ::Type{<:Edge1D}) = 2;
get_polynomialorder(::Type{<:H1P2}, ::Type{<:Triangle2D}) = 2;
get_polynomialorder(::Type{<:H1P2}, ::Type{<:Quadrilateral2D}) = 3;
get_polynomialorder(::Type{<:H1P2}, ::Type{<:Tetrahedron3D}) = 2;

function init!(FES::FESpace{FEType}) where {FEType <: H1P2}
    ncomponents = get_ncomponents(FEType)
    edim = get_edim(FEType)
    name = "P2"
    for n = 1 : ncomponents-1
        name = name * "xP2"
    end
    FES.name = name * " (H1, edim=$edim)"   

    # count number of dofs

    # total number of dofs
    ndofs = 0
    nnodes = num_sources(FES.xgrid[Coordinates]) 
    ndofs4component = 0
    if edim == 1
        ncells = num_sources(FES.xgrid[CellNodes])
        ndofs = (nnodes + ncells) * ncomponents
        ndofs4component = nnodes + ncells
    elseif edim == 2
        nfaces = num_sources(FES.xgrid[FaceNodes])
        ndofs = (nnodes + nfaces) * ncomponents
        ndofs4component = nnodes + nfaces
    elseif edim == 3
        nedges = num_sources(FES.xgrid[EdgeNodes])
        ndofs = (nnodes + nedges) * ncomponents
        ndofs4component = nnodes + nedges
    end    
    FES.ndofs = ndofs 

end


function init_dofmap!(FES::FESpace{FEType}, ::Type{CellDofs}) where {FEType <: H1P2}
    xCellNodes = FES.xgrid[CellNodes]
    xCellGeometries = FES.xgrid[CellGeometries]
    ncomponents = get_ncomponents(FEType)
    ncells = num_sources(xCellNodes)
    nnodes = num_sources(FES.xgrid[Coordinates]) 
    edim = get_edim(FEType)
    if edim == 1
        dofs4item = zeros(Int32,ncomponents*(max_num_targets_per_source(xCellNodes)+1))
        ndofs4component = nnodes + ncells
    elseif edim == 2
        nfaces = num_sources(FES.xgrid[FaceNodes])
        xCellFaces = FES.xgrid[CellFaces]
        dofs4item = zeros(Int32,ncomponents*(max_num_targets_per_source(xCellNodes)+max_num_targets_per_source(xCellFaces)))
        ndofs4component = nnodes + nfaces
    elseif edim == 3
        nedges = num_sources(FES.xgrid[EdgeNodes])
        xCellEdges = FES.xgrid[CellEdges]
        dofs4item = zeros(Int32,ncomponents*(max_num_targets_per_source(xCellNodes)+max_num_targets_per_source(xCellEdges)))
        ndofs4component = nnodes + nedges
    end    
    xCellDofs = VariableTargetAdjacency(Int32)
    nnodes4item = 0
    nextra4item = 0
    for cell = 1 : ncells
        nnodes4item = num_targets(xCellNodes,cell)
        for k = 1 : nnodes4item
            dofs4item[k] = xCellNodes[k,cell]
        end
        if edim == 1 # in 1D also cell midpoints are dofs
            nextra4item = 1
            dofs4item[nnodes4item+1] = nnodes + cell
        elseif edim == 2 # in 2D also face midpoints are dofs
            nextra4item = num_targets(xCellFaces,cell)
            for k = 1 : nextra4item
                dofs4item[nnodes4item+k] = nnodes + xCellFaces[k,cell]
            end
        elseif edim == 3 # in 3D also edge midpoints are dofs
            nextra4item = num_targets(xCellEdges,cell)
            for k = 1 : nextra4item
                dofs4item[nnodes4item+k] = nnodes + xCellEdges[k,cell]
            end
        end
        ndofs4item = nnodes4item + nextra4item
        for n = 1 : ncomponents-1, k = 1:ndofs4item
            dofs4item[k+n*ndofs4item] = n*ndofs4component + dofs4item[k]
        end    
        ndofs4item *= ncomponents
        append!(xCellDofs,dofs4item[1:ndofs4item])
    end
    # save dofmap
    FES.dofmaps[CellDofs] = xCellDofs
end

function init_dofmap!(FES::FESpace{FEType}, ::Type{FaceDofs}) where {FEType <: H1P2}
    xFaceNodes = FES.xgrid[FaceNodes]
    xBFaces = FES.xgrid[BFaces]
    nfaces = num_sources(xFaceNodes)
    nnodes = num_sources(FES.xgrid[Coordinates]) 
    xFaceDofs = VariableTargetAdjacency(Int32)
    ncomponents = get_ncomponents(FEType)
    edim = get_edim(FEType)
    if edim == 1
        ncells = num_sources(FES.xgrid[CellNodes])
        dofs4item = zeros(Int32,ncomponents)
        ndofs4component = nnodes + ncells
    elseif edim == 2
        nfaces = num_sources(FES.xgrid[FaceNodes])
        xCellFaces = FES.xgrid[CellFaces]
        dofs4item = zeros(Int32,ncomponents*(max_num_targets_per_source(xFaceNodes)+1))
        ndofs4component = nnodes + nfaces
    elseif edim == 3
        nedges = num_sources(FES.xgrid[EdgeNodes])
        xFaceEdges = FES.xgrid[FaceEdges]
        maxfaceedges = max_num_targets_per_source(xFaceEdges)
        dofs4item = zeros(Int32,ncomponents*(max_num_targets_per_source(xFaceNodes)+maxfaceedges))
        ndofs4component = nnodes + nedges
    end    
    nnodes4item = 0
    nextra4item = 0
    for face = 1 : nfaces
        nnodes4item = num_targets(xFaceNodes,face)
        for k = 1 : nnodes4item
            dofs4item[k] = xFaceNodes[k,face]
        end
        if edim == 1
            nextra4item = 0
        elseif edim == 2 # in 2D also face midpoints are dofs
            nextra4item = 1
            dofs4item[nnodes4item+1] = nnodes + face
        elseif edim == 3 # in 3D also edge midpoints are dofs
            nextra4item = num_targets(xFaceEdges,face)
            for k = 1 : nextra4item
                dofs4item[nnodes4item+k] = nnodes + xFaceEdges[k,face]
            end
        end
        ndofs4item = nnodes4item + nextra4item
        for n = 1 : ncomponents-1, k = 1:ndofs4item
            dofs4item[k+n*ndofs4item] = n*ndofs4component + dofs4item[k]
        end    
        ndofs4item *= ncomponents
        append!(xFaceDofs,dofs4item[1:ndofs4item])
    end
    # save dofmap
    FES.dofmaps[FaceDofs] = xFaceDofs
end

function init_dofmap!(FES::FESpace{FEType}, ::Type{BFaceDofs}) where {FEType <: H1P2}
    xBFaceNodes = FES.xgrid[BFaceNodes]
    xBFaces = FES.xgrid[BFaces]
    nbfaces = num_sources(xBFaceNodes)
    xBFaceDofs = VariableTargetAdjacency(Int32)
    ncomponents = get_ncomponents(FEType)
    nnodes = num_sources(FES.xgrid[Coordinates]) 
    edim = get_edim(FEType)
    if edim == 1
        ncells = num_sources(FES.xgrid[CellNodes])
        dofs4item = zeros(Int32,ncomponents)
        ndofs4component = nnodes + ncells
    elseif edim == 2
        nfaces = num_sources(FES.xgrid[FaceNodes])
        xCellFaces = FES.xgrid[CellFaces]
        dofs4item = zeros(Int32,ncomponents*(max_num_targets_per_source(xBFaceNodes)+1))
        ndofs4component = nnodes + nfaces
    elseif edim == 3
        nedges = num_sources(FES.xgrid[EdgeNodes])
        xFaceEdges = FES.xgrid[FaceEdges]
        maxfaceedges = max_num_targets_per_source(xFaceEdges)
        dofs4item = zeros(Int32,ncomponents*(max_num_targets_per_source(xBFaceNodes)+maxfaceedges))
        ndofs4component = nnodes + nedges
    end    
    nnodes4item = 0
    nextra4item = 0
    for bface = 1: nbfaces
        nnodes4item = num_targets(xBFaceNodes,bface)
        for k = 1 : nnodes4item
            dofs4item[k] = xBFaceNodes[k,bface]
        end
        if edim == 1
            nextra4item = 0
        elseif edim == 2 # in 2D also face midpoints are dofs
            nextra4item = 1
            dofs4item[nnodes4item+1] = nnodes + xBFaces[bface]
        elseif edim == 3 # in 3D also edge midpoints are dofs
            nextra4item = num_targets(xFaceEdges,xBFaces[bface])
            for k = 1 : nextra4item
                dofs4item[nnodes4item+k] = nnodes + xFaceEdges[k,xBFaces[bface]]
            end
        end
        ndofs4item = nnodes4item + nextra4item
        for n = 1 : ncomponents-1, k = 1:ndofs4item
            dofs4item[k+n*ndofs4item] = n*ndofs4component + dofs4item[k]
        end    
        ndofs4item *= ncomponents
        append!(xBFaceDofs,dofs4item[1:ndofs4item])
    end
    # save dofmap
    FES.dofmaps[BFaceDofs] = xBFaceDofs
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
    end
    for node = 1 : nnodes
        for c = 1 : ncomponents
            Target[c,node] = Source[offset4component[c]+node]
        end    
    end    
end


function get_basis_on_cell(::Type{H1P2{1,1}}, ::Type{<:Vertex0D})
    function closure(xref)
        return [1]
    end
end

function get_basis_on_cell(::Type{H1P2{2,1}}, ::Type{<:Vertex0D})
    function closure(xref)
        return [1 0;
                0 1]
    end
end


function get_basis_on_cell(FEType::Type{<:H1P2}, ::Type{<:Edge1D})
    function closure(xref)
        ncomponents = get_ncomponents(FEType)
        refbasis = zeros(eltype(xref),ncomponents*3,ncomponents)
        temp = 1 - xref[1]
        for k = 1 : ncomponents
            refbasis[3*k-2,k] = 2*temp*(temp - 1//2)            # node 1
            refbasis[3*k-1,k] = 2*xref[1]*(xref[1] - 1//2)      # node 2
            refbasis[3*k,k] = 4*temp*xref[1]                    # face 1
        end
        return refbasis
    end
end

function get_basis_on_cell(FEType::Type{<:H1P2}, ::Type{<:Triangle2D})
    function closure(xref)
        ncomponents = get_ncomponents(FEType)
        refbasis = zeros(eltype(xref),ncomponents*6,ncomponents)
        temp = 1 - xref[1] - xref[2]
        for k = 1 : ncomponents
            refbasis[6*k-5,k] = 2*temp*(temp - 1//2)            # node 1
            refbasis[6*k-4,k] = 2*xref[1]*(xref[1] - 1//2)      # node 2
            refbasis[6*k-3,k] = 2*xref[2]*(xref[2] - 1//2)      # node 3
            refbasis[6*k-2,k] = 4*temp*xref[1]                  # face 1
            refbasis[6*k-1,k] = 4*xref[1]*xref[2]               # face 2
            refbasis[6*k,k] = 4*xref[2]*temp                    # face 3
        end
        return refbasis
    end
end


function get_basis_on_cell(FEType::Type{<:H1P2}, ::Type{<:Tetrahedron3D})
    function closure(xref)
        ncomponents = get_ncomponents(FEType)
        refbasis = zeros(eltype(xref),ncomponents*10,ncomponents)
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
        return refbasis
    end
end


function get_basis_on_cell(FEType::Type{<:H1P2}, ::Type{<:Quadrilateral2D})
    function closure(xref)
        ncomponents = get_ncomponents(FEType)
        refbasis = zeros(eltype(xref),ncomponents*8,ncomponents)
        refbasis[1,1] = 1 - xref[1]
        refbasis[2,1] = 1 - xref[2]
        refbasis[3,1] = 2*xref[1]*xref[2]*(xref[1]+xref[2]-3//2);
        refbasis[4,1]= -2*xref[2]*refbasis[1,1]*(xref[1]-xref[2]+1//2);
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
        return refbasis
    end
end


function get_basis_on_face(FE::Type{<:H1P2}, EG::Type{<:AbstractElementGeometry})
    function closure(xref)
        return get_basis_on_cell(FE, EG)(xref)
    end    
end
