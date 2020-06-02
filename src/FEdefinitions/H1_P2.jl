
abstract type H1P2{ncomponents,edim} <: AbstractH1FiniteElement where {ncomponents<:Int,edim<:Int} end

get_edim(::Type{H1P2{ncomponents,1}}) where ncomponents = 1
get_edim(::Type{H1P2{ncomponents,2}}) where ncomponents = 2
get_edim(::Type{H1P2{ncomponents,3}}) where ncomponents = 3

get_ncomponents(::Type{H1P2{1,edim}}) where edim = 1
get_ncomponents(::Type{H1P2{2,edim}}) where edim = 2

get_polynomialorder(::Type{<:H1P2}, ::Type{<:Edge1D}) = 2;
get_polynomialorder(::Type{<:H1P2}, ::Type{<:Triangle2D}) = 2;
get_polynomialorder(::Type{<:H1P2}, ::Type{<:Quadrilateral2D}) = 3;

function init!(FES::FESpace{FEType}; dofmap_needed = true) where {FEType <: H1P2}
    ncomponents = get_ncomponents(FEType)
    edim = get_edim(FEType)
    name = "P2"
    for n = 1 : ncomponents-1
        name = name * "xP2"
    end
    FES.name = name * " (H1)"   

    # count number of dofs

    # total number of dofs
    ndofs = 0
    xCellNodes = FES.xgrid[CellNodes]
    xCellGeometries = FES.xgrid[CellGeometries]
    ncells = num_sources(xCellNodes)
    nfaces = num_sources(FES.xgrid[FaceNodes])
    nnodes = num_sources(FES.xgrid[Coordinates]) 
    if edim == 1
        ndofs = (nnodes + ncells) * ncomponents
    elseif edim == 2
        ndofs = (nnodes + nfaces) * ncomponents
    elseif edim == 3
        # TODO
    end    
    FES.ndofs = ndofs 

    # generate dofmaps
    if dofmap_needed
        xFaceGeometries = FES.xgrid[FaceGeometries]
        xFaceNodes = FES.xgrid[FaceNodes]
        xBFaceNodes = FES.xgrid[BFaceNodes]
        xCellFaces = FES.xgrid[CellFaces]
        xBFaces = FES.xgrid[BFaces]
        nbfaces = num_sources(xBFaceNodes)
        xCellDofs = VariableTargetAdjacency(Int32)
        xFaceDofs = VariableTargetAdjacency(Int32)
        xBFaceDofs = VariableTargetAdjacency(Int32)
        dofs4item = zeros(Int32,ncomponents*(max_num_targets_per_source(xCellNodes)+max_num_targets_per_source(xCellFaces)))
        nnodes4item = 0
        nextra4item = 0
        ndofs4item = 0
        ndofs4component = nnodes + nfaces
        for cell = 1 : ncells
            nnodes4item = num_targets(xCellNodes,cell)
            for k = 1 : nnodes4item
                dofs4item[k] = xCellNodes[k,cell]
            end
            if edim == 0 # in 1D also cell midpoints are dofs
                nextra4item = 1
                dofs4item[nnodes4item+1] = nnodes + cell
            elseif edim > 1 # in 2D also face midpoints are dofs
                nextra4item = num_targets(xCellFaces,cell)
                for k = 1 : nextra4item
                    dofs4item[nnodes4item+k] = nnodes + xCellFaces[k,cell]
                end
            elseif edim > 2 # in 3D also edge midpoints are dofs
                # TODO
            end
            ndofs4item = nnodes4item + nextra4item
            for n = 1 : ncomponents-1, k = 1:ndofs4item
                dofs4item[k+n*ndofs4item] = n*ndofs4component + dofs4item[k]
            end    
            ndofs4item *= ncomponents
            append!(xCellDofs,dofs4item[1:ndofs4item])
        end
        for face = 1 : nfaces
            nnodes4item = num_targets(xFaceNodes,face)
            for k = 1 : nnodes4item
                dofs4item[k] = xFaceNodes[k,face]
            end
            if edim == 0
                nextra4item = 0
            elseif edim > 1 # in 2D also face midpoints are dofs
                nextra4item = 1
                dofs4item[nnodes4item+1] = nnodes + face
            elseif edim > 2 # in 3D also edge midpoints are dofs
                # TODO
            end
            ndofs4item = nnodes4item + nextra4item
            for n = 1 : ncomponents-1, k = 1:ndofs4item
                dofs4item[k+n*ndofs4item] = n*ndofs4component + dofs4item[k]
            end    
            ndofs4item *= ncomponents
            append!(xFaceDofs,dofs4item[1:ndofs4item])
        end
        for bface = 1: nbfaces
            nnodes4item = num_targets(xBFaceNodes,bface)
            for k = 1 : nnodes4item
                dofs4item[k] = xBFaceNodes[k,bface]
            end
            if edim == 0
                nextra4item = 0
            elseif edim > 1 # in 2D also face midpoints are dofs
                nextra4item = 1
                dofs4item[nnodes4item+1] = nnodes + xBFaces[bface]
            elseif edim > 2 # in 3D also edge midpoints are dofs
                # TODO
            end
            ndofs4item = nnodes4item + nextra4item
            for n = 1 : ncomponents-1, k = 1:ndofs4item
                dofs4item[k+n*ndofs4item] = n*ndofs4component + dofs4item[k]
            end    
            ndofs4item *= ncomponents
            append!(xBFaceDofs,dofs4item[1:ndofs4item])
        end

        # save dofmaps
        FES.CellDofs = xCellDofs
        FES.FaceDofs = xFaceDofs
        FES.BFaceDofs = xBFaceDofs
    end

end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{<:H1P2}, exact_function!::Function; dofs = [], bonus_quadorder::Int = 0)
    xCoords = FE.xgrid[Coordinates]
    xFaceNodes = FE.xgrid[FaceNodes]
    nnodes = num_sources(xCoords)
    nfaces = num_sources(xFaceNodes)
    FEType = eltype(typeof(FE))
    ncomponents::Int = get_ncomponents(FEType)

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
        # interpolate at face midpoints
        for face = 1 : nfaces
            nnodes4item = num_targets(xFaceNodes,face)
            for k=1:xdim
                x[k] = 0
                for n=1:nnodes4item
                    x[k] += xCoords[k,xFaceNodes[n,face]]
                end
                x[k] /= nnodes4item    
            end    
            exact_function!(result,x)
            for k = 1 : ncomponents
                Target[nnodes+face+offset4component[k]] = result[k]
            end    
        end
    else
        item = 0
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
    end
end

function nodevalues!(Target::AbstractArray{<:Real,2}, Source::AbstractArray{<:Real,1}, FE::FESpace{<:H1P2})
    nnodes = num_sources(FE.xgrid[Coordinates])
    nfaces = num_sources(FE.xgrid[FaceNodes])
    FEType = eltype(typeof(FE))
    ncomponents::Int = get_ncomponents(FEType)
    offset4component = 0:(nnodes+nfaces):ncomponents*(nnodes+nfaces)
    for node = 1 : nnodes
        for c = 1 : ncomponents
            Target[c,node] = Source[offset4component[c]+node]
        end    
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
