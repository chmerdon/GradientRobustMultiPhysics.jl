struct FEH1MINI{ncomponents,spacedim} <: AbstractH1FiniteElement where {ncomponents<:Int,spacedim<:Int}
    name::String                         # full name of finite element (used in messages)
    xgrid::ExtendableGrid                # link to xgrid 
    CellDofs::VariableTargetAdjacency    # place to save cell dofs (filled by constructor)
    FaceDofs::VariableTargetAdjacency    # place to save face dofs (filled by constructor)
    BFaceDofs::VariableTargetAdjacency   # place to save bface dofs (filled by constructor)
    ndofs::Int32
end

function getH1MINIFiniteElement(xgrid::ExtendableGrid, ncomponents::Int)
    name = "MINI (H1, $(ncomponents)d)"    

    # generate celldofs
    dim = size(xgrid[Coordinates],1) 
    xCellNodes = xgrid[CellNodes]
    xFaceNodes = xgrid[FaceNodes]
    xCellGeometries = xgrid[CellGeometries]
    xBFaceNodes = xgrid[BFaceNodes]
    xBFaces = xgrid[BFaces]
    ncells = num_sources(xCellNodes)
    nfaces = num_sources(xFaceNodes)
    nbfaces = num_sources(xBFaceNodes)
    nnodes = num_sources(xgrid[Coordinates])

    # generate dofmaps
    xCellDofs = VariableTargetAdjacency(Int32)
    xFaceDofs = VariableTargetAdjacency(Int32)
    xBFaceDofs = VariableTargetAdjacency(Int32)
    dofs4item = zeros(Int32,ncomponents*(max_num_targets_per_source(xCellNodes)+1))
    nnodes4item = 0
    for cell = 1 : ncells
        nnodes4item = num_targets(xCellNodes,cell)
        for k = 1 : nnodes4item
            dofs4item[k] = xCellNodes[k,cell]
            for n = 1 : ncomponents-1
                dofs4item[k+n*nnodes4item] = n*nnodes + dofs4item[k]
            end    
        end
        for k = 1 : ncomponents
            dofs4item[ncomponents*nnodes4item+k] = ncomponents*nnodes + (k-1)*ncells + cell
        end
        append!(xCellDofs,dofs4item[1:ncomponents*(nnodes4item+1)])
    end
    for face = 1 : nfaces
        nnodes4item = num_targets(xFaceNodes,face)
        for k = 1 : nnodes4item
            dofs4item[k] = xFaceNodes[k,face]
            for n = 1 : ncomponents-1
                dofs4item[k+n*nnodes4item] = n*nnodes + dofs4item[k]
            end    
        end
        append!(xFaceDofs,dofs4item[1:ncomponents*nnodes4item])
    end
    for bface = 1: nbfaces
        nnodes4item = num_targets(xBFaceNodes,bface)
        for k = 1 : nnodes4item
            dofs4item[k] = xBFaceNodes[k,bface]
            for n = 1 : ncomponents-1
                dofs4item[k+n*nnodes4item] = n*nnodes + dofs4item[k]
            end    
        end
        append!(xBFaceDofs,dofs4item[1:ncomponents*nnodes4item])
    end

    return FEH1MINI{ncomponents,dim}(name,xgrid,xCellDofs,xFaceDofs,xBFaceDofs,(nnodes + ncells) * ncomponents)
end


get_ncomponents(::Type{FEH1MINI{1,2}}) = 1
get_ncomponents(::Type{FEH1MINI{2,2}}) = 2

get_polynomialorder(::Type{<:FEH1MINI{2,2}}, ::Type{<:Edge1D}) = 1
get_polynomialorder(::Type{<:FEH1MINI{2,2}}, ::Type{<:Triangle2D}) = 3;
get_polynomialorder(::Type{<:FEH1MINI{2,2}}, ::Type{<:Quadrilateral2D}) = 4;


function interpolate!(Target::AbstractArray{<:Real,1}, FE::FEH1MINI, exact_function!::Function; dofs = [], bonus_quadorder::Int = 0)
    xCoords = FE.xgrid[Coordinates]
    xdim = size(xCoords,1)
    x = zeros(Float64,xdim)
    nnodes = num_sources(xCoords)
    xCellNodes = FE.xgrid[CellNodes]
    ncells = num_sources(xCellNodes)
    nnodes4item::Int = 0
    ncomponents::Int = get_ncomponents(typeof(FE))
    result = zeros(Float64,ncomponents)
    linpart = 0.0
    if length(dofs) == 0 # interpolate at all dofs
        for j = 1 : num_sources(xCoords)
            for k=1:xdim
                x[k] = xCoords[k,j]
            end    
            exact_function!(result,x)
            for c = 1 : ncomponents
                Target[j+(c-1)*nnodes] = result[c]
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
                    linpart += Target[xCellNodes[k,cell]+(c-1)*nnodes]
                end
                Target[(c-1)*ncells+ncomponents*nnodes+cell] = result[c] - linpart / nnodes4item
            end
        end
    else
        item = 0
        for j in dofs 
            if j <= ncomponents*nnodes
                item = mod(j-1,nnodes)+1
                c = Int(ceil(j/nnodes))
                for k=1:xdim
                    x[k] = xCoords[k,item]
                end    
                exact_function!(result,x)
                Target[j] = result[c]
            else # cell bubble
                j = j - ncomponents*nnodes
                cell = mod(j-1,ncells)+1
                c = Int(ceil(j/ncells))
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
                    linpart += Target[xCellNodes[k,cell]+(c-1)*nnodes]
                end
                Target[(c-1)*ncells+ncomponents*nnodes+cell] = result[c] - linpart / nnodes4item    
            end    
        end    
    end    
end

function nodevalues!(Target::AbstractArray{<:Real,2}, Source::AbstractArray{<:Real,1}, FE::FEH1MINI)
    nnodes = num_sources(FE.xgrid[Coordinates])
    ncomponents = get_ncomponents(typeof(FE))
    offset4component = 0:nnodes:ncomponents*nnodes
    for node = 1 : nnodes
        for c = 1 : ncomponents
            Target[c,node] = Source[offset4component[c]+node]
        end    
    end    
end


function get_basis_on_face(::Type{FEH1MINI{1,2}}, ::Type{<:Edge1D})
    function closure(xref)
        return [1 - xref[1];
                xref[1]]
    end
end


function get_basis_on_face(::Type{FEH1MINI{2,2}}, ::Type{<:Edge1D})
    function closure(xref)
        temp = 1 - xref[1];
        return [temp 0.0;
                xref[1] 0.0;
                0.0 temp;
                0.0 xref[1]]
    end
end

function get_basis_on_cell(::Type{FEH1MINI{1,2}}, ::Type{<:Triangle2D})
    function closure(xref)
        temp = 1-xref[1]-xref[2]
        return [temp;
                xref[1];
                xref[2];
                9*temp*xref[1]*xref[2]]
    end
end

function get_basis_on_cell(::Type{FEH1MINI{2,2}}, ::Type{<:Triangle2D})
    function closure(xref)
        temp = 1-xref[1]-xref[2]
        cb = 9*temp*xref[1]*xref[2]
        return [temp 0.0;
                xref[1] 0.0;
                xref[2] 0.0;
                0.0 temp;
                0.0 xref[1];
                0.0 xref[2];
                cb 0.0;
                0.0 cb]
    end
end

function get_basis_on_cell(::Type{FEH1MINI{1,2}}, ::Type{<:Quadrilateral2D})
    function closure(xref)
        a = 1 - xref[1]
        b = 1 - xref[2]
        return [a*b;
                xref[1]*b;
                xref[1]*xref[2];
                xref[2]*a;
                16*a*b*xref[1]*xref[2]]
    end
end

function get_basis_on_cell(::Type{FEH1MINI{2,2}}, ::Type{<:Quadrilateral2D})
    function closure(xref)
        a = 1 - xref[1]
        b = 1 - xref[2]
        cb = 16*a*b*xref[1]*xref[2]
        return [a*b 0.0;
                xref[1]*b 0.0;
                xref[1]*xref[2] 0.0;
                xref[2]*a 0.0
                0.0 a*b;
                0.0 xref[1]*b;
                0.0 xref[1]*xref[2];
                0.0 xref[2]*a;
                cb 0.0;
                0.0 cb]
    end
end

