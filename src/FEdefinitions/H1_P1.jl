
abstract type H1P1{ncomponents} <: AbstractH1FiniteElement where {ncomponents<:Int} end

get_ncomponents(::Type{H1P1{1}}) = 1
get_ncomponents(::Type{H1P1{2}}) = 2

get_polynomialorder(::Type{<:H1P1}, ::Type{<:Edge1D}) = 1;
get_polynomialorder(::Type{<:H1P1}, ::Type{<:Triangle2D}) = 1;
get_polynomialorder(::Type{<:H1P1}, ::Type{<:Quadrilateral2D}) = 2;


function init!(FES::FESpace{FEType}; dofmap_needed = true) where {FEType <: H1P1}
    ncomponents = get_ncomponents(FEType)
    name = "P1"
    for n = 1 : ncomponents-1
        name = name * "xP1"
    end
    FES.name = name * " (H1)"   

    # count number of dofs
    nnodes = num_sources(FES.xgrid[Coordinates]) 
    FES.ndofs = nnodes * ncomponents

    # generate dofmaps
    if dofmap_needed
        xCellNodes = FES.xgrid[CellNodes]
        xFaceNodes = FES.xgrid[FaceNodes]
        xCellGeometries = FES.xgrid[CellGeometries]
        xBFaceNodes = FES.xgrid[BFaceNodes]
        xBFaces = FES.xgrid[BFaces]
        ncells = num_sources(xCellNodes)
        nfaces = num_sources(xFaceNodes)
        nbfaces = num_sources(xBFaceNodes)
        xCellDofs = VariableTargetAdjacency(Int32)
        xFaceDofs = VariableTargetAdjacency(Int32)
        xBFaceDofs = VariableTargetAdjacency(Int32)
        dofs4item = zeros(Int32,ncomponents*max_num_targets_per_source(xCellNodes))
        nnodes4item = 0
        for cell = 1 : ncells
            nnodes4item = num_targets(xCellNodes,cell)
            for k = 1 : nnodes4item
                dofs4item[k] = xCellNodes[k,cell]
                for n = 1 : ncomponents-1
                    dofs4item[k+n*nnodes4item] = n*nnodes + dofs4item[k]
                end    
            end
            append!(xCellDofs,dofs4item[1:ncomponents*nnodes4item])
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

        # save dofmaps
        FES.CellDofs = xCellDofs
        FES.FaceDofs = xFaceDofs
        FES.BFaceDofs = xBFaceDofs
    end

end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{<:H1P1}, exact_function!::Function; dofs = [], bonus_quadorder::Int = 0)
    xCoords = FE.xgrid[Coordinates]
    xdim = size(xCoords,1)
    x = zeros(Float64,xdim)
    nnodes = num_sources(xCoords)
    xCellNodes = FE.xgrid[CellNodes]
    ncells = num_sources(xCellNodes)
    nnodes4item::Int = 0
    FEType = eltype(typeof(FE))
    ncomponents::Int = get_ncomponents(FEType)
    result = zeros(Float64,ncomponents)
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
    else
        item = 0
        for j in dofs 
            item = mod(j-1,nnodes)+1
            c = Int(ceil(j/nnodes))
            for k=1:xdim
                x[k] = xCoords[k,item]
            end    
            exact_function!(result,x)
            Target[j] = result[c]
        end    
    end    
end

function nodevalues!(Target::AbstractArray{<:Real,2}, Source::AbstractArray{<:Real,1}, FE::FESpace{<:H1P1})
    nnodes = num_sources(FE.xgrid[Coordinates])
    FEType = eltype(typeof(FE))
    ncomponents::Int = get_ncomponents(FEType)
    offset4component = 0:nnodes:ncomponents*nnodes
    for node = 1 : nnodes
        for c = 1 : ncomponents
            Target[c,node] = Source[offset4component[c]+node]
        end    
    end    
end

function get_basis_on_cell(::Type{H1P1{1}}, ::Type{<:Vertex0D})
    function closure(xref)
        return [1]
    end
end

function get_basis_on_cell(::Type{H1P1{2}}, ::Type{<:Vertex0D})
    function closure(xref)
        return [1 0;
                0 1]
    end
end

function get_basis_on_cell(::Type{H1P1{1}}, ::Type{<:Edge1D})
    function closure(xref)
        return [1 - xref[1];
                xref[1]]
    end
end


function get_basis_on_cell(::Type{H1P1{2}}, ::Type{<:Edge1D})
    function closure(xref)
        temp = 1 - xref[1];
        return [temp 0.0;
                xref[1] 0.0;
                0.0 temp;
                0.0 xref[1]]
    end
end

function get_basis_on_cell(::Type{H1P1{1}}, ::Type{<:Triangle2D})
    function closure(xref)
        return [1-xref[1]-xref[2];
                xref[1];
                xref[2]]
    end
end

function get_basis_on_cell(::Type{H1P1{2}}, ::Type{<:Triangle2D})
    function closure(xref)
        return [1-xref[1]-xref[2] 0.0;
                xref[1] 0.0;
                xref[2] 0.0;
                0.0 1-xref[1]-xref[2];
                0.0 xref[1];
                0.0 xref[2]]
    end
end

function get_basis_on_cell(::Type{H1P1{1}}, ::Type{<:Quadrilateral2D})
    function closure(xref)
        a = 1 - xref[1]
        b = 1 - xref[2]
        return [a*b;
                xref[1]*b;
                xref[1]*xref[2];
                xref[2]*a;
                ]
    end
end

function get_basis_on_cell(::Type{H1P1{2}}, ::Type{<:Quadrilateral2D})
    function closure(xref)
        a = 1 - xref[1]
        b = 1 - xref[2]
        return [a*b 0.0;
                xref[1]*b 0.0;
                xref[1]*xref[2] 0.0;
                xref[2]*a 0.0
                0.0 a*b;
                0.0 xref[1]*b;
                0.0 xref[1]*xref[2];
                0.0 xref[2]*a]
    end
end

function get_basis_on_face(FE::Type{<:H1P1}, EG::Type{<:AbstractElementGeometry})
    function closure(xref)
        return get_basis_on_cell(FE, EG)(xref)
    end    
end
