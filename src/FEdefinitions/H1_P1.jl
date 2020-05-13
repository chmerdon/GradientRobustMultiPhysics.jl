struct FEH1P1{ncomponents} <: AbstractH1FiniteElement where {ncomponents<:Int}
    name::String                         # full name of finite element (used in messages)
    xgrid::ExtendableGrid                # link to xgrid 
    CellDofs::VariableTargetAdjacency    # place to save cell dofs (filled by constructor)
    FaceDofs::VariableTargetAdjacency    # place to save face dofs (filled by constructor)
    BFaceDofs::VariableTargetAdjacency   # place to save bface dofs (filled by constructor)
    ndofs::Int32
end

function getH1P1FiniteElement(xgrid::ExtendableGrid, ncomponents::Int)
    name = "P1"
    for n = 1 : ncomponents-1
        name = name * "xP1"
    end
    name = name * " (H1)"    

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

    return FEH1P1{ncomponents}(name,xgrid,xCellDofs,xFaceDofs,xBFaceDofs,nnodes * ncomponents)
end


get_ncomponents(::Type{FEH1P1{1}}) = 1
get_ncomponents(::Type{FEH1P1{2}}) = 2

get_polynomialorder(::Type{<:FEH1P1}, ::Type{<:Edge1D}) = 1;
get_polynomialorder(::Type{<:FEH1P1}, ::Type{<:Triangle2D}) = 1;
get_polynomialorder(::Type{<:FEH1P1}, ::Type{<:Quadrilateral2D}) = 2;


function interpolate!(Target::AbstractArray{<:Real,1}, FE::FEH1P1{1}, exact_function!::Function; dofs = [], bonus_quadorder::Int = 0)
    xCoords = FE.xgrid[Coordinates]
    result = zeros(Float64,1)
    xdim = size(xCoords,1)
    x = zeros(Float64,xdim)
    nnodes = num_sources(xCoords)
    if length(dofs) == 0 # interpolate at all dofs
        for j = 1 : num_sources(xCoords)
            for k=1:xdim
                x[k] = xCoords[k,j]
            end    
            exact_function!(result,x)
            Target[j] = result[1]
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

function get_basis_on_cell(::Type{FEH1P1{1}}, ::Type{<:Edge1D})
    function closure(xref)
        return [1 - xref[1];
                xref[1]]
    end
end


function get_basis_on_cell(::Type{FEH1P1{2}}, ::Type{<:Edge1D})
    function closure(xref)
        temp = 1 - xref[1];
        return [temp 0.0;
                xref[1] 0.0;
                0.0 temp;
                0.0 xref[1]]
    end
end

function get_basis_on_cell(::Type{FEH1P1{1}}, ::Type{<:Triangle2D})
    function closure(xref)
        return [1-xref[1]-xref[2];
                xref[1];
                xref[2]]
    end
end

function get_basis_on_cell(::Type{FEH1P1{2}}, ::Type{<:Triangle2D})
    function closure(xref)
        return [1-xref[1]-xref[2] 0.0;
                xref[1] 0.0;
                xref[2] 0.0;
                0.0 1-xref[1]-xref[2];
                0.0 xref[1];
                0.0 xref[2]]
    end
end

function get_basis_on_cell(::Type{FEH1P1{1}}, ::Type{<:Quadrilateral2D})
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

function get_basis_on_cell(::Type{FEH1P1{2}}, ::Type{<:Quadrilateral2D})
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

function get_basis_on_face(FE::Type{<:FEH1P1}, EG::Type{<:AbstractElementGeometry})
    function closure(xref)
        return get_basis_on_cell(FE, EG)(xref[1:end-1])
    end    
end
