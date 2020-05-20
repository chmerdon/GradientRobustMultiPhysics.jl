struct FEH1CR{ncomponents} <: AbstractH1FiniteElement where {ncomponents<:Int}
    name::String                         # full name of finite element (used in messages)
    xgrid::ExtendableGrid                # link to xgrid 
    CellDofs::VariableTargetAdjacency    # place to save cell dofs (filled by constructor)
    FaceDofs::VariableTargetAdjacency    # place to save face dofs (filled by constructor)
    BFaceDofs::VariableTargetAdjacency   # place to save bface dofs (filled by constructor)
    ndofs::Int32
end

function getH1CRFiniteElement(xgrid::ExtendableGrid, ncomponents::Int)
    name = "CR"
    for n = 1 : ncomponents-1
        name = name * "xCR"
    end
    name = name * " (H1nc)"    
 

    # generate dofmaps
    xFaceNodes = xgrid[FaceNodes]
    xCellFaces = xgrid[CellFaces]
    xCellSigns = xgrid[CellSigns]
    xBFaceNodes = xgrid[BFaceNodes]
    xBFaces = xgrid[BFaces]
    ncells = num_sources(xCellFaces)
    nfaces = num_sources(xFaceNodes)
    nbfaces = num_sources(xBFaceNodes)
    xCellDofs = VariableTargetAdjacency(Int32)
    xFaceDofs = VariableTargetAdjacency(Int32)
    xBFaceDofs = VariableTargetAdjacency(Int32)
    dofs4item = zeros(Int32,ncomponents*max_num_targets_per_source(xCellFaces))
    nfaces4item = 0
    for cell = 1 : ncells
        nfaces4item = num_targets(xCellFaces,cell)
        for k = 1 : nfaces4item, c = 1 : ncomponents
            dofs4item[k+(c-1)*nfaces4item] = xCellFaces[k,cell] + (c-1)*nfaces
        end
        append!(xCellDofs,dofs4item[1:nfaces4item*ncomponents])
    end
    for face = 1 : nfaces
        for c = 1 : ncomponents
            dofs4item[c] = face + (c-1)*nfaces
        end
        append!(xFaceDofs,dofs4item[1:ncomponents])
    end
    for bface = 1: nbfaces
        for c = 1 : ncomponents
            dofs4item[c] = xBFaces[bface] + (c-1)*nfaces
        end
        append!(xBFaceDofs,dofs4item[1:ncomponents])
    end

    return FEH1CR{ncomponents}(name,xgrid,xCellDofs,xFaceDofs,xBFaceDofs,nfaces * ncomponents)
end


get_ncomponents(::Type{FEH1CR{1}}) = 1
get_ncomponents(::Type{FEH1CR{2}}) = 2

get_polynomialorder(::Type{<:FEH1CR}, ::Type{<:Edge1D}) = 0;
get_polynomialorder(::Type{<:FEH1CR}, ::Type{<:Edge1DWithParent{<:Triangle2D}}) = 0;
get_polynomialorder(::Type{<:FEH1CR}, ::Type{<:Edge1DWithParent{<:Quadrilateral2D}}) = 1;
get_polynomialorder(::Type{<:FEH1CR}, ::Type{<:Triangle2D}) = 1;
get_polynomialorder(::Type{<:FEH1CR}, ::Type{<:Quadrilateral2D}) = 2;


function interpolate!(Target::AbstractArray{<:Real,1}, FE::FEH1CR, exact_function!::Function; dofs = [], bonus_quadorder::Int = 0)
    xCoords = FE.xgrid[Coordinates]
    xdim = size(xCoords,1)
    x = zeros(Float64,xdim)
    xFaceNodes = FE.xgrid[FaceNodes]
    nfaces = num_sources(xFaceNodes)
    nnodes4item::Int = 0
    ncomponents::Int = get_ncomponents(typeof(FE))
    result = zeros(Float64,ncomponents)
    if length(dofs) == 0 # interpolate at all dofs
        for face = 1 : nfaces
            #compute face midpoints
            fill!(x,0.0)
            nnodes4item = num_targets(xFaceNodes,face)
            for j=1:xdim
                for k=1:nnodes4item
                    x[j] += xCoords[j,xFaceNodes[k,face]]
                end
                x[j] /= nnodes4item
            end
            exact_function!(result,x)
            for c = 1 : ncomponents
                Target[face+(c-1)*nfaces] = result[c]
            end
        end
    else
        face = 0
        for j in dofs 
            face = mod(j-1,nfaces)+1
            c = Int(ceil(j/nfaces))
            #compute face midpoints
            fill!(x,0.0)
            nnodes4item = num_targets(xFaceNodes,face)
            for j=1:xdim
                for k=1:nnodes4item
                    x[j] += xCoords[j,xFaceNodes[k,face]]
                end
                x[j] /= nnodes4item
            end
            exact_function!(result,x)
            Target[j] = result[c]
        end    
    end    
end

# BEWARE
#
# all basis functions on a cell are nonzero on all edges,
# but only the dof associated to the face is eveluated
# when using get_basis_on dof
# this leads to lumped bestapproximations along the boundary for example

function get_basis_on_face(::Type{FEH1CR{1}}, ::Type{<:Edge1D})
    function closure(xref)
        return [1.0]
    end
end

function get_basis_on_face(::Type{FEH1CR{2}}, ::Type{<:Edge1D})
    function closure(xref)
        return [1.0 0.0;
                0.0 1.0]
    end
end

function get_basis_on_cell(::Type{FEH1CR{1}}, ::Type{<:Triangle2D})
    function closure(xref)
        return [1 - 2*xref[2] 0.0;
                2*(xref[1]+xref[2]) - 1 0.0;
                1 - 2*xref[1] 0.0]
    end
end

function get_basis_on_cell(::Type{FEH1CR{2}}, ::Type{<:Triangle2D})
    function closure(xref)
        temp = 1 - 2*xref[2]
        temp2 = 2*(xref[1]+xref[2]) - 1
        temp3 = 1 - 2*xref[1]
        return [temp 0.0;
                temp2 0.0;
                temp3 0.0;
                0.0 temp;
                0.0 temp2;
                0.0 temp3]
    end            
end

function get_basis_on_cell(::Type{FEH1CR{1}}, ::Type{<:Quadrilateral2D})
    function closure(xref)
        a = 1 - xref[1]
        b = 1 - xref[2]
        temp = xref[1]*a - xref[2]*b + b - 1//4
        temp2 = xref[2]*b - xref[1]*a + xref[1] - 1//4
        temp3 = temp - b + xref[2]
        temp4 = temp2 - xref[1] + a
        return [temp;
                temp2;
                temp3;
                temp4]
    end
end

function get_basis_on_cell(::Type{FEH1CR{2}}, ::Type{<:Quadrilateral2D})
    function closure(xref)
        a = 1 - xref[1]
        b = 1 - xref[2]
        temp = xref[1]*a - xref[2]*b + b - 1//4
        temp2 = xref[2]*b - xref[1]*a + xref[1] - 1//4
        temp3 = temp - b + xref[2]
        temp4 = temp2 - xref[1] + a
        return [temp 0.0;
                temp2 0.0;
                temp3 0.0;
                temp4 0.0;
                0.0 temp;
                0.0 temp2;
                0.0 temp3;
                0.0 temp4]
    end
end

