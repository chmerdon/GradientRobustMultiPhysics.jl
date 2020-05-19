struct FEH1BR{ncomponents} <: AbstractH1FiniteElementWithCoefficients where {ncomponents<:Int}
    name::String                         # full name of finite element (used in messages)
    xgrid::ExtendableGrid                # link to xgrid 
    CellDofs::VariableTargetAdjacency    # place to save cell dofs (filled by constructor)
    FaceDofs::VariableTargetAdjacency    # place to save face dofs (filled by constructor)
    BFaceDofs::VariableTargetAdjacency   # place to save bface dofs (filled by constructor)
    xFaceNormals::Array{Float64,2}        # link to coefficient values
    xFaceVolumes::Array{Float64,1}        # link to coefficient values
    xCellFaces::VariableTargetAdjacency    # link to coefficient indices
    ndofs::Int32
end

function getH1BRFiniteElement(xgrid::ExtendableGrid)
    name = "BR (H1)"    

    # generate celldofs
    dim = size(xgrid[Coordinates],1) 
    ncomponents = dim

    xCellNodes = xgrid[CellNodes]
    xFaceNodes = xgrid[FaceNodes]
    xCellFaces = xgrid[CellFaces]
    xFaceNormals = xgrid[FaceNormals]
    xFaceVolumes = xgrid[FaceVolumes]
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
    dofs4item = zeros(Int32,(ncomponents+1)*max_num_targets_per_source(xCellNodes))
    nnodes4item = 0
    for cell = 1 : ncells
        nnodes4item = num_targets(xCellNodes,cell)
        for k = 1 : nnodes4item
            dofs4item[k] = xCellNodes[k,cell]
            for n = 1 : ncomponents-1
                dofs4item[k+n*nnodes4item] = n*nnodes + dofs4item[k]
            end    
            dofs4item[ncomponents*nnodes4item+k] = ncomponents*nnodes + xCellFaces[k,cell]
        end
        append!(xCellDofs,dofs4item[1:(ncomponents+1)*nnodes4item])
    end
    for face = 1 : nfaces
        nnodes4item = num_targets(xFaceNodes,face)
        for k = 1 : nnodes4item
            dofs4item[k] = xFaceNodes[k,face]
            for n = 1 : ncomponents-1
                dofs4item[k+n*nnodes4item] = n*nnodes + dofs4item[k]
            end    
        end
        dofs4item[ncomponents*nnodes4item+1] = ncomponents*nnodes + face
        append!(xFaceDofs,dofs4item[1:ncomponents*nnodes4item+1])
    end
    for bface = 1: nbfaces
        nnodes4item = num_targets(xBFaceNodes,bface)
        for k = 1 : nnodes4item
            dofs4item[k] = xBFaceNodes[k,bface]
            for n = 1 : ncomponents-1
                dofs4item[k+n*nnodes4item] = n*nnodes + dofs4item[k]
            end    
        end
        dofs4item[ncomponents*nnodes4item+1] = ncomponents*nnodes + xBFaces[bface]
        append!(xBFaceDofs,dofs4item[1:ncomponents*nnodes4item+1])
    end
    ndofs = nnodes * ncomponents + nfaces
    return FEH1BR{ncomponents}(name,xgrid,xCellDofs,xFaceDofs,xBFaceDofs,xFaceNormals,xFaceVolumes,xCellFaces,ndofs)
end


get_ncomponents(::Type{<:FEH1BR{2}}) = 2
get_ncomponents(::Type{<:FEH1BR{3}}) = 3

get_polynomialorder(::Type{<:FEH1BR{2}}, ::Type{<:Edge1D}) = 2;
get_polynomialorder(::Type{<:FEH1BR{2}}, ::Type{<:Triangle2D}) = 2;
get_polynomialorder(::Type{<:FEH1BR{2}}, ::Type{<:Quadrilateral2D}) = 3;


function interpolate!(Target::AbstractArray{<:Real,1}, FE::FEH1BR{2}, exact_function!::Function; dofs = [], bonus_quadorder::Int = 0)
    xCoords = FE.xgrid[Coordinates]
    xFaceNodes = FE.xgrid[FaceNodes]
    xFaceNormals = FE.xgrid[FaceNormals]
    ncomponents = get_ncomponents(typeof(FE))
    result = zeros(Float64,ncomponents)
    xdim = size(xCoords,1)
    nnodes = num_sources(xCoords)
    nfaces = num_sources(xFaceNodes)
    x = zeros(Float64,xdim)
    if length(dofs) == 0 # interpolate at all dofs
        for j = 1 : num_sources(xCoords)
            for k=1:xdim
                x[k] = xCoords[k,j]
            end    
            exact_function!(result,x)
            for n = 1 : ncomponents
                Target[j+(n-1)*nnodes] = result[n]
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
            Target[ncomponents*nnodes+face] = 0.0
            for k = 1 : ncomponents
                Target[ncomponents*nnodes+face] = result[k]*xFaceNormals[k,face]
            end    
        end
    else
        item = 0
        for j in dofs 
            item = mod(j-1,nnodes)+1
            c = Int(ceil(j/nnodes))
            if item <= nnodes
                for k=1:xdim
                    x[k] = xCoords[k,item]
                end    
                exact_function!(result,x)
                Target[j] = result[c]
            elseif c > ncomponents
                face = j - ncomponents*nnodes
                nnodes4item = num_targets(xFaceNodes,face)
                for k=1:xdim
                    x[k] = 0
                    for n=1:nnodes4item
                        x[k] += xCoords[k,xFaceNodes[n,face]]
                    end
                    x[k] /= nnodes4item    
                end 
                exact_function!(result,x)
                Target[ncomponents*nnodes+face] = 0.0
                for k = 1 : ncomponents
                    Target[ncomponents*nnodes+face] = result[k]*xFaceNormals[k,face]
                end    
            end
        end    
    end    
end

function nodevalues!(Target::AbstractArray{<:Real,2}, Source::AbstractArray{<:Real,1}, FE::FEH1BR)
    nnodes = num_sources(FE.xgrid[Coordinates])
    nfaces = num_sources(FE.xgrid[FaceNodes])
    ncomponents = get_ncomponents(typeof(FE))
    offset4component = 0:nnodes:ncomponents*nnodes
    for node = 1 : nnodes
        for c = 1 : ncomponents
            Target[c,node] = Source[offset4component[c]+node]
        end    
    end    
end


function get_basis_on_face(::Type{FEH1BR{2}}, ::Type{<:Edge1D})
    function closure(xref)
        temp = 1 - xref[1];
        bf = 4 * xref[1] * temp;
        return [temp 0.0;
                xref[1] 0.0;
                0.0 temp;
                0.0 xref[1];
                bf bf]
    end
end

function get_basis_on_cell(::Type{FEH1BR{2}}, ::Type{<:Triangle2D})
    function closure(xref)
        temp = 1 - xref[1] - xref[2];
        bf1 = 4 * xref[1] * temp;
        bf2 = 4 * xref[2] * xref[1];
        bf3 = 4 * temp * xref[2];
        return [temp 0.0;
                xref[1] 0.0;
                xref[2] 0.0;
                0.0 temp;
                0.0 xref[1];
                0.0 xref[2];
                bf1 bf1;
                bf2 bf2;
                bf3 bf3]
    end
end


function get_basis_on_cell(::Type{FEH1BR{2}}, ::Type{<:Quadrilateral2D})
    function closure(xref)
        a = 1 - xref[1]
        b = 1 - xref[2]
        fb1 = 4*xref[1]*a*b
        fb2 = 4*xref[2]*xref[1]*b
        fb3 = 4*xref[1]*xref[2]*a
        fb4 = 4*xref[2]*a*b
        return [a*b 0.0;
                xref[1]*b 0.0;
                xref[1]*xref[2] 0.0;
                xref[2]*a 0.0;
                0.0 a*b;
                0.0 xref[1]*b;
                0.0 xref[1]*xref[2];
                0.0 xref[2]*a;
                fb1 fb1;
                fb2 fb2;
                fb3 fb3;
                fb4 fb4]
    end
end


function get_coefficients_on_cell!(coefficients, FE::FEH1BR{2}, ::Type{<:Triangle2D}, cell::Int)
    # multiplication with normal vectors
    fill!(coefficients,1.0)
    coefficients[1,7] = FE.xFaceNormals[1, FE.xCellFaces[1,cell]];
    coefficients[2,7] = FE.xFaceNormals[2, FE.xCellFaces[1,cell]];
    coefficients[1,8] = FE.xFaceNormals[1, FE.xCellFaces[2,cell]];
    coefficients[2,8] = FE.xFaceNormals[2, FE.xCellFaces[2,cell]];
    coefficients[1,9] = FE.xFaceNormals[1, FE.xCellFaces[3,cell]];
    coefficients[2,9] = FE.xFaceNormals[2, FE.xCellFaces[3,cell]];
end    
function get_coefficients_on_cell!(coefficients, FE::FEH1BR{2}, ::Type{<:Quadrilateral2D}, cell::Int,)
    # multiplication with normal vectors
    fill!(coefficients,1.0)
    coefficients[1,9] = FE.xFaceNormals[1, FE.xCellFaces[1,cell]];
    coefficients[2,9] = FE.xFaceNormals[2, FE.xCellFaces[1,cell]];
    coefficients[1,10] = FE.xFaceNormals[1, FE.xCellFaces[2,cell]];
    coefficients[2,10] = FE.xFaceNormals[2, FE.xCellFaces[2,cell]];
    coefficients[1,11] = FE.xFaceNormals[1, FE.xCellFaces[3,cell]];
    coefficients[2,11] = FE.xFaceNormals[2, FE.xCellFaces[3,cell]];
    coefficients[1,12] = FE.xFaceNormals[1, FE.xCellFaces[4,cell]];
    coefficients[2,12] = FE.xFaceNormals[2, FE.xCellFaces[4,cell]];
end    


function get_coefficients_on_face!(coefficients, FE::FEH1BR{2}, ::Type{<:Edge1D}, face::Int)
    # multiplication with normal vectors
    fill!(coefficients,1.0)
    coefficients[1,5] = FE.xFaceNormals[1, face];
    coefficients[2,5] = FE.xFaceNormals[2, face];
end    


function get_reconstruction_coefficients_on_cell!(coefficients, FE::FEH1BR{2}, FEreconst::FEHdivRT0, ::Type{<:Triangle2D}, cell::Int)
    # reconstruction coefficients for P1 basis functions on reference element
    fill!(coefficients,0.0)
    coefficients[1,3] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[3,cell]] * FE.xFaceNormals[1, FE.xCellFaces[3,cell]]
    coefficients[1,1] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[1,cell]] * FE.xFaceNormals[1, FE.xCellFaces[1,cell]]
    coefficients[2,1] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[1,cell]] * FE.xFaceNormals[1, FE.xCellFaces[1,cell]]
    coefficients[2,2] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[2,cell]] * FE.xFaceNormals[1, FE.xCellFaces[2,cell]]
    coefficients[3,2] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[2,cell]] * FE.xFaceNormals[1, FE.xCellFaces[2,cell]]
    coefficients[3,3] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[3,cell]] * FE.xFaceNormals[1, FE.xCellFaces[3,cell]]
    coefficients[4,3] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[3,cell]] * FE.xFaceNormals[2, FE.xCellFaces[3,cell]]
    coefficients[4,1] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[1,cell]] * FE.xFaceNormals[2, FE.xCellFaces[1,cell]]
    coefficients[5,1] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[1,cell]] * FE.xFaceNormals[2, FE.xCellFaces[1,cell]]
    coefficients[5,2] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[2,cell]] * FE.xFaceNormals[2, FE.xCellFaces[2,cell]]
    coefficients[6,2] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[2,cell]] * FE.xFaceNormals[2, FE.xCellFaces[2,cell]]
    coefficients[6,3] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[3,cell]] * FE.xFaceNormals[2, FE.xCellFaces[3,cell]]
    coefficients[7,1] = 2 // 3 * FE.xFaceVolumes[FE.xCellFaces[1,cell]]
    coefficients[8,2] = 2 // 3 * FE.xFaceVolumes[FE.xCellFaces[2,cell]]
    coefficients[9,3] = 2 // 3 * FE.xFaceVolumes[FE.xCellFaces[3,cell]]
end


function get_reconstruction_coefficients_on_cell!(coefficients, FE::FEH1BR{2}, FEreconst::FEHdivRT0, ::Type{<:Parallelogram2D}, cell::Int)
    # reconstruction coefficients for P1 basis functions on reference element
    fill!(coefficients,0.0)
    coefficients[1,4] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[4,cell]] * FE.xFaceNormals[1, FE.xCellFaces[4,cell]]
    coefficients[1,1] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[1,cell]] * FE.xFaceNormals[1, FE.xCellFaces[1,cell]]
    coefficients[2,1] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[1,cell]] * FE.xFaceNormals[1, FE.xCellFaces[1,cell]]
    coefficients[2,2] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[2,cell]] * FE.xFaceNormals[1, FE.xCellFaces[2,cell]]
    coefficients[3,2] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[2,cell]] * FE.xFaceNormals[1, FE.xCellFaces[2,cell]]
    coefficients[3,3] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[3,cell]] * FE.xFaceNormals[1, FE.xCellFaces[3,cell]]
    coefficients[4,3] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[3,cell]] * FE.xFaceNormals[1, FE.xCellFaces[3,cell]]
    coefficients[4,4] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[4,cell]] * FE.xFaceNormals[1, FE.xCellFaces[4,cell]]
    coefficients[5,4] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[4,cell]] * FE.xFaceNormals[2, FE.xCellFaces[4,cell]]
    coefficients[5,1] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[1,cell]] * FE.xFaceNormals[2, FE.xCellFaces[1,cell]]
    coefficients[6,1] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[1,cell]] * FE.xFaceNormals[2, FE.xCellFaces[1,cell]]
    coefficients[6,2] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[2,cell]] * FE.xFaceNormals[2, FE.xCellFaces[2,cell]]
    coefficients[7,2] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[2,cell]] * FE.xFaceNormals[2, FE.xCellFaces[2,cell]]
    coefficients[7,3] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[3,cell]] * FE.xFaceNormals[2, FE.xCellFaces[3,cell]]
    coefficients[8,3] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[3,cell]] * FE.xFaceNormals[2, FE.xCellFaces[3,cell]]
    coefficients[8,4] = 1 // 2 * FE.xFaceVolumes[FE.xCellFaces[4,cell]] * FE.xFaceNormals[2, FE.xCellFaces[4,cell]]
    coefficients[9,1] = 2 // 3 * FE.xFaceVolumes[FE.xCellFaces[1,cell]]
    coefficients[10,2] = 2 // 3 * FE.xFaceVolumes[FE.xCellFaces[2,cell]]
    coefficients[11,3] = 2 // 3 * FE.xFaceVolumes[FE.xCellFaces[3,cell]]
    coefficients[12,4] = 2 // 3 * FE.xFaceVolumes[FE.xCellFaces[4,cell]]
end