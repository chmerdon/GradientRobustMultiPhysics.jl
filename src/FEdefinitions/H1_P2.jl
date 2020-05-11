struct FEH1P2{ncomponents} <: AbstractH1FiniteElement where {ncomponents<:Int}
    name::String                         # full name of finite element (used in messages)
    xgrid::ExtendableGrid                # link to xgrid 
    CellDofs::VariableTargetAdjacency    # place to save cell dofs (filled by constructor)
    FaceDofs::VariableTargetAdjacency    # place to save face dofs (filled by constructor)
    BFaceDofs::VariableTargetAdjacency   # place to save bface dofs (filled by constructor)
    ndofs::Int32
end

function getH1P2FiniteElement(xgrid::ExtendableGrid, ncomponents::Int)
    name = "P2"
    for n = 1 : ncomponents-1
        name = name * "xP2"
    end
    name = name * " (H1)"    

    # generate celldofs
    dim = size(xgrid[Coordinates],1) 
    xCellNodes = xgrid[CellNodes]
    xCellFaces = xgrid[CellFaces]
    xFaceNodes = xgrid[FaceNodes]
    xCellGeometries = xgrid[CellGeometries]
    xBFaceNodes = xgrid[BFaceNodes]
    xBFaces = xgrid[BFaces]
    ncells = num_sources(xCellNodes)
    nfaces = num_sources(xFaceNodes)
    nbfaces = num_sources(xBFaceNodes)
    nnodes = num_sources(xgrid[Coordinates])
    ndofs4component = nnodes + nfaces

    # generate dofmaps
    xCellDofs = VariableTargetAdjacency(Int32)
    xFaceDofs = VariableTargetAdjacency(Int32)
    xBFaceDofs = VariableTargetAdjacency(Int32)
    dofs4item = zeros(Int32,ncomponents*(max_num_targets_per_source(xCellNodes)+max_num_targets_per_source(xCellFaces)))
    nnodes4item = 0
    nextra4item = 0
    ndofs4item = 0
    edim = 0
    for cell = 1 : ncells
        edim = dim_element(xCellGeometries[cell])
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

    # total number of dofs
    ndofs = 0
    if edim == 1
        ndofs = (nnodes + ncells) * ncomponents
    elseif edim == 2
        ndofs = (nnodes + nfaces) * ncomponents
    elseif edim == 3
        # TODO
    end    
    return FEH1P2{ncomponents}(name,xgrid,xCellDofs,xFaceDofs,xBFaceDofs,ndofs)
end


get_ncomponents(::Type{FEH1P2{1}}) = 1
get_ncomponents(::Type{FEH1P2{2}}) = 2

get_polynomialorder(::Type{<:FEH1P2}, ::Type{<:Edge1D}) = 2;
get_polynomialorder(::Type{<:FEH1P2}, ::Type{<:Triangle2D}) = 2;
get_polynomialorder(::Type{<:FEH1P2}, ::Type{<:Quadrilateral2D}) = 3;


function interpolate!(Target::AbstractArray{<:Real,1}, FE::FEH1P2, exact_function!::Function; dofs = [])
    xCoords = FE.xgrid[Coordinates]
    xFaceNodes = FE.xgrid[FaceNodes]
    nnodes = num_sources(xCoords)
    nfaces = num_sources(xFaceNodes)
    ncomponents = get_ncomponents(typeof(FE))

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

function get_basis_on_cell(::Type{FEH1P2{1}}, ::Type{<:Edge1D})
    function closure(xref)
        temp = 1 - xref[1];
        return [2*temp*(temp - 1//2),
                2*xref[1]*(xref[1] - 1//2),
                4*temp*xref[1]]
    end
end


function get_basis_on_cell(::Type{FEH1P2{2}}, ::Type{<:Edge1D})
    function closure(xref)
        temp = 1 - xref[1]
        a = 2*temp*(temp - 1//2)
        b = 2*xref[1]*(xref[1] - 1//2)
        c = 4*temp*xref[1]
        return [a 0.0;
                b 0.0;
                c 0.0;
                0.0 a;
                0.0 b;
                0.0 c]
    end
end

function get_basis_on_cell(::Type{FEH1P2{1}}, ::Type{<:Triangle2D})
    function closure(xref)
        temp = 1 - xref[1] - xref[2]
        return [2*temp*(temp - 1//2);
                2*xref[1]*(xref[1] - 1//2);
                2*xref[2]*(xref[2] - 1//2);
                4*temp*xref[1];
                4*xref[1]*xref[2];
                4*xref[2]*temp]
    end
end

function get_basis_on_cell(::Type{FEH1P2{2}}, ::Type{<:Triangle2D})
    function closure(xref)
        temp = 1 - xref[1] - xref[2];
        a = 2*temp*(temp - 1//2);
        b = 2*xref[1]*(xref[1] - 1//2);
        c = 2*xref[2]*(xref[2] - 1//2);
        d = 4*temp*xref[1];
        e = 4*xref[1]*xref[2];
        f = 4*temp*xref[2];
        return [a 0.0;    
                b 0.0;
                c 0.0;
                d 0.0;
                e 0.0;
                f 0.0;
                0.0 a;
                0.0 b;
                0.0 c;
                0.0 d;
                0.0 e;
                0.0 f]
    end
end

function get_basis_on_cell(::Type{FEH1P2{1}}, ::Type{<:Quadrilateral2D})
    function closure(xref)
        a = 1 - xref[1]
        b = 1 - xref[2]
        return [-2*a*b*(xref[1]+xref[2]-1//2);
                -2*xref[1]*b*(xref[2]-xref[1]+1//2);
                2*xref[1]*xref[2]*(xref[1]+xref[2]-3//2);
                -2*xref[2]*a*(xref[1]-xref[2]+1//2);
                4*xref[1]*a*b
                4*xref[2]*xref[1]*b
                4*xref[1]*xref[2]*a
                4*xref[2]*a*b
                ]
    end
end

function get_basis_on_cell(::Type{FEH1P2{2}}, ::Type{<:Quadrilateral2D})
    function closure(xref)
        a = 1 - xref[1]
        b = 1 - xref[2]
        c = 2*xref[1]*xref[2]*(xref[1]+xref[2]-3//2);
        d = -2*xref[2]*a*(xref[1]-xref[2]+1//2);
        e = 4*xref[1]*a*b
        f = 4*xref[2]*xref[1]*b
        g = 4*xref[1]*xref[2]*a
        h = 4*xref[2]*a*b
        a = -2*a*b*(xref[1]+xref[2]-1//2);
        b = -2*xref[1]*b*(xref[2]-xref[1]+1//2);
        return [a 0.0;    
                b 0.0;
                c 0.0;
                d 0.0;
                e 0.0;
                f 0.0;
                g 0.0;
                h 0.0;
                0.0 a;
                0.0 b;
                0.0 c;
                0.0 d;
                0.0 e;
                0.0 f;
                0.0 g;
                0.0 h]
    end
end

function get_basis_on_face(FE::Type{<:FEH1P2}, EG::Type{<:AbstractElementGeometry})
    function closure(xref)
        return get_basis_on_cell(FE, EG)(xref[1:end-1])
    end    
end
