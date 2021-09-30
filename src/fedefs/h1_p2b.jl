
"""
````
abstract type H1P2B{ncomponents,edim} <: AbstractH1FiniteElement where {ncomponents<:Int,edim<:Int}
````

Continuous piecewise second-order polynomials.

allowed ElementGeometries:
- Triangle2D
"""
abstract type H1P2B{ncomponents,edim} <: AbstractH1FiniteElement where {ncomponents<:Int,edim<:Int} end

function Base.show(io::Core.IO, ::Type{<:H1P2B{ncomponents,edim}}) where {ncomponents,edim}
    print(io,"H1P2B{$ncomponents,$edim}")
end

get_ncomponents(FEType::Type{<:H1P2B}) = FEType.parameters[1]
get_edim(FEType::Type{<:H1P2B}) = FEType.parameters[2]

get_ndofs(::Type{ON_CELLS}, FEType::Type{<:H1P2B}, EG::Type{<:Triangle2D}) = 7*FEType.parameters[1]
get_ndofs(::Union{Type{<:ON_FACES}, Type{<:ON_BFACES}}, FEType::Type{<:H1P2B}, EG::Type{<:AbstractElementGeometry1D}) = 3*FEType.parameters[1]

get_polynomialorder(::Type{<:H1P2B}, ::Type{<:Edge1D}) = 2;
get_polynomialorder(::Type{<:H1P2B}, ::Type{<:Triangle2D}) = 3;
get_polynomialorder(::Type{<:H1P2B}, ::Type{<:Tetrahedron3D}) = 4;

get_dofmap_pattern(FEType::Type{<:H1P2B}, ::Type{CellDofs}, EG::Type{<:AbstractElementGeometry2D}) = "N1F1I1"
get_dofmap_pattern(FEType::Type{<:H1P2B}, ::Type{FaceDofs}, EG::Type{<:AbstractElementGeometry1D}) = "N1I1C1"
get_dofmap_pattern(FEType::Type{<:H1P2B}, ::Type{BFaceDofs}, EG::Type{<:AbstractElementGeometry1D}) = "N1I1C1"

isdefined(FEType::Type{<:H1P2B}, ::Type{<:Triangle2D}) = true

get_ref_cellmoments(::Type{<:H1P2B}, ::Type{<:Triangle2D}) = [0//1, 0//1, 0//1, 1//3, 1//3, 1//3, 1//1] # integrals of 1D basis functions over reference cell (divided by volume)

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{AT_NODES}, exact_function!; items = [], time = 0) where {FEType <: H1P2B}
    edim = get_edim(FEType)
    nnodes = size(FE.xgrid[Coordinates],2)
    offset = nnodes + num_sources(FE.xgrid[CellNodes])
    if edim == 2
        offset += num_sources(FE.xgrid[FaceNodes])
    elseif edim == 3
        offset += num_sources(FE.xgrid[EdgeNodes])
    end

    point_evaluation!(Target, FE, AT_NODES, exact_function!; items = items, component_offset = offset, time = time)
end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{ON_EDGES}, exact_function!; items = [], time = 0) where {FEType <: H1P2B}
    edim = get_edim(FEType)
    if edim == 3
        # delegate edge nodes to node interpolation
        subitems = slice(FE.xgrid[EdgeNodes], items)
        interpolate!(Target, FE, AT_NODES, exact_function!; items = subitems, time = time)

        # perform edge mean interpolation
        ensure_edge_moments!(Target, FE, ON_EDGES, exact_function!; items = items, time = time)
    end
end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{ON_FACES}, exact_function!; items = [], time = 0) where {FEType <: H1P2B}
    edim = get_edim(FEType)
    if edim == 2
        # delegate face nodes to node interpolation
        subitems = slice(FE.xgrid[FaceNodes], items)
        interpolate!(Target, FE, AT_NODES, exact_function!; items = subitems, time = time)

        # perform face mean interpolation
        ensure_edge_moments!(Target, FE, ON_FACES, exact_function!; items = items, time = time)
    elseif edim == 3
        # delegate face edges to edge interpolation
        subitems = slice(FE.xgrid[FaceEdges], items)
        interpolate!(Target, FE, ON_EDGES, exact_function!; items = subitems, time = time)

        # perform face mean interpolation
        # todo
    elseif edim == 1
        # delegate face nodes to node interpolation
        subitems = slice(FE.xgrid[FaceNodes], items)
        interpolate!(Target, FE, AT_NODES, exact_function!; items = subitems, time = time)
    end
end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{ON_CELLS}, exact_function!; items = [], time = 0) where {FEType <: H1P2B}
    edim = get_edim(FEType)
    if edim == 2
        # delegate cell faces to face interpolation
        subitems = slice(FE.xgrid[CellFaces], items)
        interpolate!(Target, FE, ON_FACES, exact_function!; items = subitems, time = time)
    elseif edim == 3
        # delegate cell edges to edge interpolation
        subitems = slice(FE.xgrid[CellEdges], items)
        interpolate!(Target, FE, ON_EDGES, exact_function!; items = subitems, time = time)
    elseif edim == 1
        # delegate cell nodes to node interpolation
        subitems = slice(FE.xgrid[CellNodes], items)
        interpolate!(Target, FE, AT_NODES, exact_function!; items = subitems, time = time)
    end

    # fix cell bubble value by preserving integral mean
    ensure_cell_moments!(Target, FE, exact_function!; facedofs = 1, edgedofs = 0, items = items, time = time)
end

function get_basis(AT::Union{Type{<:ON_FACES}, Type{<:ON_BFACES}}, FEType::Type{<:H1P2B}, EG::Type{<:AbstractElementGeometry})
    # on faces same as P2
    return get_basis(AT, H1P2{get_ncomponents(FEType), get_edim(FEType)}, EG)
end

function get_basis(AT::Type{ON_CELLS}, FEType::Type{<:H1P2B}, EG::Type{<:Triangle2D})
    ncomponents = get_ncomponents(FEType)
    edim = get_edim(FEType)
    refbasis_P2 = get_basis(AT, H1P2{1,edim}, EG)
    offset = get_ndofs(AT, H1P2{1,edim}, EG) + 1
    function closure(refbasis, xref)
        refbasis_P2(refbasis, xref)
        # add cell bubbles to P2 basis
        refbasis[offset,1] = 60*(1-xref[1]-xref[2])*xref[1]*xref[2]
        for k = 1 : ncomponents-1, j = 1 : offset
            refbasis[k*offset+j,k+1] = refbasis[j,1]
        end
    end
end




function get_reconstruction_coefficients!(xgrid, ::Type{ON_CELLS}, FE::Type{<:H1P2B{2,2}}, FER::Type{<:HDIVRT1{2}}, EG::Type{<:Triangle2D})
    xFaceVolumes::Array{Float64,1} = xgrid[FaceVolumes]
    xFaceNormals::Array{Float64,2} = xgrid[FaceNormals]
    xCellFaceSigns::Union{VariableTargetAdjacency{Int32},Array{Int32,2}} = xgrid[CellFaceSigns]
    xCellFaces::Union{VariableTargetAdjacency{Int32},Array{Int32,2}} = xgrid[CellFaces]
    #xCellNodes::Union{VariableTargetAdjacency{Int32},Array{Int32,2}} = xgrid[CellNodes]
    #xCellVolumes::Array{Float64,1} = xgrid[CellVolumes]
    face_rule::Array{Int,2} = face_enum_rule(EG)
    node::Int = 0
    face::Int = 0
    nnf::Int = size(face_rule,1)
    ndofs4component::Int = 2*nnf + 1
    RT1_coeffs::Array{Float64,1} = [-1//12, 1//12]
    #xCoordinates = FE.xgrid[Coordinates]
    #C = zeros(Float64,2,3)  # vertices
    #E = zeros(Float64,2,3)  # edge midpoints
    #M = zeros(Float64,2)    # midpoint of current cell
    #A = zeros(Float64,2,8)  # integral means of RT 1 functions
    #b = zeros(Float64,2)    # right-hand side for integral mean
    #id::Int = 0
    #det::Float64 = 0
    function closure(coefficients::Array{<:Real,2}, cell::Int)
        # fill!(coefficients,0.0)

        # get coordinates of cells
        #fill!(M,0)
        #fill!(E,0)
        #for n = 1 : 3, k = 1 : 2
        #    C[k,n] = xCoordinates[k,xCellNodes[n,cell]]
        #    M[k] += C[k,n] / 3
        #end
        for f = 1 : nnf
            face = xCellFaces[f,cell]
            #for n = 1 : 2, k = 1 : 2
            #    E[k,f] += C[k,face_rule[f,n]] / 2
            #end
            for n = 1 : 2
                node = face_rule[f,n]
                for k = 1 : 2
                    # RT0 reconstruction coefficients for node P2 functions on reference element
                    coefficients[ndofs4component*(k-1)+node,2*(f-1)+1] = 1 // 6 * xFaceVolumes[face] * xFaceNormals[k, face]

                    # RT1 reconstruction coefficients for node P2 functions on reference element
                    coefficients[ndofs4component*(k-1)+node,2*(f-1)+2] = RT1_coeffs[n] * xFaceVolumes[face] * xFaceNormals[k, face] * xCellFaceSigns[f, cell]
                end
            end

            for k = 1 : 2
                # RT0 reconstruction coefficients for face P2 functions (=face bubbles) on reference element
                coefficients[ndofs4component*(k-1)+f+nnf,2*(f-1)+1] = 2 // 3 * xFaceVolumes[face] * xFaceNormals[k, face]
            end
        end

        # # compute integral means of RT1 functions
        # for k = 1 : 2
        #     A[k,1] = (M[k] - C[k,3])/2 * xCellFaceSigns[1, cell]
        #     A[k,2] = C[k,2] - E[k,2]
        #     A[k,3] = (M[k] - C[k,1])/2 * xCellFaceSigns[2, cell]
        #     A[k,4] = C[k,3] - E[k,3]
        #     A[k,5] = (M[k] - C[k,2])/2 * xCellFaceSigns[3, cell]
        #     A[k,6] = C[k,1] - E[k,1]
        # end
        # # directly assign inverted A[1:2,7:8] for faster solve of local systems
        # A[2,8] = (E[1,1] - C[1,3]) # A[1,7]
        # A[2,7] = -(E[2,1] - C[2,3]) # A[2,7]
        # A[1,8] = -(E[1,3] - C[1,2]) # A[1,8]
        # A[1,7] = (E[2,3] - C[2,2]) # A[2,8]

        # det = A[1,7]*A[2,8] - A[2,7]*A[1,8]
        # A[1:2,7:8] ./= det

        # # correct integral means with interior RT1 functions
        # for k = 1 : 2
        #     for n = 1 : 3
        #         # nodal P2 functions have integral mean zero
        #         id = ndofs4component*(k-1) + n
        #         fill!(b,0)
        #         for c = 1 : 2, j = 1 : 6
        #             b[c] -= coefficients[id,j] * A[c,j]
        #         end
        #         coefficients[id,7:8] = A[:,7:8]*b

        #         # face P2 functions have integral mean 1//3
        #         id = ndofs4component*(k-1) + n + nnf
        #         fill!(b,0)
        #         b[k] = xCellVolumes[cell] / 3
        #         for c = 1 : 2, j = 1 : 6
        #             b[c] -= coefficients[id,j] * A[c,j]
        #         end
        #         coefficients[id,7:8] = A[:,7:8]*b
        #     end

        #     # cell bubbles have integral mean 1
        #     id = ndofs4component*k
        #     fill!(b,0)
        #     b[k] = xCellVolumes[cell]
        #     coefficients[id,7:8] = A[:,7:8]*b
        # end
    
        return nothing
    end
end



function get_reconstruction_coefficients!(xgrid, ::Type{ON_CELLS}, FE::Type{<:H1P2B{2,2}}, FER::Type{<:HDIVBDM2{2}}, EG::Type{<:Triangle2D})
    xFaceVolumes::Array{Float64,1} = xgrid[FaceVolumes]
    xFaceNormals::Array{Float64,2} = xgrid[FaceNormals]
    xCellFaceSigns::Union{VariableTargetAdjacency{Int32},Array{Int32,2}} = xgrid[CellFaceSigns]
    xCellFaces = xgrid[CellFaces]
    face_rule::Array{Int,2} = face_enum_rule(EG)
    node::Int = 0
    face::Array{Int,1} = [0] # <-- seems to avoid allocations in line 247
    nnf::Int = size(face_rule,1)
    ndofs4component::Int = 2*nnf + 1
    coeffs1::Array{Float64,1} = [-1//12, 1//12]
    function closure(coefficients::Array{<:Real,2}, cell::Int)
        # fill!(coefficients,0.0)

        for f = 1 : nnf
            face[1] = xCellFaces[f,cell]
            for n = 1 : 2
                node = face_rule[f,n]
                for k = 1 : 2
                    # RT0 reconstruction coefficients for node P2 functions on reference element
                    coefficients[ndofs4component*(k-1)+node,3*(f-1)+1] = 1 // 6 * xFaceVolumes[face[1]] * xFaceNormals[k, face[1]]

                    # 1st BDM2 reconstruction coefficients for node P2 functions on reference element
                    coefficients[ndofs4component*(k-1)+node,3*(f-1)+2] = coeffs1[n] * xFaceVolumes[face[1]] * xFaceNormals[k, face[1]] * xCellFaceSigns[f, cell]

                    # 2nd BDM2 reconstruction coefficients for node P2 functions on reference element
                    coefficients[ndofs4component*(k-1)+node,3*(f-1)+3] = 1 // 90 * xFaceVolumes[face[1]] * xFaceNormals[k, face[1]]
                end
            end

            for k = 1 : 2
                # RT0 reconstruction coefficients for face P2 functions (=face bubbles) on reference element
                coefficients[ndofs4component*(k-1)+f+nnf,3*(f-1)+1] = 2 // 3 * xFaceVolumes[face[1]] * xFaceNormals[k, face[1]]

                # 2nd BDM2 reconstruction coefficients for face P2 functions on reference element
                coefficients[ndofs4component*(k-1)+f+nnf,3*(f-1)+3] = -1 // 45 * xFaceVolumes[face[1]] * xFaceNormals[k, face[1]]
            end
        end

        return nothing
    end
end