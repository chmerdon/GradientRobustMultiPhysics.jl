
abstract type ReconstructionCoefficients{FE1,FE2,AT} <: AbstractGridFloatArray2D end
abstract type ReconstructionDofs{FE1,FE2,AT} <: AbstractGridIntegerArray2D end

struct ReconstructionHandler{FE1,FE2,AT,EG, FType <: Function}
    FES::FESpace{FE1,AT}
    FER::FESpace{FE2,AT}
    interior_offset::Int
    interior_ndofs::Int
    interior_coefficients::Matrix{Float64} # coefficients for interior basis functions are precomputed
    boundary_coefficients::FType # coefficient on boundary are recalculated every time
end

function ReconstructionHandler(FES,FES_Reconst,AT,EG)
    FE1 = typeof(FES).parameters[1]
    FE2 = typeof(FES_Reconst).parameters[1]
    xgrid = FES.xgrid
    interior_offset = interior_dofs_offset(AT,FE2,EG)
    interior_ndofs = get_ndofs(AT,FE2,EG) - interior_offset
    if interior_ndofs > 0
        coeffs = xgrid[ReconstructionCoefficients{FE1,FE2,AT}]
    else
        coeffs = zeros(Float64,0,0)
    end
    rcoeff_handler = get_reconstruction_coefficients!(xgrid, AT, FE1, FE2, EG)
    return ReconstructionHandler{FE1,FE2,AT,EG, typeof(rcoeff_handler)}(FES,FES_Reconst,interior_offset,interior_ndofs,coeffs,rcoeff_handler)
end

function get_rcoefficients!(coefficients, RH::ReconstructionHandler{FE1,FE2,AT,EG}, item) where {FE1,FE2,AT,EG}
    RH.boundary_coefficients(coefficients, item)
    for dof = 1 : size(coefficients,1), k = 1 : RH.interior_ndofs
        coefficients[dof,RH.interior_offset + k] = RH.interior_coefficients[(dof-1)*RH.interior_ndofs+k, item]
    end
    return nothing
end

# P2B > RT1/BDM2 reconstruction
interior_dofs_offset(AT::Type{<:AbstractAssemblyType}, FE::Type{<:AbstractFiniteElement}, EG::Type{<:AbstractElementGeometry}) = get_ndofs(AT,FE,EG)
interior_dofs_offset(::Type{<:ON_CELLS}, ::Type{<:HDIVRT1{2}}, ::Type{<:Triangle2D}) = 6
interior_dofs_offset(::Type{<:ON_CELLS}, ::Type{<:HDIVBDM2{2}}, ::Type{<:Triangle2D}) = 9

function ExtendableGrids.instantiate(xgrid::ExtendableGrid, ::Type{ReconstructionCoefficients{FE1,FE2,AT}}) where {FE1<:H1P2B{2,2}, FE2<:HDIVRT1{2}, AT <: ON_CELLS}
    @info "Computing interior reconstruction coefficients for $FE1 > $FE2 ($AT)"
    xCellFaces = xgrid[CellFaces]
    xCoordinates = xgrid[Coordinates]
    xCellNodes = xgrid[CellNodes]
    xCellVolumes::Array{Float64,1} = xgrid[CellVolumes]
    xCellFaceSigns = xgrid[CellFaceSigns]
    EG = xgrid[UniqueCellGeometries]

    @assert EG == [Triangle2D]

    face_rule::Array{Int,2} = face_enum_rule(EG[1])
    nnf::Int = size(face_rule,1)
    ndofs4component::Int = 2*nnf + 1
    ndofs1::Int = get_ndofs(AT,FE1,EG[1])
    ncells = num_sources(xCellFaces)
    rcoeff_handler! = get_reconstruction_coefficients!(xgrid, AT, FE1, FE2, EG[1])
    coefficients = zeros(Float64,ndofs1,6)
    interior_coefficients = zeros(Float64,2*ndofs1,ncells)

    C = zeros(Float64,2,3)  # vertices
    E = zeros(Float64,2,3)  # edge midpoints
    M = zeros(Float64,2)    # midpoint of current cell
    A = zeros(Float64,2,8)  # integral means of RT1 functions (from analytic formulas)
    b = zeros(Float64,2)    # right-hand side for integral mean
    dof::Int = 0
    det::Float64 = 0
    for cell = 1 : ncells

        # get reconstruction coefficients for boundary dofs
        rcoeff_handler!(coefficients, cell)

        # get coordinates of cells
        fill!(M,0)
        fill!(E,0)
        for n = 1 : 3, k = 1 : 2
            C[k,n] = xCoordinates[k,xCellNodes[n,cell]]
            M[k] += C[k,n] / 3
        end
        
        # get edge midpoints
        for f = 1 : size(face_rule,1)
            for n = 1 : 2, k = 1 : 2
                E[k,f] += C[k,face_rule[f,n]] / 2
            end
        end

        # compute integral means of RT1 functions
        for k = 1 : 2
            A[k,1] = (M[k] - C[k,3])/2 * xCellFaceSigns[1, cell]
            A[k,2] = C[k,2] - E[k,2]
            A[k,3] = (M[k] - C[k,1])/2 * xCellFaceSigns[2, cell]
            A[k,4] = C[k,3] - E[k,3]
            A[k,5] = (M[k] - C[k,2])/2 * xCellFaceSigns[3, cell]
            A[k,6] = C[k,1] - E[k,1]
        end
        # directly assign inverted A[1:2,7:8] for faster solve of local systems
        A[2,8] = (E[1,1] - C[1,3]) # A[1,7]
        A[2,7] = -(E[2,1] - C[2,3]) # A[2,7]
        A[1,8] = -(E[1,3] - C[1,2]) # A[1,8]
        A[1,7] = (E[2,3] - C[2,2]) # A[2,8]

        det = A[1,7]*A[2,8] - A[2,7]*A[1,8]
        A[1:2,7:8] ./= det

        # correct integral means with interior RT1 functions
        for k = 1 : 2
            for n = 1 : 3
                # nodal P2 functions have integral mean zero
                dof = (ndofs4component*(k-1) + n)
                fill!(b,0)
                for c = 1 : 2, j = 1 : 6
                    b[c] -= coefficients[dof,j] * A[c,j]
                end
                for k = 1 : 2
                    interior_coefficients[(dof-1)*2+k,cell] = A[k,7]*b[1] + A[k,8]*b[2]
                end

                # face P2 functions have integral mean 1//3
                dof = (ndofs4component*(k-1) + n + nnf)
                fill!(b,0)
                b[k] = xCellVolumes[cell] / 3
                for c = 1 : 2, j = 1 : 6
                    b[c] -= coefficients[dof,j] * A[c,j]
                end
                for k = 1 : 2
                    interior_coefficients[(dof-1)*2+k,cell] = A[k,7]*b[1] + A[k,8]*b[2]
                end
            end

            # cell bubbles have integral mean 1
            dof = ndofs4component*k
            fill!(b,0)
            b[k] = xCellVolumes[cell]
            for k = 1 : 2
                interior_coefficients[(dof-1)*2+k,cell] = A[k,7]*b[1] + A[k,8]*b[2]
            end
        end
    end
    interior_coefficients
end



function ExtendableGrids.instantiate(xgrid::ExtendableGrid, ::Type{ReconstructionCoefficients{FE1,FE2,AT}}) where {FE1<:H1P2B{2,2}, FE2<:HDIVBDM2{2}, AT <: ON_CELLS}
    @info "Computing interior reconstruction coefficients for $FE1 > $FE2 ($AT)"
    xCellFaces = xgrid[CellFaces]
    xCellVolumes::Array{Float64,1} = xgrid[CellVolumes]
    EG = xgrid[UniqueCellGeometries]

    @assert EG == [Triangle2D]

    ndofs1::Int = get_ndofs(AT,FE1,EG[1])
    ncells = num_sources(xCellFaces)
    rcoeff_handler! = get_reconstruction_coefficients!(xgrid, AT, FE1, FE2, EG[1])
    interior_offset = 9
    interior_ndofs = 3
    coefficients = zeros(Float64,ndofs1,interior_offset)
    interior_coefficients = zeros(Float64,interior_ndofs*ndofs1,ncells)


    qf = QuadratureRule{Float64,EG[1]}(4)
    weights::Array{Float64,1} = qf.w
    # evaluation of FE1 and FE2 basis
    FES1 = FESpace{FE1,AT}(xgrid)
    FES2 = FESpace{FE2,AT}(xgrid)
    FEB1 = FEBasisEvaluator{Float64,FE1,EG[1],Identity,ON_CELLS}(FES1, qf)
    FEB2 = FEBasisEvaluator{Float64,FE2,EG[1],Identity,ON_CELLS}(FES2, qf)
    # evaluation of gradient of P1 functions
    FE3 = H1P1{1}
    FES3 = FESpace{FE3,AT}(xgrid)
    FEB3 = FEBasisEvaluator{Float64,FE3,EG[1],Gradient,ON_CELLS}(FES3, qf)
    # evaluation of curl of bubble functions
    FE4 = H1BUBBLE{1}
    FES4 = FESpace{FE4,AT}(xgrid)
    FEB4 = FEBasisEvaluator{Float64,FE4,EG[1],CurlScalar,ON_CELLS}(FES4, qf)

    basisvals1::Array{Float64,3} = FEB1.cvals
    basisvals2::Array{Float64,3} = FEB2.cvals
    basisvals3::Array{Float64,3} = FEB3.cvals
    basisvals4::Array{Float64,3} = FEB4.cvals
    IMM_face = zeros(Float64,interior_ndofs,interior_offset)
    IMM = zeros(Float64,interior_ndofs,interior_ndofs)
    for k = 1 : interior_ndofs
        IMM[k,k] = 1
    end
    lb = zeros(Float64,interior_ndofs)
    lx = zeros(Float64,interior_ndofs)
    temp::Float64 = 0
    offset::Int = 0
    IMMfact = lu(IMM)
    for cell = 1 : ncells
        # update basis
        update!(FEB1,cell)
        update!(FEB2,cell)
        update!(FEB3,cell)
        update!(FEB4,cell)

        # get reconstruction coefficients for boundary dofs
        rcoeff_handler!(coefficients, cell)

        # compute local mass matrices
        fill!(IMM,0)
        fill!(IMM_face,0)
        for i in eachindex(weights)
            for dof = 1:interior_ndofs
                # interior FE2 basis functions times grad(P1) of first two P1 functions
                for dof2 = 1 : interior_ndofs - 1
                    temp = 0
                    for k = 1 : 2
                        temp += basisvals2[k,interior_offset + dof,i] * basisvals3[k,dof2,i]
                    end
                    IMM[dof2,dof] += temp * xCellVolumes[cell] * weights[i]
                end
                # interior FE2 basis functions times curl(bubble)
                temp = 0
                for k = 1 : 2
                    temp += basisvals2[k,interior_offset + dof,i] * basisvals4[k,1,i]
                end
                IMM[3,dof] += temp * xCellVolumes[cell] * weights[i]

                # mass matrix of face basis functions x grad(P1) and curl(bubble)
                if dof < 3
                    for dof2 = 1 : interior_offset
                        temp = 0
                        for k = 1 : 2
                            temp += basisvals3[k,dof,i] * basisvals2[k,dof2,i]
                        end
                        IMM_face[dof,dof2] += temp * xCellVolumes[cell] * weights[i]
                    end
                    # mass matrix of face basis functions x interior basis functions
                elseif dof == 3
                    for dof2 = 1 : interior_offset
                        temp = 0
                        for k = 1 : 2
                            temp += basisvals4[k,1,i] * basisvals2[k,dof2,i]
                        end
                        IMM_face[dof,dof2] += temp * xCellVolumes[cell] * weights[i] 
                    end
                end
            end
        end

        # solve local systems
        IMMfact = lu(IMM)
        for dof1 = 1 : ndofs1
            # right-hand side
            fill!(lb,0)
            for i in eachindex(weights)
                for idof = 1:interior_ndofs-1
                    temp = 0
                    for k = 1 : 2
                        temp += basisvals1[k,dof1,i] * basisvals3[k,idof,i]
                    end
                    lb[idof] += temp *  xCellVolumes[cell] * weights[i]
                end
                temp = 0
                for k = 1 : 2
                    temp += basisvals1[k,dof1,i] * basisvals4[k,1,i]
                end
                lb[3] += temp *  xCellVolumes[cell] * weights[i]
            end

            # subtract face interpolation from right-hand side
            for idof = 1 : interior_ndofs, dof2 = 1 : interior_offset
                lb[idof] -= coefficients[dof1,dof2] * IMM_face[idof,dof2]
            end
        
            # solve local system
            ldiv!(lx, IMMfact, lb)
            offset = interior_ndofs*(dof1-1)
            for idof = 1 : interior_ndofs
                interior_coefficients[offset+idof,cell] = lx[idof]
            end
        end
    end
    interior_coefficients
end