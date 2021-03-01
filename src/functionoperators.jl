
# this type steers how an operator is evaluated on an item where it is discontinuous
abstract type DiscontinuityTreatment end
abstract type Average <: DiscontinuityTreatment end # average the values on both sides of the face
abstract type Jump <: DiscontinuityTreatment end # calculate the jump between both sides of the face

"""
$(TYPEDEF)

root type for FunctionOperators.
"""
abstract type AbstractFunctionOperator end # to dispatch which evaluator of the FE_basis_caller is used
"""
$(TYPEDEF)

identity operator: evaluates finite element function.
"""
abstract type Identity <: AbstractFunctionOperator end # 1*v_h
"""
$(TYPEDEF)

identity operator: evaluates only the c-th component of the finite element function.
"""
abstract type IdentityComponent{c} <: AbstractFunctionOperator where {c<:Int} end # 1*v_h(c)
"""
$(TYPEDEF)

identity jump operator: evaluates face jumps of finite element function
"""
abstract type IdentityDisc{DT<:DiscontinuityTreatment} <: Identity end # [[v_h]]
"""
$(TYPEDEF)

reconstruction identity operator: evaluates a reconstructed version of the finite element function.

FEreconst specifies the reconstruction space and reconstruction algorithm if it is defined for the finite element that it is applied to.
"""
abstract type AbstractFiniteElement end
abstract type ReconstructionIdentity{FEreconst<:AbstractFiniteElement} <: Identity end # 1*R(v_h)
abstract type ReconstructionIdentityDisc{FEreconst<:AbstractFiniteElement, DT<:DiscontinuityTreatment} <: ReconstructionIdentity{FEreconst} end # 1*R(v_h)
"""
$(TYPEDEF)

evaluates the normal-flux of the finite element function.

only available on FACES/BFACES and currently only for H1 and Hdiv elements
"""
abstract type NormalFlux <: AbstractFunctionOperator end # v_h * n_F # only for Hdiv/H1 on Faces/BFaces
"""
$(TYPEDEF)

reconstruction normal flux: evaluates the normal flux of a reconstructed version of the finite element function.

FEreconst specifies the reconstruction space and reconstruction algorithm if it is defined for the finite element that it is applied to.
"""
abstract type ReconstructionNormalFlux{FEreconst<:AbstractFiniteElement} <: NormalFlux end # R(v_h) * n_F

abstract type TangentFlux <: AbstractFunctionOperator end # v_h * t_F # only for HCurlScalar on Edges
"""
$(TYPEDEF)

evaluates the gradient of the finite element function.
"""
abstract type Gradient <: AbstractFunctionOperator end # 1*v_h
abstract type GradientDisc{DT<:DiscontinuityTreatment} <: Gradient end # 1*v_h
"""
$(TYPEDEF)

reconstruction gradient operator: evaluates the gradient of a reconstructed version of the finite element function.

FEreconst specifies the reconstruction space and reconstruction algorithm if it is defined for the finite element that it is applied to.
"""
abstract type ReconstructionGradient{FEreconst<:AbstractFiniteElement} <: Gradient end # 1*R(v_h)
abstract type ReconstructionGradientDisc{FEreconst<:AbstractFiniteElement, DT<:DiscontinuityTreatment} <: ReconstructionGradient{FEreconst} end # 1*R(v_h)
"""
$(TYPEDEF)

evaluates the symmetric part of the gradient of the finite element function.
"""
abstract type SymmetricGradient <: AbstractFunctionOperator end # sym(D_geom(v_h))
"""
$(TYPEDEF)

evaluates the gradient of the tangential part of some vector-valued finite element function.
"""
abstract type TangentialGradient <: AbstractFunctionOperator end # D_geom(v_h x n) = only gradient of tangential part of vector-valued function
"""
$(TYPEDEF)

evaluates the Laplacian of some (possibly vector-valued) finite element function.
"""
abstract type Laplacian <: AbstractFunctionOperator end # L_geom(v_h)
"""
$(TYPEDEF)

evaluates the full Hessian of some (possibly vector-valued) finite element function.
"""
abstract type Hessian <: AbstractFunctionOperator end # D^2(v_h)
"""
$(TYPEDEF)

evaluates the curl of some scalar function in 2D, i.e. the rotated gradient.
"""
abstract type CurlScalar <: AbstractFunctionOperator end # only 2D: CurlScalar(v_h) = D(v_h)^\perp

"""
$(TYPEDEF)

evaluates the curl of some two-dimensional vector field, i.e. Curl2D((u1,u2)) = du2/dx1 - du1/dx2
"""
abstract type Curl2D <: AbstractFunctionOperator end

"""
$(TYPEDEF)

evaluates the curl of some three-dimensional vector field, i.e. Curl3D(u) = \nabla \times u
"""
abstract type Curl3D <: AbstractFunctionOperator end # only 3D: Curl3D(v_h) = D \times v_h
"""
$(TYPEDEF)

evaluates the divergence of the finite element function.
"""
abstract type Divergence <: AbstractFunctionOperator end # div(v_h)
"""
$(TYPEDEF)

evaluates the divergence of the reconstructed finite element function.

FEreconst specifies the reconstruction space and reconstruction algorithm if it is defined for the finite element that it is applied to.
"""
abstract type ReconstructionDivergence{FEreconst<:AbstractFiniteElement} <: Divergence end # 1*R(v_h)

abstract type Trace <: AbstractFunctionOperator end # tr(v_h)
abstract type Deviator <: AbstractFunctionOperator end # dev(v_h)


NeededDerivative4Operator(::Type{<:Identity}) = 0
NeededDerivative4Operator(::Type{<:IdentityComponent}) = 0
NeededDerivative4Operator(::Type{<:NormalFlux}) = 0
NeededDerivative4Operator(::Type{TangentFlux}) = 0
NeededDerivative4Operator(::Type{<:Gradient}) = 1
NeededDerivative4Operator(::Type{SymmetricGradient}) = 1
NeededDerivative4Operator(::Type{TangentialGradient}) = 1
NeededDerivative4Operator(::Type{Laplacian}) = 2
NeededDerivative4Operator(::Type{Hessian}) = 2
NeededDerivative4Operator(::Type{CurlScalar}) = 1
NeededDerivative4Operator(::Type{Curl2D}) = 1
NeededDerivative4Operator(::Type{Curl3D}) = 1
NeededDerivative4Operator(::Type{<:Divergence}) = 1
NeededDerivative4Operator(::Type{Trace}) = 0
NeededDerivative4Operator(::Type{Deviator}) = 0

DefaultName4Operator(::Type{Identity}) = "id"
DefaultName4Operator(::Type{IdentityDisc{Jump}}) = "[[id]]"
DefaultName4Operator(::Type{IdentityDisc{Average}}) = "{{id}}"
DefaultName4Operator(::Type{<:ReconstructionIdentity}) = "id R"
DefaultName4Operator(IC::Type{<:IdentityComponent}) = "id_$(IC.parameters[1])"
DefaultName4Operator(::Type{NormalFlux}) = "NormalFlux"
DefaultName4Operator(::Type{<:ReconstructionNormalFlux}) = "R NormalFlux"
DefaultName4Operator(::Type{TangentFlux}) = "TangentFlux"
DefaultName4Operator(::Type{<:Gradient}) = "grad"
DefaultName4Operator(::Type{SymmetricGradient}) = "symgrad"
DefaultName4Operator(::Type{TangentialGradient}) = "TangentialGradient"
DefaultName4Operator(::Type{Laplacian}) = "Laplace"
DefaultName4Operator(::Type{Hessian}) = "Hessian"
DefaultName4Operator(::Type{CurlScalar}) = "Curl"
DefaultName4Operator(::Type{Curl2D}) = "Curl"
DefaultName4Operator(::Type{Curl3D}) = "Curl"
DefaultName4Operator(::Type{Divergence}) = "div"
DefaultName4Operator(::Type{<:ReconstructionDivergence}) = "div R"
DefaultName4Operator(::Type{<:ReconstructionGradient}) = "grad R"
DefaultName4Operator(::Type{Trace}) = "tr"
DefaultName4Operator(::Type{Deviator}) = "dev"

# length for operator result
Length4Operator(::Type{<:Identity}, xdim::Int, ncomponents::Int) = ncomponents
Length4Operator(::Type{<:IdentityComponent}, xdim::Int, ncomponents::Int) = 1
Length4Operator(::Type{<:NormalFlux}, xdim::Int, ncomponents::Int) = 1
Length4Operator(::Type{TangentFlux}, xdim::Int, ncomponents::Int) = 1
Length4Operator(::Type{<:Divergence}, xdim::Int, ncomponents::Int) = ceil(ncomponents/xdim)
Length4Operator(::Type{Trace}, xdim::Int, ncomponents::Int) = ceil(sqrt(ncomponents))
Length4Operator(::Type{CurlScalar}, xdim::Int, ncomponents::Int) = ((xdim == 2) ? xdim*ncomponents : ceil(xdim*(ncomponents/xdim)))
Length4Operator(::Type{Curl2D}, xdim::Int, ncomponents::Int) = 1
Length4Operator(::Type{Curl3D}, xdim::Int, ncomponents::Int) = 3
Length4Operator(::Type{<:Gradient}, xdim::Int, ncomponents::Int) = xdim*ncomponents
Length4Operator(::Type{TangentialGradient}, xdim::Int, ncomponents::Int) = 1
Length4Operator(::Type{SymmetricGradient}, xdim::Int, ncomponents::Int) = ((xdim == 2) ? 3 : 6)*ceil(ncomponents/xdim)
Length4Operator(::Type{Hessian}, xdim::Int, ncomponents::Int) = xdim*xdim*ncomponents
Length4Operator(::Type{Laplacian}, xdim::Int, ncomponents::Int) = ncomponents

QuadratureOrderShift4Operator(::Type{<:Identity}) = 0
QuadratureOrderShift4Operator(::Type{<:IdentityComponent}) = 0
QuadratureOrderShift4Operator(::Type{<:NormalFlux}) = 0
QuadratureOrderShift4Operator(::Type{TangentFlux}) = 0
QuadratureOrderShift4Operator(::Type{<:Gradient}) = -1
QuadratureOrderShift4Operator(::Type{CurlScalar}) = -1
QuadratureOrderShift4Operator(::Type{Curl2D}) = -1
QuadratureOrderShift4Operator(::Type{Curl3D}) = -1
QuadratureOrderShift4Operator(::Type{<:Divergence}) = -1
QuadratureOrderShift4Operator(::Type{SymmetricGradient}) = -1
QuadratureOrderShift4Operator(::Type{TangentialGradient}) = -1
QuadratureOrderShift4Operator(::Type{Laplacian}) = -2
QuadratureOrderShift4Operator(::Type{Hessian}) = -2


# this function decides which basis should be evaluated for the evaluation of an operator
# e.g. Hdiv elements can use the face basis for the evaluation of the normal flux operator,
# but an H1 element must evaluate the cell basis for GradientDisc{Jump} ON_FACES
function DefaultBasisAssemblyType4Operator(operator::Type{<:AbstractFunctionOperator}, patternAT::Type{<:AbstractAssemblyType}, continuity::Type{<:AbstractFiniteElement})
    if patternAT == ON_CELLS
        return ON_CELLS
    elseif patternAT <: Union{<:ON_FACES,<:ON_BFACES}
        if continuity <: AbstractH1FiniteElement
            if QuadratureOrderShift4Operator(operator) == 0
                return patternAT
            else
                return ON_CELLS
            end
        elseif continuity <: AbstractHdivFiniteElement
            if QuadratureOrderShift4Operator(operator) == 0 && operator <: NormalFlux
                return patternAT
            else
                return ON_CELLS
            end
        elseif continuity <: AbstractHcurlFiniteElement
            if QuadratureOrderShift4Operator(operator) == 0 && operator <: TangentFlux
                return patternAT
            else
                return ON_CELLS
            end
        else
            return ON_CELLS
        end
    elseif patternAT <: Union{<:ON_EDGS,<:ON_BEDGES}
        if continuity <: AbstractH1FiniteElement
            if QuadratureOrderShift4Operator(operator) == 0
                return patternAT
            else
                return ON_CELLS
            end
        elseif continuity <: AbstractHdivFiniteElement
            return ON_CELLS
        elseif continuity <: AbstractHcurlFiniteElement
            if QuadratureOrderShift4Operator(operator) == 0 && operator <: TangentFlux
                return patternAT
            else
                return ON_CELLS
            end
        else
            return ON_CELLS
        end
    else return patternAT
    end
end


function DofitemInformation4Operator(FES::FESpace, AT::Type{<:AbstractAssemblyType}, basisAT::Type{<:AbstractAssemblyType}, FO::Type{<:AbstractFunctionOperator})
    # check if operator is discontinuous for this AT
    xgrid = FES.xgrid
    discontinuous = false
    posdt = 0
    for j = 1 : length(FO.parameters)
        if FO.parameters[j] <: DiscontinuityTreatment
            discontinuous = true
            posdt = j
            break;
        end
    end
    if discontinuous
        # call discontinuity handlers
        # for e.g. IdentityDisc, GradientDifsc etc.
        # if AT == ON_FACES: if continuity allows it, assembly ON_FACES two times on each face
        #                    otherwise leads to ON_CELL assembly on neighbouring CELLS
        # if AT == ON_EDGES: todo (should lead to ON_CELL assembly on neighbouring CELLS, morea than two)
        DofitemInformation4Operator(FES, AT, basisAT, FO.parameters[posdt])
    else
        # call standard handlers
        # assembly as specified by AT
        DofitemInformation4Operator(FES, AT, basisAT)
    end
end

# default handlers for continupus operators
function DofitemInformation4Operator(FES::FESpace, AT::Type{<:AbstractAssemblyType}, basisAT::Type{<:AbstractAssemblyType})
    xgrid = FES.xgrid
    xItemGeometries = xgrid[GridComponentGeometries4AssemblyType(AT)]
    EG = xgrid[GridComponentUniqueGeometries4AssemblyType(AT)]
    # operator is assumed to be continuous, hence only needs to be evaluated on one dofitem = item
    function closure(dofitems, EG4dofitem, itempos4dofitem, coefficient4dofitem, orientation4dofitem, item)
        dofitems[1] = item
        dofitems[2] = 0
        itempos4dofitem[1] = 1
        coefficient4dofitem[1] = 1
        # find EG index for geometry
        for j=1:length(EG)
            if xItemGeometries[item] == EG[j]
                EG4dofitem[1] = j
                break;
            end
        end
        return EG4dofitem[1]
    end
    return closure
end

function DofitemInformation4Operator(FES::FESpace, EG, EGdofitems, AT::Type{<:ON_CELLS}, DiscType::Type{<:Union{Jump, Average}})
    return DofitemInformation4Operator(FES, EG, EGdofitems, AT)
end

# special handlers for jump operators with ON_CELL basis
function DofitemInformation4Operator(FES::FESpace, AT::Type{<:ON_FACES}, basisAT::Type{<:ON_CELLS}, DiscType::Type{<:Union{Jump, Average}})
    xgrid = FES.xgrid
    xFaceCells = xgrid[FaceCells]
    xCellFaces = xgrid[CellFaces]
    xFaceGeometries = xgrid[FaceGeometries]
    xCellGeometries = xgrid[CellGeometries]
    xCellFaceOrientations = xgrid[CellFaceOrientations]
    EG = xgrid[GridComponentUniqueGeometries4AssemblyType(AT)]
    EGdofitems = xgrid[GridComponentUniqueGeometries4AssemblyType(basisAT)]
    if DiscType == Jump
        coeff_left = 1
        coeff_right = -1
    elseif DiscType == Average
        coeff_left = 0.5
        coeff_right = 0.5
    end
    # operator is discontinous ON_FACES and needs to be evaluated on the two neighbouring cells
    function closure!(dofitems, EG4dofitem, itempos4dofitem, coefficient4dofitem, orientation4dofitem, item)
        dofitems[1] = xFaceCells[1,item]
        dofitems[2] = xFaceCells[2,item]
        for k = 1 : num_targets(xCellFaces,dofitems[1])
            if xCellFaces[k,dofitems[1]] == item
                itempos4dofitem[1] = k
            end
        end
        orientation4dofitem[1] = xCellFaceOrientations[itempos4dofitem[1], dofitems[1]]
        if dofitems[2] > 0
            for k = 1 : num_targets(xCellFaces,dofitems[2])
                if xCellFaces[k,dofitems[2]] == item
                    itempos4dofitem[2] = k
                end
            end
            orientation4dofitem[2] = xCellFaceOrientations[itempos4dofitem[2], dofitems[2]]
            coefficient4dofitem[1] = coeff_left
            coefficient4dofitem[2] = coeff_right
        else
            coefficient4dofitem[1] = 1
            coefficient4dofitem[2] = 0
            if AT == ON_IFACES
                # if assembly is only on interior faces, ignore boundary faces by setting dofitems to zero
                dofitems[1] = 0
            end
        end

        # find EG index for geometry
        for k = 1 : 2
            if dofitems[k] > 0
                for j=1:length(EGdofitems)
                    if xCellGeometries[dofitems[k]] == EGdofitems[j]
                        EG4dofitem[k] = length(EG) + j
                        break;
                    end
                end
            end
        end
        for j=1:length(EG)
            if xFaceGeometries[item] == EG[j]
                return j
            end
        end
    end
    return closure!
end


# special handlers for jump operators on FACES of broken spaces that otherwise have the necessary continuity (so that they can use the ON_FACES basis)
function DofitemInformation4Operator(FES::FESpace, AT::Type{<:ON_FACES}, basisAT::Type{<:ON_FACES}, DiscType::Type{<:Union{Jump, Average}})
    xgrid = FES.xgrid
    xFaceCells = xgrid[FaceCells]
    xCellFaces = xgrid[CellFaces]
    xItemGeometries = xgrid[FaceGeometries]
    EG = xgrid[GridComponentUniqueGeometries4AssemblyType(AT)]
    EGdofitems = xgrid[GridComponentUniqueGeometries4AssemblyType(basisAT)]
    if DiscType == Jump
        coeff_left = 1
        coeff_right = -1
    elseif DiscType == Average
        coeff_left = 0.5
        coeff_right = 0.5
    end
    localndofs = zeros(Int, length(EGdofitems))
    for j = 1 : length(EGdofitems)
        localndofs[j] = get_ndofs(basisAT, typeof(FES).parameters[1], EGdofitems[j])
    end
    # operator is discontinous ON_FACES and needs to be evaluated in the face dofs of the neighbourings cells
    function closure!(dofitems, EG4dofitem, itempos4dofitem, coefficient4dofitem, orientation4dofitem, dofoffset4dofitem, item)
        if xFaceCells[2,item] > 0
            coefficient4dofitem[1] = coeff_left
            coefficient4dofitem[2] = coeff_right
            dofitems[1] = item
            dofitems[2] = item
        else
            coefficient4dofitem[1] = 1
            coefficient4dofitem[2] = 0
            dofitems[1] = item
            dofitems[2] = 0
        end
        # find EG index for geometry
        for j=1:length(EG)
            if xItemGeometries[item] == EG[j]
                EG4dofitem[1] = j
                EG4dofitem[2] = j
                dofoffset4dofitem[2] = localndofs[j]
                break;
            end
        end
        return EG4dofitem[1]
    end
    return closure!
end


# special handlers for jump operators with ON_CELL basis
function DofitemInformation4Operator(FES::FESpace, AT::Type{<:ON_BFACES}, basisAT::Type{<:ON_CELLS}, DiscType::Type{<:Union{Jump, Average}})
    xgrid = FES.xgrid
    xFaceCells = xgrid[FaceCells]
    xCellFaces = xgrid[CellFaces]
    xBFaceGeometries = xgrid[BFaceGeometries]
    xCellGeometries = xgrid[CellGeometries]
    xBFaces = xgrid[BFaces]
    xCellFaceOrientations = xgrid[CellFaceOrientations]
    EG = xgrid[GridComponentUniqueGeometries4AssemblyType(AT)]
    EGdofitems = xgrid[GridComponentUniqueGeometries4AssemblyType(basisAT)]

    # operator is discontinous ON_FACES and needs to be evaluated on the two neighbouring cells
    function closure!(dofitems, EG4dofitem, itempos4dofitem, coefficient4dofitem, orientation4dofitem, item)
        dofitems[1] = xFaceCells[1,xBFaces[item]]
        dofitems[2] = xFaceCells[2,xBFaces[item]]
        for k = 1 : num_targets(xCellFaces,dofitems[1])
            if xCellFaces[k,dofitems[1]] == xBFaces[item]
                itempos4dofitem[1] = k
            end
        end
        orientation4dofitem[1] = xCellFaceOrientations[itempos4dofitem[1], dofitems[1]]
        coefficient4dofitem[1] = 1

        # find EG index for geometry
        for j=1:length(EG)
            if xCellGeometries[dofitems[1]] == EGdofitems[j]
                EG4dofitem[1] = length(EG) + j
                break;
            end
        end
        for j=1:length(EG)
            if xBFaceGeometries[item] == EG[j]
                return j
            end
        end
    end
    return closure!
end
