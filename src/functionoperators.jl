
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
abstract type ReconstructionIdentity{FEreconst<:AbstractFiniteElement} <: Identity end # 1*R(v_h)
abstract type ReconstructionIdentityDisc{FEreconst<:AbstractFiniteElement, DT<:DiscontinuityTreatment} <: ReconstructionIdentity{FEreconst} end # 1*R(v_h)
"""
$(TYPEDEF)

evaluates the normal-flux of the finite element function.

only available on FACES/BFACES and currently only for H1 and Hdiv elements
"""
abstract type NormalFlux <: AbstractFunctionOperator end # v_h * n_F # only for Hdiv/H1 on Faces/BFaces

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

abstract type Laplacian <: AbstractFunctionOperator end # L_geom(v_h)
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

# operator to be used for Dirichlet boundary data
DefaultDirichletBoundaryOperator4FE(::Type{<:AbstractL2FiniteElement}) = Identity
DefaultDirichletBoundaryOperator4FE(::Type{<:AbstractH1FiniteElement}) = Identity
DefaultDirichletBoundaryOperator4FE(::Type{<:AbstractHdivFiniteElement}) = NormalFlux
DefaultDirichletBoundaryOperator4FE(::Type{<:AbstractHcurlFiniteElement}) = TangentFlux

NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{<:Identity}) = 0
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{<:IdentityComponent}) = 0
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{NormalFlux}) = 0
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{TangentFlux}) = 0
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{<:Gradient}) = 1
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{SymmetricGradient}) = 1
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{TangentialGradient}) = 1
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{Laplacian}) = 2
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{Hessian}) = 2
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{CurlScalar}) = 1
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{Curl2D}) = 1
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{Curl3D}) = 1
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{<:Divergence}) = 1
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{Trace}) = 0
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{Deviator}) = 0

DefaultName4Operator(::Type{Identity}) = "id"
DefaultName4Operator(::Type{IdentityDisc{Jump}}) = "[[id]]"
DefaultName4Operator(::Type{IdentityDisc{Average}}) = "{{id}}"
DefaultName4Operator(::Type{<:ReconstructionIdentity}) = "id R"
DefaultName4Operator(IC::Type{<:IdentityComponent}) = "id_$(IC.parameters[1])"
DefaultName4Operator(::Type{NormalFlux}) = "NormalFlux"
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
Length4Operator(::Type{NormalFlux}, xdim::Int, ncomponents::Int) = ncomponents
Length4Operator(::Type{TangentFlux}, xdim::Int, ncomponents::Int) = ncomponents
Length4Operator(::Type{<:Divergence}, xdim::Int, ncomponents::Int) = ceil(ncomponents/xdim)
Length4Operator(::Type{Trace}, xdim::Int, ncomponents::Int) = ceil(sqrt(ncomponents))
Length4Operator(::Type{CurlScalar}, xdim::Int, ncomponents::Int) = ((xdim == 2) ? xdim*ncomponents : ceil(xdim*(ncomponents/xdim)))
Length4Operator(::Type{Curl2D}, xdim::Int, ncomponents::Int) = 1
Length4Operator(::Type{Curl3D}, xdim::Int, ncomponents::Int) = 3
Length4Operator(::Type{<:Gradient}, xdim::Int, ncomponents::Int) = xdim*ncomponents
Length4Operator(::Type{TangentialGradient}, xdim::Int, ncomponents::Int) = 1
Length4Operator(::Type{SymmetricGradient}, xdim::Int, ncomponents::Int) = ((xdim == 2) ? 3 : 6)*ceil(ncomponents/xdim)
Length4Operator(::Type{Hessian}, xdim::Int, ncomponents::Int) = xdim*xdim*ncomponents

QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{<:Identity}) = 0
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{<:IdentityComponent}) = 0
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{NormalFlux}) = 0
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{TangentFlux}) = 0
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{<:Gradient}) = -1
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{CurlScalar}) = -1
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{Curl2D}) = -1
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{Curl3D}) = -1
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{<:Divergence}) = -1
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{SymmetricGradient}) = -1
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{TangentialGradient}) = -1
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{Laplacian}) = -2
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{Hessian}) = -2

Dofmap4AssemblyType(FES::FESpace, ::Type{ON_CELLS}) = FES.dofmaps[CellDofs]
Dofmap4AssemblyType(FES::FESpace, ::Type{<:ON_FACES}) = FES.dofmaps[FaceDofs]
Dofmap4AssemblyType(FES::FESpace, ::Type{ON_BFACES}) = FES.dofmaps[BFaceDofs]
Dofmap4AssemblyType(FES::FESpace, ::Type{<:ON_EDGES}) = FES.dofmaps[EdgeDofs]
Dofmap4AssemblyType(FES::FESpace, ::Type{ON_BEDGES}) = FES.dofmaps[BEdgeDofs]

# default junctions
function DofitemAT4Operator(AT::Type{<:AbstractAssemblyType}, FO::Type{<:AbstractFunctionOperator})
    # check if operator is discontinuous for this AT
    for j = 1 : length(FO.parameters)
        if typeof(FO.parameters[j]) != Int64
            if FO.parameters[j] <: DiscontinuityTreatment
                return ON_CELLS
            end
        end
    end
    return AT
end

function DofitemInformation4Operator(FES::FESpace, EG, EGdofitem, AT::Type{<:AbstractAssemblyType}, FO::Type{<:AbstractFunctionOperator})
    # check if operator is discontinuous for this AT
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
        DofitemInformation4Operator(FES, EG, EGdofitem, AT, FO.parameters[posdt])
    else
        # call standard handlers
        DofitemInformation4Operator(FES, EG, EGdofitem, AT)
    end
end

# default handlers
function DofitemInformation4Operator(FES::FESpace, EG, EGdofitem, AT::Type{<:AbstractAssemblyType})
    xItemGeometries = FES.xgrid[GridComponentGeometries4AssemblyType(AT)]
    # operator is assumed to be continuous, hence only needs to be evaluated on one dofitem = item
    function closure(dofitems, EG4dofitem, itempos4dofitem, coefficient4dofitem, item)
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

# special handlers for jump operators
function DofitemInformation4Operator(FES::FESpace, EG, EGdofitems, AT::Type{<:ON_FACES}, DiscType::Type{<:Union{Jump, Average}})
    xFaceCells = FES.xgrid[FaceCells]
    xCellFaces = FES.xgrid[CellFaces]
    xFaceGeometries = FES.xgrid[FaceGeometries]
    xCellGeometries = FES.xgrid[CellGeometries]
    if DiscType == Jump
        coeff_left = 1
        coeff_right = -1
    elseif DiscType == Average
        coeff_left = 0.5
        coeff_right = 0.5
    end
    # operator is discontinous ON_FACES and needs to be evaluated on the two neighbouring cells
    function closure!(dofitems, EG4dofitem, itempos4dofitem, coefficient4dofitem, item)
        dofitems[1] = xFaceCells[1,item]
        dofitems[2] = xFaceCells[2,item]
        for k = 1 : num_targets(xCellFaces,dofitems[1])
            if xCellFaces[k,dofitems[1]] == item
                itempos4dofitem[1] = k
            end
        end
        if dofitems[2] > 0
            for k = 1 : num_targets(xCellFaces,dofitems[2])
                if xCellFaces[k,dofitems[2]] == item
                    itempos4dofitem[2] = k
                end
            end
        elseif AT == ON_IFACES
            # if assembly is only on interior faces, ignore boundary faces by setting dofitems to zero
            dofitems[1] = 0
        end
        coefficient4dofitem[1] = coeff_left
        coefficient4dofitem[2] = coeff_right

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
