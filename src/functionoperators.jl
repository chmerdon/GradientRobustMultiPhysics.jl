
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
abstract type NormalFluxDisc{DT<:DiscontinuityTreatment} <: NormalFlux end # v_h * n_F # only for Hdiv/H1 on Faces/BFaces
"""
$(TYPEDEF)

reconstruction normal flux: evaluates the normal flux of a reconstructed version of the finite element function.

FEreconst specifies the reconstruction space and reconstruction algorithm if it is defined for the finite element that it is applied to.
"""
abstract type ReconstructionNormalFlux{FEreconst<:AbstractFiniteElement} <: NormalFlux end # R(v_h) * n_F

"""
$(TYPEDEF)

evaluates the tangent-flux of the finite element function.
"""
abstract type TangentFlux <: AbstractFunctionOperator end # v_h * t_F # only for HCurlScalar on Edges
abstract type TangentFluxDisc{DT<:DiscontinuityTreatment} <: TangentFlux end # v_h * t_F # only for HCurlScalar on Edges
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
NeededDerivative4Operator(::Type{<:TangentFlux}) = 0
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

DefaultName4Operator(::Type{<:AbstractFunctionOperator}) = "??"
DefaultName4Operator(::Type{Identity}) = "id"
DefaultName4Operator(::Type{IdentityDisc{Jump}}) = "[[id]]"
DefaultName4Operator(::Type{IdentityDisc{Average}}) = "{{id}}"
DefaultName4Operator(::Type{<:ReconstructionIdentity}) = "R"
DefaultName4Operator(IC::Type{<:IdentityComponent}) = "id_$(IC.parameters[1])"
DefaultName4Operator(::Type{NormalFlux}) = "NormalFlux"
DefaultName4Operator(::Type{NormalFluxDisc{Jump}}) = "[[NormalFlux]]"
DefaultName4Operator(::Type{NormalFluxDisc{Average}}) = "{{NormalFlux}}"
DefaultName4Operator(::Type{<:ReconstructionNormalFlux}) = "R NormalFlux"
DefaultName4Operator(::Type{TangentFlux}) = "TangentialFlux"
DefaultName4Operator(::Type{TangentFluxDisc{Jump}}) = "[[TangentialFlux]]"
DefaultName4Operator(::Type{TangentFluxDisc{Average}}) = "{{TangentialFlux}}"
DefaultName4Operator(::Type{<:Gradient}) = "∇"
DefaultName4Operator(::Type{SymmetricGradient}) = "ϵ"
DefaultName4Operator(::Type{TangentialGradient}) = "TangentialGradient"
DefaultName4Operator(::Type{Laplacian}) = "Δ"
DefaultName4Operator(::Type{Hessian}) = "H"
DefaultName4Operator(::Type{CurlScalar}) = "curl"
DefaultName4Operator(::Type{Curl2D}) = "Curl"
DefaultName4Operator(::Type{Curl3D}) = "∇×"
DefaultName4Operator(::Type{Divergence}) = "div"
DefaultName4Operator(::Type{<:ReconstructionDivergence}) = "div R"
DefaultName4Operator(::Type{<:ReconstructionGradient}) = "∇ R"
DefaultName4Operator(::Type{Trace}) = "tr"
DefaultName4Operator(::Type{Deviator}) = "dev"

function Base.show(io::Core.IO, FO::Type{<:AbstractFunctionOperator})
    print(io,"$(DefaultName4Operator(FO))")
end

# length for operator result
Length4Operator(::Type{<:Identity}, xdim::Int, ncomponents::Int) = ncomponents
Length4Operator(::Type{<:IdentityComponent}, xdim::Int, ncomponents::Int) = 1
Length4Operator(::Type{<:NormalFlux}, xdim::Int, ncomponents::Int) = 1
Length4Operator(::Type{<:TangentFlux}, xdim::Int, ncomponents::Int) = 1
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
QuadratureOrderShift4Operator(::Type{<:TangentFlux}) = 0
QuadratureOrderShift4Operator(::Type{<:Gradient}) = -1
QuadratureOrderShift4Operator(::Type{CurlScalar}) = -1
QuadratureOrderShift4Operator(::Type{Curl2D}) = -1
QuadratureOrderShift4Operator(::Type{Curl3D}) = -1
QuadratureOrderShift4Operator(::Type{<:Divergence}) = -1
QuadratureOrderShift4Operator(::Type{SymmetricGradient}) = -1
QuadratureOrderShift4Operator(::Type{TangentialGradient}) = -1
QuadratureOrderShift4Operator(::Type{Laplacian}) = -2
QuadratureOrderShift4Operator(::Type{Hessian}) = -2
