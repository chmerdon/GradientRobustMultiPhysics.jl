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

reconstruction identity operator: evaluates a reconstructed version of the finite element function.

FEreconst specifies the reconstruction space and reconstruction algorithm if it is defined for the finite element that it is applied to.
"""
abstract type ReconstructionIdentity{FEreconst<:AbstractFiniteElement} <: Identity end # 1*R(v_h)
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
abstract type Gradient <: AbstractFunctionOperator end # D_geom(v_h)
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
abstract type Rotation <: AbstractFunctionOperator end # only 3D: Rot(v_h) = D \times v_h
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
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{NormalFlux}) = 0
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{TangentFlux}) = 0
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{Gradient}) = 1
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{SymmetricGradient}) = 1
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{TangentialGradient}) = 1
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{Laplacian}) = 2
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{Hessian}) = 2
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{CurlScalar}) = 1
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{Rotation}) = 1
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{<:Divergence}) = 1
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{Trace}) = 0
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{Deviator}) = 0

# length for operator result
Length4Operator(::Type{<:Identity}, xdim::Int, ncomponents::Int) = ncomponents
Length4Operator(::Type{NormalFlux}, xdim::Int, ncomponents::Int) = ceil(ncomponents/xdim)
Length4Operator(::Type{TangentFlux}, xdim::Int, ncomponents::Int) = ceil(ncomponents/(xdim-1))
Length4Operator(::Type{Divergence}, xdim::Int, ncomponents::Int) = ceil(ncomponents/xdim)
Length4Operator(::Type{Trace}, xdim::Int, ncomponents::Int) = ceil(sqrt(ncomponents))
Length4Operator(::Type{CurlScalar}, xdim::Int, ncomponents::Int) = ((xdim == 2) ? xdim*ncomponents : ceil(xdim*(ncomponents/xdim)))
Length4Operator(::Type{Gradient}, xdim::Int, ncomponents::Int) = xdim*ncomponents
Length4Operator(::Type{TangentialGradient}, xdim::Int, ncomponents::Int) = 1
Length4Operator(::Type{SymmetricGradient}, xdim::Int, ncomponents::Int) = ((xdim == 2) ? 3 : 6)*ceil(ncomponents/xdim)
Length4Operator(::Type{Hessian}, xdim::Int, ncomponents::Int) = xdim*xdim*ncomponents

QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{<:Identity}) = 0
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{NormalFlux}) = 0
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{TangentFlux}) = 0
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{Gradient}) = -1
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{CurlScalar}) = -1
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{<:Divergence}) = -1
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{SymmetricGradient}) = -1
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{TangentialGradient}) = -1
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{Laplacian}) = -2
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{Hessian}) = -2

# junctions for dofmaps
Dofmap4Operator(FE::FESpace,::Type{ON_CELLS}, ::Type{<:AbstractFunctionOperator}) = FE.dofmaps[CellDofs]
Dofmap4Operator(FE::FESpace,::Type{ON_FACES}, ::Type{<:AbstractFunctionOperator}) = FE.dofmaps[FaceDofs]
Dofmap4Operator(FE::FESpace,::Type{ON_BFACES}, ::Type{<:AbstractFunctionOperator}) = FE.dofmaps[BFaceDofs]