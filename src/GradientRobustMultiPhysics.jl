module GradientRobustMultiPhysics

using Base: Bool
using ExtendableGrids # + some exports from there
export Edge1D, Triangle2D, Parallelogram2D, Tetrahedron3D, Parallelepiped3D
using GridVisualize
using ExtendableSparse
using SuiteSparse
using SparseArrays
using StaticArrays
using DiffResults
using WriteVTK
using LinearAlgebra
using ForwardDiff
using DocStringExtensions
using Printf
using Logging

include("logging.jl")


include("userdata.jl")
export AbstractDataFunction, AbstractActionKernel, AbstractExtendedDataFunction
export UserData, ActionKernel, NLActionKernel, DataFunction, ExtendedDataFunction, eval_data!
export is_timedependent, is_regiondependent, is_itemdependent

include("quadrature.jl")
export QuadratureRule
export VertexRule
export integrate!, integrate, ref_integrate!

include("functionoperators.jl")
export AbstractFunctionOperator
export Identity, IdentityComponent, IdentityDisc
export ReconstructionIdentity, ReconstructionIdentityDisc
export ReconstructionGradient, ReconstructionGradientDisc
export ReconstructionDivergence
export ReconstructionNormalFlux
export NormalFlux, NormalFluxDisc, TangentFlux, TangentFluxDisc
export Gradient, GradientDisc
export SymmetricGradient, TangentialGradient
export Divergence, ReconstructionDivergence
export CurlScalar, Curl2D, Curl3D
export Laplacian, Hessian, SymmetricHessian
export Trace, Deviator
export NeededDerivatives4Operator, QuadratureOrderShift4Operator
export Dofmap4AssemblyType, DofitemAT4Operator
export DefaultDirichletBoundaryOperator4FE

export DiscontinuityTreatment, Jump, Average
export OperatorPair, OperatorTriple


include("finiteelements.jl")
export DofMap, CellDofs, FaceDofs, EdgeDofs, BFaceDofs, BEdgeDofs
export AbstractFiniteElement
export FESpace

export AbstractH1FiniteElement
export H1BUBBLE, H1P0, H1P1, H1P2, H1P2B, H1MINI, H1CR, H1P3

export AbstractH1FiniteElementWithCoefficients
export H1BR, H1P1TEB

export AbstractHdivFiniteElement
export HDIVRT0, HDIVBDM1, HDIVRT1, HDIVBDM2

export AbstractHcurlFiniteElement
export HCURLN0

export get_assemblytype
export get_polynomialorder, get_ndofs, get_ndofs_all
export get_ncomponents, get_edim
export get_basis, get_coefficients, get_basissubset
export reconstruct!

export interpolate! # must be defined separately by each FEdefinition
export nodevalues,nodevalues! # = P1interpolation, abstract averaging method that works for any element, but can be overwritten by FEdefinition to something simpler

export FEVectorBlock, FEVector
export FEMatrixBlock, FEMatrix, _addnz
export fill!, addblock!, addblock_matmul!, lrmatmul, add!

export get_reconstruction_matrix

export displace_mesh,displace_mesh!

include("reconstructions.jl")
export ReconstructionHandler, get_rcoefficients!

include("febasisevaluator.jl")
export FEBasisEvaluator, update_febe!, eval_febe!


include("actions.jl")
export AbstractAction, Action, MultiplyScalarAction, NoAction, fdot_action
export set_time!, update_action!, apply_action!


include("accumvector.jl")
export AccumulatingVector


include("assemblypatterns.jl")
export AssemblyPatternType, AssemblyPreparations
export ItemIntegrator, L2ErrorIntegrator, L2NormIntegrator, L2DifferenceIntegrator
export LinearForm
export BilinearForm, SymmetricBilinearForm, LumpedBilinearForm
export APT_BilinearForm, APT_SymmetricBilinearForm, APT_LumpedBilinearForm
export TrilinearForm
export MultilinearForm
export NonlinearForm
export prepare_assembly!
export assemble!, evaluate!, evaluate
export AssemblyManager, update_assembly!
export SegmentIntegrator
export PointEvaluator


include("pdeoperators.jl")
export AbstractAssemblyTrigger
export AbstractPDEOperator

export BackwardEulerTimeDerivative

export AbstractBilinearForm
export StiffnessOperator, LaplaceOperator
export HookStiffnessOperator3D, HookStiffnessOperator2D, HookStiffnessOperator1D
export ReactionOperator
export ConvectionOperator, ConvectionRotationFormOperator
export LagrangeMultiplier

export AbstractTrilinearForm
export AbstractMultilinearForm
export AbstractNonlinearForm, GenerateNonlinearForm

export FVConvectionDiffusionOperator
export DiagonalOperator, CopyOperator
export CustomMatrixOperator

export RhsOperator
export restrict_operator
export SchurComplement

export assemble_operator!, eval_assemble!


include("boundarydata.jl")
export BoundaryOperator
export AbstractBoundaryType, HomogeneousDirichletBoundary, InterpolateDirichletBoundary, BestapproxDirichletBoundary
export boundarydata!


include("globalconstraints.jl")
export AbstractGlobalConstraint
export FixedIntegralMean
export CombineDofs
export apply_constraint!


include("pdedescription.jl")
export PDEDescription
export add_unknown!
export add_operator!
export add_rhsdata!
export add_boundarydata!
export add_constraint!


include("solvers.jl")
export AbstractLinearSystem

export solve!, assemble!
export TimeControlSolver, advance!, advance_until_stationarity!, advance_until_time!
export show_statistics
export AbstractTimeIntegrationRule
export BackwardEuler, CrankNicolson

include("diffeq_interface.jl")
export eval_rhs!, eval_jacobian!, mass_matrix, jac_prototype

include("pdeprototypes.jl")
export IncompressibleNavierStokesProblem
export LinearElasticityProblem
export PoissonProblem
export L2BestapproximationProblem
export H1BestapproximationProblem

include("dataexport.jl")
export writeVTK!, writeCSV!
export print_table, print_convergencehistory

include("plots.jl")
export plot, plot_convergencehistory


end #module
