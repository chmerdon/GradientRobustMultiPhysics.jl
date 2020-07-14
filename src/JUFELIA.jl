module JUFELIA

using ExtendableGrids
using ExtendableSparse
using SparseArrays
using BenchmarkTools
using LinearAlgebra
using ForwardDiff # for FEBasisEvaluator
using Printf

include("AssemblyJunctions.jl");
export AbstractAssemblyType
export AbstractAssemblyTypeCELL, AbstractAssemblyTypeFACE, AbstractAssemblyTypeBFACE, AbstractAssemblyTypeBFACECELL
export GridComponentNodes4AssemblyType
export GridComponentVolumes4AssemblyType
export GridComponentGeometries4AssemblyType
export GridComponentRegions4AssemblyType


include("L2GTransformer.jl");
export L2GTransformer, update!, eval!, mapderiv!, piola!


include("FEXGrid.jl")
export FaceNodes, FaceGeometries, FaceVolumes, FaceRegions, FaceCells, FaceNormals
export CellFaces, CellSigns, CellVolumes
export BFaces, BFaceCellPos, BFaceVolumes
export nfaces_per_cell, facetype_of_cellface
export uniqueEG,split_grid_into, uniform_refine, barycentric_refine


include("QuadratureRules.jl")
export QuadratureRule, VertexRule, integrate!


include("FiniteElements.jl")
export AbstractFiniteElement
export FESpace

export AbstractH1FiniteElement
export H1P1, H1P2, H1MINI, H1CR
export L2P0, L2P1

export AbstractH1FiniteElementWithCoefficients
export H1BR

export AbstractHdivFiniteElement
export HDIVRT0

export AbstractHcurlFiniteElement

export get_ncomponents
export reconstruct!


include("FEBlockArrays.jl");
export FEVectorBlock, FEVector
export FEMatrixBlock, FEMatrix
export fill!, addblock, addblock_matmul


include("FEBasisEvaluator.jl")
export FEBasisEvaluator, update!

export AbstractFunctionOperator
export Identity, ReconstructionIdentity
export NormalFlux, TangentFlux
export Gradient, SymmetricGradient, TangentialGradient
export Divergence, ReconstructionDivergence
export Curl, Rotation
export Laplacian, Hessian
export Trace, Deviator

export NeededDerivatives4Operator, QuadratureOrderShift4Operator, FEPropertyDofs4AssemblyType
export DefaultDirichletBoundaryOperator4FE


include("FEInterpolations.jl")
export interpolate! # must be defined separately by each FEdefinition
export nodevalues! # = P1interpolation, abstract averaging method that works for any element, but can be overwritten by FEdefinition to something simpler


include("AbstractAction.jl")
export AbstractAction
export DoNotChangeAction
export MultiplyScalarAction
export MultiplyVectorAction
export MultiplyMatrixAction
export RegionWiseMultiplyScalarAction
export RegionWiseMultiplyVectorAction
export FunctionAction
export XFunctionAction
export ItemWiseXFunctionAction
export RegionWiseXFunctionAction

include("AbstractAssemblyPattern.jl")
export AbstractAssemblyPattern,ItemIntegrator,LinearForm,BilinearForm,SymmetricBilinearForm,TrilinearForm
export assemble!, evaluate!, evaluate
export L2ErrorIntegrator


include("PDEOperators.jl")
export AbstractAssemblyTrigger
export AbstractPDEOperator

export BackwardEulerTimeDerivative

export AbstractBilinearForm
export StiffnessOperator, LaplaceOperator, HookStiffnessOperator2D, HookStiffnessOperator1D, ReactionOperator

export DiagonalOperator, CopyOperator
export ConvectionOperator
export LagrangeMultiplier
export FVUpwindDivergenceOperator

export RhsOperator, BLFeval


include("PDEBoundaryData.jl")
export BoundaryOperator
export AbstractBoundaryType, HomogeneousDirichletBoundary, InterpolateDirichletBoundary, BestapproxDirichletBoundary


include("PDEGlobalConstraints.jl")
export AbstractGlobalConstraint
export FixedIntegralMean
export CombineDofs


include("PDEDescription.jl")
export PDEDescription


include("PDESolver.jl")
export TimeControlSolver, advance
export assemble!, solve!

export AbstractTimeIntegrationRule
export BackwardEuler


include("PDEProtoTypes.jl")
export CompressibleNavierStokesProblem
export IncompressibleNavierStokesProblem
export LinearElasticityProblem
export PoissonProblem
export L2BestapproximationProblem
export H1BestapproximationProblem


end #module