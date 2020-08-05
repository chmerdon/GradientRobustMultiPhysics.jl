module GradientRobustMultiPhysics

using ExtendableGrids
using ExtendableSparse
using SuiteSparse
using SparseArrays
using BenchmarkTools
using LinearAlgebra
using ForwardDiff
using DocStringExtensions
using Printf


include("junctions.jl");
export AbstractAssemblyType
export AssemblyTypeCELL, AssemblyTypeFACE, AssemblyTypeBFACE, AssemblyTypeBFACECELL
export GridComponentNodes4AssemblyType
export GridComponentVolumes4AssemblyType
export GridComponentGeometries4AssemblyType
export GridComponentRegions4AssemblyType


include("l2gtransformations.jl");
export L2GTransformer, update!, eval!, mapderiv!, piola!


include("gridstuff.jl")
export CellFaces, CellEdges, CellFaceSigns, CellVolumes
export FaceNodes, FaceGeometries, FaceVolumes, FaceRegions, FaceCells, FaceNormals
export EdgeNodes, EdgeGeometries, EdgeVolumes
export BFaces, BFaceCellPos, BFaceVolumes
export nfaces_for_geometry, facetype_of_cellface
export uniqueEG


include("meshrefinements.jl")
export split_grid_into
export uniform_refine
export barycentric_refine


include("quadrature.jl")
export QuadratureRule
export VertexRule
export integrate!, integrate


include("finiteelements.jl")
export AbstractFiniteElement
export FESpace

export AbstractH1FiniteElement
export H1P1, H1P2, H1MINI, H1CR
export L2P0, L2P1 # will be AbstractL2FiniteElement one day

export AbstractH1FiniteElementWithCoefficients
export H1BR

export AbstractHdivFiniteElement
export HDIVRT0, HDIVBDM1

export AbstractHcurlFiniteElement

export get_ncomponents
export reconstruct!


include("fevector.jl");
export FEVectorBlock, FEVector
export fill!, addblock!


include("fematrix.jl");
export FEMatrixBlock, FEMatrix
export fill!, addblock!, addblock_matmul!


include("febasisevaluator.jl")
export FEBasisEvaluator, update!

export AbstractFunctionOperator
export Identity, ReconstructionIdentity
export NormalFlux, TangentFlux
export Gradient, SymmetricGradient, TangentialGradient
export Divergence, ReconstructionDivergence
export Curl, Rotation
export Laplacian, Hessian
export Trace, Deviator

export NeededDerivatives4Operator, QuadratureOrderShift4Operator
export FEPropertyDofs4AssemblyType
export DefaultDirichletBoundaryOperator4FE


include("interpolations.jl")
export interpolate! # must be defined separately by each FEdefinition
export nodevalues! # = P1interpolation, abstract averaging method that works for any element, but can be overwritten by FEdefinition to something simpler


include("actions.jl")
export AbstractAction
export DoNotChangeAction
export MultiplyScalarAction, MultiplyVectorAction
export MultiplyMatrixAction, RegionWiseMultiplyScalarAction, RegionWiseMultiplyVectorAction
export FunctionAction, XFunctionAction
export ItemWiseXFunctionAction, RegionWiseXFunctionAction


include("assemblypatterns.jl")
export AbstractAssemblyPattern
export ItemIntegrator
export LinearForm
export BilinearForm, SymmetricBilinearForm
export TrilinearForm
export assemble!, evaluate!, evaluate
export L2ErrorIntegrator


include("pdeoperators.jl")
export AbstractAssemblyTrigger
export AbstractPDEOperator

export BackwardEulerTimeDerivative

export AbstractBilinearForm
export StiffnessOperator, LaplaceOperator, HookStiffnessOperator2D, HookStiffnessOperator1D
export ReactionOperator
export ConvectionOperator
export LagrangeMultiplier

export AbstractTrilinearForm

export FVUpwindDivergenceOperator
export DiagonalOperator, CopyOperator

export RhsOperator
export BLFeval
export TLFeval


include("boundarydata.jl")
export BoundaryOperator
export AbstractBoundaryType, HomogeneousDirichletBoundary, InterpolateDirichletBoundary, BestapproxDirichletBoundary


include("globalconstraints.jl")
export AbstractGlobalConstraint
export FixedIntegralMean
export CombineDofs


include("pdedescription.jl")
export PDEDescription
export add_unknown!
export add_operator!
export add_rhsdata!
export add_boundarydata!
export add_constraint!


include("solvers.jl")
export solve!, assemble!
export TimeControlSolver, advance!

export AbstractTimeIntegrationRule
export BackwardEuler


include("pdeprototypes.jl")
export CompressibleNavierStokesProblem
export IncompressibleNavierStokesProblem
export LinearElasticityProblem
export PoissonProblem
export L2BestapproximationProblem
export H1BestapproximationProblem


end #module