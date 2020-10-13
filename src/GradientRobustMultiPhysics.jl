module GradientRobustMultiPhysics

using ExtendableGrids
export Edge1D, Triangle2D, Parallelogram2D, Tetrahedron3D, Parallelepiped3D, num_sources, simplexgrid, VectorOfConstants
using ExtendableSparse
using SuiteSparse
using SparseArrays
using BenchmarkTools
using IterativeSolvers
using DiffResults
using LinearAlgebra
using ForwardDiff
using DocStringExtensions
using Printf


include("junctions.jl");
export AbstractAssemblyType
export ON_CELLS, ON_FACES, ON_IFACES, ON_BFACES, ON_EDGES, ON_BEDGES
export GridComponentNodes4AssemblyType
export GridComponentVolumes4AssemblyType
export GridComponentGeometries4AssemblyType
export GridComponentRegions4AssemblyType


include("l2gtransformations.jl");
export L2GTransformer, update!, eval!, mapderiv!, piola!

include("shape_specs.jl")
export refcoords_for_geometry
export nnodes_for_geometry
export nfaces_for_geometry
export nedges_for_geometry
export face_enum_rule
export edge_enum_rule
export facetype_of_cellface
export edgetype_of_celledge
export celledges_for_cellface
export Volume4ElemType
export Normal4ElemType!
export Tangent4ElemType!


include("gridstuff.jl")
export Coordinates
export CellNodes, CellGeometries, CellVolumes, CellRegions, CellFaces, CellEdges, CellFaceSigns, CellEdgeSigns
export FaceNodes, FaceGeometries, FaceVolumes, FaceRegions, FaceCells, FaceEdges, FaceNormals
export EdgeNodes, EdgeGeometries, EdgeVolumes, EdgeRegions, EdgeCells, EdgeTangents
export BFaces, BFaceCellPos, BFaceVolumes
export BEdgeNodes, BEdges, BEdgeVolumes, BEdgeGeometries
export unique, UniqueCellGeometries, UniqueFaceGeometries, UniqueBFaceGeometries, UniqueEdgeGeometries, UniqueBEdgeGeometries

include("serialadjacency.jl")
export SerialVariableTargetAdjacency


include("meshrefinements.jl")
export split_grid_into
export uniform_refine
export barycentric_refine


include("quadrature.jl")
export QuadratureRule
export VertexRule
export integrate!, integrate


include("finiteelements.jl")
export DofMap, CellDofs, FaceDofs, EdgeDofs, BFaceDofs, BEdgeDofs
export AbstractFiniteElement
export FESpace

export AbstractH1FiniteElement
export H1P1, H1P2, H1MINI, H1CR
export L2P0, L2P1 # will be AbstractL2FiniteElement one day

export AbstractH1FiniteElementWithCoefficients
export H1BR

export AbstractHdivFiniteElement
export HDIVRT0, HDIVBDM1, HDIVRT1

export AbstractHcurlFiniteElement
export HCURLN0

export get_ncomponents
export reconstruct!


include("fevector.jl");
export FEVectorBlock, FEVector
export fill!, addblock!


include("fematrix.jl");
export FEMatrixBlock, FEMatrix
export fill!, addblock!, addblock_matmul!, lrmatmul


include("functionoperators.jl")
export AbstractFunctionOperator
export Identity, IdentityComponent
export ReconstructionIdentity, ReconstructionIdentityDisc
export ReconstructionGradient, ReconstructionGradientDisc
export ReconstructionDivergence
export NormalFlux, TangentFlux
export Gradient, GradientDisc
export SymmetricGradient, TangentialGradient
export Divergence, ReconstructionDivergence
export CurlScalar, Curl2D, Curl3D
export Laplacian, Hessian
export Trace, Deviator
export NeededDerivatives4Operator, QuadratureOrderShift4Operator
export Dofmap4AssemblyType, DofitemAT4Operator
export DefaultDirichletBoundaryOperator4FE
export DefaultName4Operator

export DiscontinuityTreatment, Jump, Average
export IdentityDisc, GradientDisc


include("febasisevaluator.jl")
export FEBasisEvaluator, update!, eval!


include("interpolations.jl")
export interpolate! # must be defined separately by each FEdefinition
export nodevalues! # = P1interpolation, abstract averaging method that works for any element, but can be overwritten by FEdefinition to something simpler


include("actions.jl")
export AbstractAction
export DoNotChangeAction
export MultiplyScalarAction, MultiplyVectorAction, MultiplyMatrixAction
export ItemWiseMultiplyScalarAction
export RegionWiseMultiplyScalarAction, RegionWiseMultiplyVectorAction
export FunctionAction
export ItemWiseFunctionAction
export XFunctionAction
export ItemWiseXFunctionAction, RegionWiseXFunctionAction

include("accumvector.jl")
export AccumulatingVector

include("assemblypatterns.jl")
export AbstractAssemblyPattern
export ItemIntegrator
export LinearForm
export BilinearForm, SymmetricBilinearForm
export TrilinearForm
export MultilinearForm
export NonlinearForm
export assemble!, evaluate!, evaluate
export L2ErrorIntegrator


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

export RhsOperator
export BLFeval
export TLFeval
export MLFeval

export LHSoperator_also_modifies_RHS


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
export AbstractLinSolveType
export DirectUMFPACK, IterativeBigStabl_LUPC
export solve!, assemble!
export TimeControlSolver, advance!, advance_until_stationarity!, advance_until_time!

export AbstractTimeIntegrationRule
export BackwardEuler


include("pdeprototypes.jl")
export CompressibleNavierStokesProblem
export IncompressibleNavierStokesProblem
export LinearElasticityProblem
export PoissonProblem
export L2BestapproximationProblem
export H1BestapproximationProblem

include("commongrids.jl")
export reference_domain
export grid_unitcube
export grid_unitsquare, grid_unitsquare_mixedgeometries

include("writevtk.jl")
export writeVTK!

include("plots.jl")
export plot


end #module