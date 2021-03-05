module GradientRobustMultiPhysics

using ExtendableGrids
using GridVisualize
export Edge1D, Triangle2D, Parallelogram2D, Tetrahedron3D, Parallelepiped3D, num_sources, simplexgrid, VectorOfConstants
using ExtendableSparse
using SuiteSparse
using SparseArrays
using StaticArrays
using BenchmarkTools
using IterativeSolvers
using DiffResults
using LinearAlgebra
using ForwardDiff
using DocStringExtensions
using Printf

## stuff that may go to ExtendableGrids

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
export CellNodes, CellGeometries, CellVolumes, CellRegions, CellFaces, CellEdges, CellFaceSigns, CellFaceOrientations, CellEdgeSigns
export FaceNodes, FaceGeometries, FaceVolumes, FaceRegions, FaceCells, FaceEdges, FaceNormals
export EdgeNodes, EdgeGeometries, EdgeVolumes, EdgeRegions, EdgeCells, EdgeTangents
export NodePatchGroups
export BFaces, BFaceCellPos, BFaceVolumes
export BEdgeNodes, BEdges, BEdgeVolumes, BEdgeGeometries
export unique, UniqueCellGeometries, UniqueFaceGeometries, UniqueBFaceGeometries, UniqueEdgeGeometries, UniqueBEdgeGeometries
export GridComponent4TypeProperty
export ITEMTYPE_CELL, ITEMTYPE_FACE, ITEMTYPE_BFACE, ITEMTYPE_EDGE, ITEMTYPE_BEDGE
export PROPERTY_NODES, PROPERTY_REGION, PROPERTY_VOLUME, PROPERTY_UNIQUEGEOMETRY, PROPERTY_GEOMETRY
export get_facegrid, get_bfacegrid, get_edgegrid

include("meshrefinements.jl")
export split_grid_into
export uniform_refine
export barycentric_refine

include("adaptive_meshrefinements.jl")
export bulk_mark
export RGB_refine

include("serialadjacency.jl")
export SerialVariableTargetAdjacency




include("userdata.jl")
export AbstractDataFunction, AbstractActionKernel, AbstractExtendedDataFunction
export UserData, ActionKernel, NLActionKernel, DataFunction, ExtendedDataFunction, eval!
export is_timedependent, is_regiondependent, is_itemdependent

include("assemblytypes.jl");
export AbstractAssemblyType
export AT_NODES, ON_CELLS, ON_FACES, ON_IFACES, ON_BFACES, ON_EDGES, ON_BEDGES
export ItemType4AssemblyType
export GridComponentNodes4AssemblyType
export GridComponentVolumes4AssemblyType
export GridComponentGeometries4AssemblyType
export GridComponentRegions4AssemblyType

include("l2gtransformations.jl");
export L2GTransformer, update!, eval!, mapderiv!, piola!

include("quadrature.jl")
export QuadratureRule
export VertexRule
export integrate!, integrate

include("functionoperators.jl")
export AbstractFunctionOperator
export Identity, IdentityComponent
export ReconstructionIdentity, ReconstructionIdentityDisc
export ReconstructionGradient, ReconstructionGradientDisc
export ReconstructionDivergence
export ReconstructionNormalFlux
export NormalFlux, NormalFluxDisc, TangentFlux, TangentFluxDisc
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


include("finiteelements.jl")
export DofMap, CellDofs, FaceDofs, EdgeDofs, BFaceDofs, BEdgeDofs
export AbstractFiniteElement
export FESpace

export AbstractH1FiniteElement
export H1P0, H1P1, H1P2, H1P2B, H1MINI, H1CR

export AbstractH1FiniteElementWithCoefficients
export H1BR

export AbstractHdivFiniteElement
export HDIVRT0, HDIVBDM1, HDIVRT1

export AbstractHcurlFiniteElement
export HCURLN0

export get_assemblytype
export get_polynomialorder, get_ndofs, get_ndofs_all
export get_ncomponents, get_edim
export get_basis, get_coefficients, get_basissubset
export reconstruct!

export interpolate! # must be defined separately by each FEdefinition
export nodevalues! # = P1interpolation, abstract averaging method that works for any element, but can be overwritten by FEdefinition to something simpler

export FEVectorBlock, FEVector
export FEMatrixBlock, FEMatrix, _addnz
export fill!, addblock!, addblock_matmul!, lrmatmul

export get_reconstruction_matrix


include("febasisevaluator.jl")
export FEBasisEvaluator, update!, eval!


include("actions.jl")
export AbstractAction, Action, MultiplyScalarAction, DoNotChangeAction


include("accumvector.jl")
export AccumulatingVector


include("assemblypatterns.jl")
export AssemblyPatternType, AssemblyPreparations
export ItemIntegrator, L2ErrorIntegrator, L2NormIntegrator, L2DifferenceIntegrator
export LinearForm
export BilinearForm, SymmetricBilinearForm
export TrilinearForm
export MultilinearForm
export NonlinearForm
export prepare_assembly!
export assemble!, evaluate!, evaluate


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
export BLF2RHS
export TLF2RHS
export MLF2RHS

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
export AbstractLinSolveType, LinearSystem
export DirectUMFPACK, IterativeBigStabl_LUPC
export solve!, assemble!
export TimeControlSolver, advance!, advance_until_stationarity!, advance_until_time!
export show_statistics
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
export grid_lshape
export grid_unitsquare, grid_unitsquare_mixedgeometries

include("dataexport.jl")
export writeVTK!, writeCSV!

include("plots.jl")
export plot


end #module
