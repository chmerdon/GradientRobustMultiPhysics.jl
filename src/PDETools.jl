module PDETools

using FEAssembly
using FiniteElements
using ExtendableGrids
using FEXGrid
using ExtendableSparse
using SparseArrays
using BenchmarkTools

include("PDETools_PDEDescription.jl")
export AbstractPDEOperator
export DiagonalOperator
export StiffnessOperator, LaplaceOperator, HookStiffnessOperator2D, HookStiffnessOperator1D
export ReactionOperator
export ConvectionOperator
export RhsOperator
export BoundaryOperator
export LagrangeMultiplier
export AbstractBoundaryType, HomogeneousDirichletBoundary, InterpolateDirichletBoundary, BestapproxDirichletBoundary, NeumannBoundary
export AbstractGlobalConstraint, FixedIntegralMean, CombineDofs
export PDEDescription

include("PDETools_PDESolver.jl")
export assemble!, solve!

include("PDETools_PDEProtoTypes.jl")
export IncompressibleNavierStokesProblem
export LinearElasticityProblem
export PoissonProblem
export L2BestapproximationProblem
export H1BestapproximationProblem

end # module