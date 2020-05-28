module PDETools

using FEAssembly
using FiniteElements
using ExtendableGrids
using FEXGrid
using BenchmarkTools

include("PDETools_PDEDescription.jl")
export AbstractPDEOperator, StiffnessOperator, ReactionOperator, ConvectionOperator, RhsOperator, BoundaryOperator
export LagrangeMultiplier
export AbstractBoundaryType, HomogeneousDirichletBoundary, InterpolateDirichletBoundary, BestapproxDirichletBoundary, NeumannBoundary
export AbstractGlobalConstraint, FixedIntegralMean
export PDEDescription

include("PDETools_PDESolver.jl")
export assemble!, solve!

include("PDETools_PDEProtoTypes.jl")
export IncompressibleNavierStokesProblem
export LinearElasticityProblem
export L2BestapproximationProblem
export H1BestapproximationProblem

end # module