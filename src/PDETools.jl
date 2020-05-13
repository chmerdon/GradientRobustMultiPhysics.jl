module PDETools

using FEAssembly
using FiniteElements
using ExtendableGrids
using FEXGrid
using BenchmarkTools

include("PDETools_PDEDescription.jl")
export AbstractPDEOperator, LaplaceOperator, ReactionOperator, ConvectionOperator, RhsOperator, BoundaryOperator
export LagrangeMultiplier
export AbstractBoundaryType, HomogeneousDirichletBoundary, InterpolateDirichletBoundary, BestapproxDirichletBoundary, NeumannBoundary
export AbstractGlobalConstraint, FixedIntegralMean
export PDEDescription

include("PDETools_PDESolver.jl")
export assemble!, solve!

end # module