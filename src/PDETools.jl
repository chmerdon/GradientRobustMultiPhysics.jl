module PDETools

using FEAssembly
using FiniteElements
using ExtendableGrids
using FEXGrid
using BenchmarkTools

include("PDETools_PDEDescription.jl")
export AbstractPDEOperator, LaplaceOperator, ConvectionOperator, RhsOperator, BoundaryOperator
export LagrangeMultiplier
export AbstractBoundaryType, HomogeneousDirichletBoundary, InterpolateDirichletBoundary, BestapproxDirichletBoundary, NeumannBoundary
export PDEDescription

include("PDETools_PDESolver.jl")
export assemble!, solve!

end # module