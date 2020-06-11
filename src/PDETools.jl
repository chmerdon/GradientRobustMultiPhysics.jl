module PDETools

using FEAssembly
using FiniteElements
using ExtendableGrids
using FEXGrid
using ExtendableSparse
using SparseArrays
using BenchmarkTools
using LinearAlgebra

# type to steer when a PDE block is (re)assembled
abstract type AbstractAssemblyTrigger end
abstract type AssemblyNever <: AbstractAssemblyTrigger end    # is never assembled
abstract type AssemblyInitial <: AbstractAssemblyTrigger end    # is only assembled in initial assembly
abstract type AssemblyEachTimeStep <: AssemblyInitial end       # is (re)assembled in each timestep
abstract type AssemblyAlways <: AssemblyEachTimeStep end        # is always (re)assembled
abstract type AssemblyFinal <: AbstractAssemblyTrigger end       # is only assembled after solving



include("PDETools_PDEOperators.jl")
export AbstractPDEOperator

export BackwardEulerTimeDerivative

export AbstractBilinearForm
export StiffnessOperator, LaplaceOperator, HookStiffnessOperator2D, HookStiffnessOperator1D, ReactionOperator

export DiagonalOperator, CopyOperator
export ConvectionOperator
export LagrangeMultiplier
export FVUpwindDivergenceOperator

export RhsOperator


include("PDETools_PDEBoundaryData.jl")
export BoundaryOperator
export AbstractBoundaryType, HomogeneousDirichletBoundary, InterpolateDirichletBoundary, BestapproxDirichletBoundary


include("PDETools_PDEGlobalConstraints.jl")
export AbstractGlobalConstraint
export FixedIntegralMean
export CombineDofs


include("PDETools_PDEDescription.jl")
export PDEDescription


include("PDETools_PDESolver.jl")
export TimeControlSolver, advance
export assemble!, solve!

export AbstractTimeIntegrationRule
export BackwardEuler


include("PDETools_PDEProtoTypes.jl")
export CompressibleNavierStokesProblem
export IncompressibleNavierStokesProblem
export LinearElasticityProblem
export PoissonProblem
export L2BestapproximationProblem
export H1BestapproximationProblem

end # module