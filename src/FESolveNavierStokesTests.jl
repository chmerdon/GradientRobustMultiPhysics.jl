module FESolveNavierStokesTests

using SparseArrays
using LinearAlgebra
using FESolveStokes
using FESolveNavierStokes
using FESolveCommon
using Triangulate
using Grid
using Quadrature
using FiniteElements

include("../src/PROBLEMdefinitions/GRID_unitsquare.jl")
include("../src/PROBLEMdefinitions/STOKES_2D_polynomials.jl");

function TestNavierStokesSolver2D(fem_velocity::String, fem_pressure::String, order::Int)
  println("Testing Navier-Stokes solver in 2D for fem=" * fem_velocity * "/" * fem_pressure);
  polynomial_coefficients = ones(Float64,2,order+1)
  PD, exact_velocity!, exact_pressure! = getProblemData(order, 1.0, true, 4);
  FESolveStokes.show(PD)
  grid = gridgen_unitsquare(0.1)
  FE_velocity = FiniteElements.string2FE(fem_velocity, grid, 2, 2)
  FE_pressure = FiniteElements.string2FE(fem_pressure, grid, 2, 1)
  val4dofs = FiniteElements.createFEVector(FE_velocity)
  ndofs_velocity = FiniteElements.get_ndofs(FE_velocity)
  ndofs_pressure = FiniteElements.get_ndofs(FE_pressure)
  val4dofs = zeros(Float64,ndofs_velocity+ndofs_pressure)
  residual = solveNavierStokesProblem!(val4dofs,PD,FE_velocity,FE_pressure,0,20);
  println("solver residual = " * string(residual));
  error_velocity = sqrt(FESolveCommon.assemble_operator!(FESolveCommon.DOMAIN_L2_FplusA,exact_velocity!,FE_velocity,val4dofs; degreeF = order, factorA = -1.0))
  println("L2_error_velocity = " * string(error_velocity));
  error_pressure = sqrt(FESolveCommon.assemble_operator!(FESolveCommon.DOMAIN_L2_FplusA,exact_pressure!,FE_pressure,val4dofs[ndofs_velocity+1:end]; degreeF = order, factorA = -1.0))
  println("L2_error_pressure = " * string(error_pressure));
  return abs(error_pressure+error_velocity) < eps(10000.0)
end


end
