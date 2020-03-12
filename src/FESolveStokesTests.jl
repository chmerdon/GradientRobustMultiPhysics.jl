module FESolveStokesTests

using SparseArrays
using LinearAlgebra
using FESolveStokes
using FESolveCommon
using Triangulate
using Grid
using Quadrature
using FiniteElements

include("../src/PROBLEMdefinitions/GRID_unitsquare.jl")
include("../src/PROBLEMdefinitions/STOKES_2D_polynomials.jl");

function TestStokesSolver2D(fem_velocity::String, fem_pressure::String, order::Int)
  println("Testing Stokes solver in 2D for fem=" * fem_velocity * "/" * fem_pressure);
  polynomial_coefficients = ones(Float64,2,order+1)
  PD, exact_velocity!, exact_pressure! = getProblemData(order, 1.0, false, 4);
  FESolveStokes.show(PD)
  grid = gridgen_unitsquare(0.1)
  FE_velocity = FiniteElements.string2FE(fem_velocity, grid, 2, 2)
  FE_pressure = FiniteElements.string2FE(fem_pressure, grid, 2, 1)
  val4dofs = FiniteElements.createFEVector(FE_velocity)
  ndofs_velocity = FiniteElements.get_ndofs(FE_velocity)
  ndofs_pressure = FiniteElements.get_ndofs(FE_pressure)
  val4dofs = zeros(Float64,ndofs_velocity+ndofs_pressure)
  residual = solveStokesProblem!(val4dofs,PD,FE_velocity,FE_pressure);
  println("solver residual = " * string(residual));
  integral4cells = zeros(size(grid.nodes4cells,1),2);
  integrate!(integral4cells,eval_L2_interpolation_error!(exact_velocity!, val4dofs[1:ndofs_velocity], FE_velocity),grid,2*order,2);
  error_velocity = sqrt(sum(integral4cells));
  println("L2_error_velocity = " * string(error_velocity));
  integral4cells2 = zeros(size(grid.nodes4cells,1),1);
  integrate!(integral4cells2,eval_L2_interpolation_error!(exact_pressure!, val4dofs[ndofs_velocity+1:end], FE_pressure),grid,2*order,1);
  error_pressure = sqrt(sum(integral4cells2));
  println("L2_error_pressure = " * string(error_pressure));
  
  return abs(error_velocity+error_pressure) < eps(1000.0)
end


end
