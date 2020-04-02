module FESolvePoissonTests

using FESolvePoisson
using FESolveCommon
using Triangulate
using Grid
using Quadrature
using FiniteElements




include("../src/PROBLEMdefinitions/GRID_unitinterval.jl")
include("../src/PROBLEMdefinitions/POISSON_1D_polynomials.jl");
  
function TestPoissonSolver1D(fem::String, order::Int)
  println("Testing Poisson solver in 1D for fem=",fem);
  polynomial_coefficients = ones(Float64,order+1)
  PD, exact_solution! = getProblemData(polynomial_coefficients)
  grid = gridgen_unitinterval(0.1)
  FE = FiniteElements.string2FE(fem, grid, 1, 1)
  val4dofs = FiniteElements.createFEVector(FE);
  residual = solvePoissonProblem!(val4dofs,PD,FE);
  println("solver residual = " * string(residual));
  error = sqrt(FESolveCommon.assemble_operator!(FESolveCommon.DOMAIN_L2_FplusA,exact_solution!,FE,val4dofs; degreeF = order, factorA = -1.0))
  println("L2_error = " * string(error));
  return abs(error) < eps(100.0)
end


include("../src/PROBLEMdefinitions/GRID_unitsquare.jl")
include("../src/PROBLEMdefinitions/POISSON_2D_polynomials.jl");

function TestPoissonSolver2D(fem::String, order::Int)
  println("Testing Poisson solver in 2D for fem=",fem);
  polynomial_coefficients = ones(Float64,2,order+1)
  PD, exact_solution! = getProblemData(polynomial_coefficients, 1.0)
  FESolvePoisson.show(PD)
  grid = gridgen_unitsquare(0.1)
  FE = FiniteElements.string2FE(fem, grid, 2, 1)
  val4dofs = FiniteElements.createFEVector(FE);
  residual = solvePoissonProblem!(val4dofs,PD,FE);
  println("solver residual = " * string(residual));
  error = sqrt(FESolveCommon.assemble_operator!(FESolveCommon.DOMAIN_L2_FplusA,exact_solution!,FE,val4dofs; degreeF = order, factorA = -1.0))
  println("L2_error = " * string(error));
  return abs(error) < eps(100.0)
end


end
