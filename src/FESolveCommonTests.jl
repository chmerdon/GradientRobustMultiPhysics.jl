module FESolveCommonTests

using FESolvePoisson
using FESolveCommon
using Grid
using Triangulate
using Quadrature
using FiniteElements


include("../src/PROBLEMdefinitions/GRID_unitinterval.jl")
include("../src/PROBLEMdefinitions/POISSON_1D_polynomials.jl");

  
function TestInterpolation1D(fem::String, order::Int)
  println("Testing FE interpolation in 1D for fem=",fem);
  polynomial_coefficients = ones(Float64,order+1)
  PD, exact_solution! = getProblemData(polynomial_coefficients)
  grid = gridgen_unitinterval(0.1)
  FE = FiniteElements.string2FE(fem, grid, 1, 1)
  val4dofs = FiniteElements.createFEVector(FE);
  @time computeFEInterpolation!(val4dofs, exact_solution!, FE);
  integral4cells = zeros(size(grid.nodes4cells, 1), 1);
  integrate!(integral4cells, eval_L2_interpolation_error!(exact_solution!, val4dofs, FE), grid, 2*order+1);
  integral = sqrt(sum(integral4cells));
  println("L2_interpolation_error = " * string(integral));
  return abs(integral) < eps(10.0)
end

function TestBestApproximation1D(norm::String, fem::String, order::Int)
  println("Testing " * norm * " bestapproximation in 1D for fem=",fem);
  polynomial_coefficients = ones(Float64,order+1)
  PD, exact_solution!, exact_gradient! = getProblemData(polynomial_coefficients)
  grid = gridgen_unitinterval(0.1)
  FE = FiniteElements.string2FE(fem, grid, 1, 1)
  val4dofs = FiniteElements.createFEVector(FE);
  if norm == "L2"
      computeBestApproximation!(val4dofs,"L2",exact_solution!,exact_solution!,FE,order);
  elseif norm == "H1"
      computeBestApproximation!(val4dofs,"H1",exact_gradient!,exact_solution!,FE,order);
  end
  integral4cells = zeros(size(grid.nodes4cells, 1), 1);
  integrate!(integral4cells, eval_L2_interpolation_error!(exact_solution!, val4dofs, FE), grid, 2*order+1);
  integral = sqrt(sum(integral4cells));
  println("L2_interpolation_error = " * string(integral));
  return abs(integral) < eps(100.0)
end



include("../src/PROBLEMdefinitions/GRID_unitsquare.jl")
include("../src/PROBLEMdefinitions/POISSON_2D_polynomials.jl");


function TestInterpolation2D(fem::String, order::Int)
  println("Testing FE interpolation in 2D for fem=",fem);
  polynomial_coefficients = ones(Float64,2,order+1)
  PD, exact_solution! = getProblemData(polynomial_coefficients, 1.0)
  grid = gridgen_unitsquare(0.1)
  FE = FiniteElements.string2FE(fem, grid, 2, 1)
  val4dofs = FiniteElements.createFEVector(FE);
  @time computeFEInterpolation!(val4dofs, exact_solution!, FE);
  integral4cells = zeros(size(grid.nodes4cells, 1), 2);
  integrate!(integral4cells, eval_L2_interpolation_error!(exact_solution!, val4dofs, FE), grid, 2*order+1, 2);
  integral = sqrt(sum(integral4cells));
  println("L2_interpolation_error = " * string(integral));
  return abs(integral) < eps(10.0)
end

function TestBestApproximation2D(norm::String, fem::String, order::Int)
  println("Testing " * norm * " bestapproximation in 2D for fem=",fem);
  polynomial_coefficients = ones(Float64,2,order+1)
  PD, exact_solution!, exact_gradient! = getProblemData(polynomial_coefficients, 1.0)
  grid = gridgen_unitsquare(0.1)
  FE = FiniteElements.string2FE(fem, grid, 2, 1)
  val4dofs = FiniteElements.createFEVector(FE);
  if norm == "L2"
      computeBestApproximation!(val4dofs,"L2",exact_solution!,exact_solution!,FE,order);
  elseif norm == "H1"
      computeBestApproximation!(val4dofs,"H1",exact_gradient!,exact_solution!,FE,order);
  end
  integral4cells = zeros(size(grid.nodes4cells, 1), 2);
  integrate!(integral4cells, eval_L2_interpolation_error!(exact_solution!, val4dofs, FE), grid, 2*order+1, 2);
  integral = sqrt(sum(integral4cells));
  println("L2_interpolation_error = " * string(integral));
  return abs(integral) < eps(100.0)
end

end
