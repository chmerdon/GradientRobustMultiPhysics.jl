module FESolveCommonTests

using FESolvePoisson
using FESolveCommon
using Grid
using Triangulate
using Quadrature
using FiniteElements


include("../src/PROBLEMdefinitions/GRID_unitinterval.jl")
include("../src/PROBLEMdefinitions/POISSON_1D_polynomials.jl");

  
function TestInterpolation1D(fem::String, order::Int, ncomponents::Int = 1)
  println("Testing FE interpolation in 1D for fem=" * fem * " with ncomponents=", ncomponents);
  polynomial_coefficients = ones(Float64,order+1)
  PD, exact_solution! = getProblemData(polynomial_coefficients)
  
  # add more components
  function stretched_exact_solution!(result,x)
      exact_solution!(result,x)
      for k=2 : ncomponents
          result[k] = result[1] - k
      end  
  end    

  grid = gridgen_unitinterval(0.1)
  FE = FiniteElements.string2FE(fem, grid, 1, ncomponents)
  val4dofs = FiniteElements.createFEVector(FE);
  @time computeFEInterpolation!(val4dofs, stretched_exact_solution!, FE);
  error = sqrt(FESolveCommon.assemble_operator!(FESolveCommon.DOMAIN_L2_FplusA,stretched_exact_solution!,FE,val4dofs; degreeF = order, factorA = -1.0))
  println("L2_interpolation_error = " * string(error));
  return abs(error) < eps(10.0)
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
  error = sqrt(FESolveCommon.assemble_operator!(FESolveCommon.DOMAIN_L2_FplusA,exact_solution!,FE,val4dofs; degreeF = order, factorA = -1.0))
  println("L2_interpolation_error = " * string(error));
  return abs(error) < eps(100.0)
end



include("../src/PROBLEMdefinitions/GRID_unitsquare.jl")
include("../src/PROBLEMdefinitions/POISSON_2D_polynomials.jl");


function TestInterpolation2D(fem::String, order::Int, ncomponents::Int = 1)
  println("Testing FE interpolation in 2D for fem=" * fem * " with ncomponents=", ncomponents);
  polynomial_coefficients = ones(Float64,2,order+1)
  PD, exact_solution! = getProblemData(polynomial_coefficients, 1.0)

  # add more components
  function stretched_exact_solution!(result,x)
      exact_solution!(result,x)
      for k=2 : ncomponents
          result[k] = result[1] - k
      end  
  end    

  grid = gridgen_unitsquare(0.1)
  FE = FiniteElements.string2FE(fem, grid, 2, ncomponents)
  val4dofs = FiniteElements.createFEVector(FE);
  @time computeFEInterpolation!(val4dofs, stretched_exact_solution!, FE);
  error = sqrt(FESolveCommon.assemble_operator!(FESolveCommon.DOMAIN_L2_FplusA,stretched_exact_solution!,FE,val4dofs; degreeF = order, factorA = -1.0))
  println("L2_interpolation_error = " * string(error));
  return abs(error) < eps(10.0)
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
  error = sqrt(FESolveCommon.assemble_operator!(FESolveCommon.DOMAIN_L2_FplusA,exact_solution!,FE,val4dofs; degreeF = order, factorA = -1.0))
  println("L2_interpolation_error = " * string(error));
  return abs(error) < eps(1000.0)
end

end
