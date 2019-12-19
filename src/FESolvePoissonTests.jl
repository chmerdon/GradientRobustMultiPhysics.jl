module FESolvePoissonTests

using SparseArrays
using LinearAlgebra
using FESolvePoisson
using FESolveCommon
using Grid
using Quadrature
using FiniteElements


function load_test_grid(nrefinements::Int = 1)
    # define grid
    coords4nodes_init = [0.0 0.0;
                        1.0 0.0;
                        1.0 1.0;
                        0.0 1.0;
                        0.5 0.5];
    nodes4cells_init = [1 2 5;
                        2 3 5;
                        3 4 5;
                        4 1 5];
               
    return Grid.Mesh(coords4nodes_init,nodes4cells_init,nrefinements);
end


function load_test_grid1D(nrefinements::Int = 0)
    # define grid
    coords4nodes_init = Array{Float64,2}([0.0 0.5 1.0]');
    nodes4cells_init = [1 2; 2 3];
    return Grid.Mesh(coords4nodes_init,nodes4cells_init,nrefinements);
end



  # define problem data
  # = linear function f(x,y) = x + y and its derivatives
  function volume_data1D!(result, x)
    result[1] = x[1] + 1
  end
  
  function volume_data_P1!(result, x)
    result[1] = x[1] + x[2]
  end
  function volume_data_P2!(result, x)
    result[1] = x[1]^2 + x[2]^2
  end
  
  function volume_data_gradient!(result,x)
    result = ones(Float64,size(x));
  end
  function volume_data_gradient_P2!(result,x)
    result[1] = 2*x[1];
    result[2] = 2*x[2];
  end
  
  function volume_data_laplacian_P2!(result,x)
    result[1] = -4;
  end
  function volume_data_laplacian_P1!(result,x)
    result[1] = 0;
  end
  
function TestPoissonSolver1D()
  grid = load_test_grid1D();
  println("Testing H1-Bestapproximation via Poisson solver in 1D...");
  FE = FiniteElements.get_P1FiniteElement(grid);
  val4dofs = zeros(FE.ndofs);
  residual = solvePoissonProblem!(val4dofs,volume_data_laplacian_P1!,volume_data1D!,grid,FE,1);
  println("solver residual = " * string(residual));
  integral4cells = zeros(size(grid.nodes4cells,1),1);
  integrate!(integral4cells,eval_interpolation_error!(volume_data1D!, val4dofs, FE),grid,1);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));
  return abs(integral) < eps(10.0)
end

function TestPoissonSolver2DP1()
  grid = load_test_grid(2);
  println("Testing H1-Bestapproximation via Poisson solver in 2D for P1-FEM...");
  FE = FiniteElements.get_P1FiniteElement(grid,false);
  val4dofs = zeros(FE.ndofs);
  residual = solvePoissonProblem!(val4dofs,volume_data_laplacian_P1!,volume_data_P1!,grid,FE,1);
  println("solver residual = " * string(residual));
  integral4cells = zeros(size(grid.nodes4cells,1),1);
  integrate!(integral4cells,eval_interpolation_error!(volume_data_P1!, val4dofs, FE),grid,1);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));
  return abs(integral) < eps(10.0)
end


function TestPoissonSolver2DCR()
  grid = load_test_grid();
  println("Testing H1-Bestapproximation via Poisson solver in 2D for CR-FEM...");
  FE = FiniteElements.get_CRFiniteElement(grid);
  val4dofs = zeros(FE.ndofs);
  residual = solvePoissonProblem!(val4dofs,volume_data_laplacian_P1!,volume_data_P1!,grid,FE,1);
  println("solver residual = " * string(residual));
  integral4cells = zeros(size(grid.nodes4cells,1),1);
  integrate!(integral4cells,eval_interpolation_error!(volume_data_P1!, val4dofs, FE),grid,1);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));
  return abs(integral) < eps(10.0)
end


function TestPoissonSolver2DP2()
  grid = load_test_grid(2);
  println("Testing H1-Bestapproximation via Poisson solver in 2D for P2-FEM...");
  FE = FiniteElements.get_P2FiniteElement(grid,true);
  val4dofs = zeros(FE.ndofs);
  residual = solvePoissonProblem!(val4dofs,volume_data_laplacian_P2!,volume_data_P2!,grid,FE,3);
  println("solver residual = " * string(residual));
  integral4cells = zeros(size(grid.nodes4cells,1),1);
  integrate!(integral4cells,eval_interpolation_error!(volume_data_P2!, val4dofs, FE),grid,4);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));
  return abs(integral) < eps(10.0)
end


end
