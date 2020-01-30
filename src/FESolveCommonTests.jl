module FESolveCommonTests

using ExtendableSparse
using SparseArrays
using LinearAlgebra
using FESolveCommon
using Grid
using Quadrature
using FiniteElements


function load_test_grid(nrefinements::Int = 1)
    # define grid
    coords4nodes_init = [0.0 0.0;
                        1.0 0.0;
                        1.0 1.0;
                        0.1 1.0;
                        0.5 0.6];
    nodes4cells_init = [1 2 5;
                        2 3 5;
                        3 4 5;
                        4 1 5];
               
    return Grid.Mesh{Float64}(coords4nodes_init,nodes4cells_init,Grid.ElemType2DTriangle(),nrefinements);
end


function load_test_grid1D(nrefinements::Int = 0)
    # define grid
    coords4nodes_init = Array{Float64,2}([0.0 0.5 1.0]');
    nodes4cells_init = [1 2; 2 3];
    return Grid.Mesh{Float64}(coords4nodes_init,nodes4cells_init,Grid.ElemType1DInterval(),nrefinements);
end



  # define problem data
  # = linear function f(x,y) = x + y and its derivatives
  function volume_data1D!(result, x)
    result[1] = x[1] .+ 1
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
    for i in eachindex(result)
      @inbounds result[i,1] = 2*x[i, 1];
      @inbounds result[i,2] = 2*x[i, 2];
    end
  end
  
  function volume_data_laplacian_P2!(result,x)
    result[:] = ones(Float64,size(result));
    result .*= -4;
  end
  function volume_data_laplacian_P1!(result,x)
    result[:] = zeros(Float64,size(result));
  end

  

function TestInterpolation1D()
  grid = load_test_grid1D();
  
  # compute volume4cells
  Grid.ensure_volume4cells!(grid);  
  println("Testing P1 Interpolation in 1D...");
  FE = FiniteElements.getP1FiniteElement(grid,1);
  val4dofs = FiniteElements.createFEVector(FE);
  @time computeFEInterpolation!(val4dofs, volume_data1D!, FE);
  integral4cells = zeros(size(grid.nodes4cells, 1), 1);
  integrate!(integral4cells, eval_L2_interpolation_error!(volume_data1D!, val4dofs, FE), grid, 1);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));

  return abs(integral) < eps(10.0)
end


function TestL2BestApproximation1D()
  grid = load_test_grid1D();
  println("Testing L2-Bestapproximation in 1D...");
  FE = FiniteElements.getP1FiniteElement(grid,1);
  val4dofs = FiniteElements.createFEVector(FE);
  computeBestApproximation!(val4dofs,"L2",volume_data1D!,volume_data1D!,FE,2);
  integral4cells = zeros(size(grid.nodes4cells,1),1);
  integrate!(integral4cells, eval_L2_interpolation_error!(volume_data1D!, val4dofs, FE), grid, 1);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));
  return abs(integral) < eps(10.0)
end

function TestH1BestApproximation1D()
  grid = load_test_grid1D();
  println("Testing H1-Bestapproximation in 1D...");
  FE = FiniteElements.getP1FiniteElement(grid,1);
  val4dofs = FiniteElements.createFEVector(FE);
  computeBestApproximation!(val4dofs,"H1",volume_data_gradient!,volume_data1D!,FE,2);
  integral4cells = zeros(size(grid.nodes4cells,1),1);
  integrate!(integral4cells, eval_L2_interpolation_error!(volume_data1D!, val4dofs, FE), grid, 1);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));
  return abs(integral) < eps(10.0)
end

function TestInterpolation2D()
  grid = load_test_grid(0);
  Grid.ensure_volume4cells!(grid);  
  println("Testing P1 Interpolation in 2D...");
  FE = FiniteElements.getP1FiniteElement(grid,1);
  val4dofs = FiniteElements.createFEVector(FE);
  computeFEInterpolation!(val4dofs, volume_data_P1!, FE);
  integral4cells = zeros(size(grid.nodes4cells, 1), 1);
  integrate!(integral4cells, eval_L2_interpolation_error!(volume_data_P1!, val4dofs, FE), grid, 1);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));
  return abs(integral) < eps(10.0)
end

function TestL2BestApproximation2DP1()
  grid = load_test_grid();
  println("Testing L2-Bestapproximation in 2D for P1-FEM...");
  FE = FiniteElements.getP1FiniteElement(grid,1);
  val4dofs = FiniteElements.createFEVector(FE);
  computeBestApproximation!(val4dofs,"L2",volume_data_P1!,volume_data_P1!,FE,2);
  integral4cells = zeros(size(grid.nodes4cells,1),1);
  integrate!(integral4cells, eval_L2_interpolation_error!(volume_data_P1!, val4dofs, FE), grid, 1);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));
  return abs(integral) < eps(10.0)
end

function TestL2BestApproximation2DP2()
  grid = load_test_grid();
  println("Testing L2-Bestapproximation in 2D for P2-FEM...");
  FE = FiniteElements.getP2FiniteElement(grid,1);
  val4dofs = FiniteElements.createFEVector(FE);
  computeBestApproximation!(val4dofs,"L2",volume_data_P2!,volume_data_P2!,FE,4);
  integral4cells = zeros(size(grid.nodes4cells,1),1);
  integrate!(integral4cells, eval_L2_interpolation_error!(volume_data_P2!, val4dofs, FE), grid, 4);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));
  return abs(integral) < eps(10.0)
end


function TestL2BestApproximation2DCR()
  grid = load_test_grid();
  println("Testing L2-Bestapproximation in 2D for CR-FEM...");
  FE = FiniteElements.getCRFiniteElement(grid,2,1);
  val4dofs = FiniteElements.createFEVector(FE);
  computeBestApproximation!(val4dofs,"L2",volume_data_P1!,volume_data_P1!,FE,2);
  integral4cells = zeros(size(grid.nodes4cells,1),1);
  integrate!(integral4cells, eval_L2_interpolation_error!(volume_data_P1!, val4dofs, FE), grid, 1);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));
  return abs(integral) < eps(10.0)
end


function TestH1BestApproximation2D()
  grid = load_test_grid();
  println("Testing H1-Bestapproximation in 2D...");
  FE = FiniteElements.getP1FiniteElement(grid,1);
  val4dofs = FiniteElements.createFEVector(FE);
  computeBestApproximation!(val4dofs,"H1",volume_data_gradient!,volume_data_P1!,FE,2);
  integral4cells = zeros(size(grid.nodes4cells,1),1);
  integrate!(integral4cells, eval_L2_interpolation_error!(volume_data_P1!, val4dofs, FE), grid, 1);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));
  return abs(integral) < eps(10.0)
end

end
