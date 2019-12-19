module FESolveCommonTests


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
  FE = FiniteElements.get_P1FiniteElement(grid);
  val4dofs = zeros(FE.ndofs);
  @time computeFEInterpolation!(val4dofs, volume_data1D!, grid, FE);
  integral4cells = zeros(size(grid.nodes4cells, 1), 1);
  integrate!(integral4cells, eval_interpolation_error!(volume_data1D!, val4dofs, FE), grid, 1);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));

  return abs(integral) < eps(10.0)
end


function TestL2BestApproximation1D()
  grid = load_test_grid1D();
  println("Testing L2-Bestapproximation in 1D...");
  FE = FiniteElements.get_P1FiniteElement(grid);
  val4dofs = zeros(FE.ndofs);
  computeBestApproximation!(val4dofs,"L2",volume_data1D!,volume_data1D!,grid,FE,2);
  integral4cells = zeros(size(grid.nodes4cells,1),1);
  integrate!(integral4cells, eval_interpolation_error!(volume_data1D!, val4dofs, FE), grid, 1);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));
  return abs(integral) < eps(10.0)
end

function TestL2BestApproximation1DBoundaryGrid()
  grid = get_boundary_grid(load_test_grid(2););
  println("Testing L2-Bestapproximation on boundary grid of 2D triangulation...");
  FE = FiniteElements.get_P1FiniteElement(grid,true);
  val4dofs = zeros(FE.ndofs);
  computeBestApproximation!(val4dofs,"L2",volume_data_P1!,Nothing,grid,FE,2);
  integral4cells = zeros(size(grid.nodes4cells,1),1);
  integrate!(integral4cells, eval_interpolation_error!(volume_data_P1!, val4dofs, FE), grid, 1);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));
  return abs(integral) < eps(10.0)
end



function TestH1BestApproximation1D()
  grid = load_test_grid1D();
  println("Testing H1-Bestapproximation in 1D...");
  FE = FiniteElements.get_P1FiniteElement(grid);
  val4dofs = zeros(FE.ndofs);
  computeBestApproximation!(val4dofs,"H1",volume_data_gradient!,volume_data1D!,grid,FE,2);
  integral4cells = zeros(size(grid.nodes4cells,1),1);
  integrate!(integral4cells, eval_interpolation_error!(volume_data1D!, val4dofs, FE), grid, 1);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));
  return abs(integral) < eps(10.0)
end

function TestInterpolation2D()
  grid = load_test_grid(0);
  Grid.ensure_volume4cells!(grid);  
  println("Testing P1 Interpolation in 2D...");
  FE = FiniteElements.get_P1FiniteElement(grid);
  val4dofs = zeros(FE.ndofs);
  computeFEInterpolation!(val4dofs, volume_data_P1!, grid, FE);
  integral4cells = zeros(size(grid.nodes4cells, 1), 1);
  integrate!(integral4cells, eval_interpolation_error!(volume_data_P1!, val4dofs, FE), grid, 1);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));
  return abs(integral) < eps(10.0)
end

function TestL2BestApproximation2DP1()
  grid = load_test_grid();
  println("Testing L2-Bestapproximation in 2D for P1-FEM...");
  FE = FiniteElements.get_P1FiniteElement(grid);
  val4dofs = zeros(FE.ndofs);
  computeBestApproximation!(val4dofs,"L2",volume_data_P1!,volume_data_P1!,grid,FE,2);
  val4dofs2 = zeros(FE.ndofs);
  computeFEInterpolation!(val4dofs2, volume_data_P1!, grid, FE);
  integral4cells = zeros(size(grid.nodes4cells,1),1);
  integrate!(integral4cells, eval_interpolation_error!(volume_data_P1!, val4dofs, FE), grid, 1);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));
  return abs(integral) < eps(10.0)
end

function TestL2BestApproximation2DP2()
  grid = load_test_grid();
  println("Testing L2-Bestapproximation in 2D for P2-FEM...");
  FE = FiniteElements.get_P2FiniteElement(grid);
  val4dofs = zeros(FE.ndofs);
  computeBestApproximation!(val4dofs,"L2",volume_data_P2!,volume_data_P2!,grid,FE,4);
  integral4cells = zeros(size(grid.nodes4cells,1),1);
  integrate!(integral4cells, eval_interpolation_error!(volume_data_P2!, val4dofs, FE), grid, 4);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));
  return abs(integral) < eps(10.0)
end


function TestL2BestApproximation2DCR()
  grid = load_test_grid();
  println("Testing L2-Bestapproximation in 2D for CR-FEM...");
  FE = FiniteElements.get_CRFiniteElement(grid);
  val4dofs = zeros(FE.ndofs);
  computeBestApproximation!(val4dofs,"L2",volume_data_P1!,volume_data_P1!,grid,FE,2);
  integral4cells = zeros(size(grid.nodes4cells,1),1);
  integrate!(integral4cells, eval_interpolation_error!(volume_data_P1!, val4dofs, FE), grid, 1);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));
  return abs(integral) < eps(10.0)
end


function TestH1BestApproximation2D()
  grid = load_test_grid();
  println("Testing H1-Bestapproximation in 2D...");
  FE = FiniteElements.get_P1FiniteElement(grid,true);
  val4dofs = zeros(FE.ndofs);
  computeBestApproximation!(val4dofs,"H1",volume_data_gradient!,volume_data_P1!,grid,FE,2);
  integral4cells = zeros(size(grid.nodes4cells,1),1);
  integrate!(integral4cells, eval_interpolation_error!(volume_data_P1!, val4dofs, FE), grid, 1);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));
  return abs(integral) < eps(10.0)
end


function TimeStiffnessMatrixP1()
  grid = load_test_grid(7);
  ncells::Int = size(grid.nodes4cells,1);
  println("ncells=",ncells);
  Grid.ensure_volume4cells!(grid);
  dim=2
  
  aa = Vector{typeof(grid.coords4nodes[1])}(undef, (dim+1)^2*ncells);
  ii = Vector{Int64}(undef, (dim+1)^2*ncells);
  jj = Vector{Int64}(undef, (dim+1)^2*ncells);
  
  println("\n Stiffness-Matrix with exact gradients (fast version)");
  @time FESolveCommon.global_stiffness_matrix!(aa,ii,jj,grid);
  M1 = sparse(ii,jj,aa);
  show(size(M1))
  println("\n Stiffness-Matrix with exact gradients");
  FE = FiniteElements.get_P1FiniteElement(grid,false);
  @time FESolveCommon.global_stiffness_matrix4FE!(aa,ii,jj,grid,FE);
  M2 = sparse(ii,jj,aa);
  show(norm(M1-M2))
  println("\n Stiffness-Matrix with ForwardDiff gradients");
  FE = FiniteElements.get_P1FiniteElement(grid,true);
  @time FESolveCommon.global_stiffness_matrix4FE!(aa,ii,jj,grid,FE);
  M3 = sparse(ii,jj,aa);
  show(norm(M1-M3))
end


function TimeStiffnessMatrixP2()
  grid = load_test_grid(7);
  ncells::Int = size(grid.nodes4cells,1);
  println("ncells=",ncells);
  Grid.ensure_volume4cells!(grid);
  dim=2
  
  aa = Vector{typeof(grid.coords4nodes[1])}(undef, (2*(dim+1))^2*ncells);
  ii = Vector{Int64}(undef, (2*(dim+1))^2*ncells);
  jj = Vector{Int64}(undef, (2*(dim+1))^2*ncells);
  
  println("\n Stiffness-Matrix with exact gradients");
  FE = FiniteElements.get_P2FiniteElement(grid, false);
  @time FESolveCommon.global_stiffness_matrix4FE!(aa,ii,jj,grid,FE);
  M1 = sparse(ii,jj,aa);
  
  println("\n Stiffness-Matrix with ForwardDiff gradients");
  FE = FiniteElements.get_P2FiniteElement(grid, true);
  @time FESolveCommon.global_stiffness_matrix4FE!(aa,ii,jj,grid,FE);
  M2 = sparse(ii,jj,aa);
  show(norm(M1-M2))
end


function TimeStiffnessMatrixCR()
  grid = load_test_grid(6);
  ncells::Int = size(grid.nodes4cells,1);
  println("ncells=",ncells);
  Grid.ensure_volume4cells!(grid);
  dim=2
  
  aa = Vector{typeof(grid.coords4nodes[1])}(undef, 9*ncells);
  ii = Vector{Int64}(undef, 9*ncells);
  jj = Vector{Int64}(undef, 9*ncells);
  
  println("\n Stiffness-Matrix with exact gradients");
  FE = FiniteElements.get_CRFiniteElement(grid, false);
  @time FESolveCommon.global_stiffness_matrix4FE!(aa,ii,jj,grid,FE);
  M1 = sparse(ii,jj,aa);
  
  println("\n Stiffness-Matrix with ForwardDiff gradients");
  FE = FiniteElements.get_CRFiniteElement(grid, true);
  @time FESolveCommon.global_stiffness_matrix4FE!(aa,ii,jj,grid,FE);
  M2 = sparse(ii,jj,aa);
  show(norm(M1-M2))
end

function TimeMassMatrix()
  grid = load_test_grid(5);
  ncells::Int = size(grid.nodes4cells,1);
  println("ncells=",ncells);
  Grid.ensure_volume4cells!(grid);
  dim=2
  
  aa = Vector{typeof(grid.coords4nodes[1])}(undef, (dim+1)^2*ncells);
  ii = Vector{Int64}(undef, (dim+1)^2*ncells);
  jj = Vector{Int64}(undef, (dim+1)^2*ncells);
  gradients4cells = zeros(typeof(grid.coords4nodes[1]),dim+1,dim,ncells);
  
  # old mass matrix
  println("\nold mass matrix routine...")
  @time FESolveCommon.global_mass_matrix_old!(aa,ii,jj,grid);
  M = sparse(ii,jj,aa);
  
  # new mass matrix
  println("\nnew mass matrix routine...")
  @time FESolveCommon.global_mass_matrix!(aa,ii,jj,grid);
  M2 = sparse(ii,jj,aa);
  
  show(norm(M-M2))
  
  # new mass matrix
  println("\nnew mass matrix routine with FE...")
  FE = FiniteElements.get_P1FiniteElement(grid);
  @time FESolveCommon.global_mass_matrix4FE!(aa,ii,jj,grid,FE);
  M2 = sparse(ii,jj,aa);
  show(norm(M-M2))
end

end
