module FESolveStokesTests

using SparseArrays
using LinearAlgebra
using FESolveCommon
using FESolveStokes
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

  
function TestStokesTH(show_plot::Bool = false)

    # data for test problem u = (y^2,x^2), p = x
    function volume_data!(result, x) 
        result[1] = -1.0;
        result[2] = -4.0;
    end

    function exact_velocity!(result,x)
        result[1] = x[2]^2;
        result[2] = x[1]^2;
    end
    
    function exact_pressure!(result,x)
        result[1] = x[1] - 2*x[2] + 1 // 2;
    end

    # define grid
    coords4nodes_init = [0 0;
                        1 0;
                        1 1;
                        0 1];
    nodes4cells_init = [1 2 3;
                        1 3 4];
               
    println("Loading grid...");
    @time grid = Grid.Mesh{Float64}(coords4nodes_init,nodes4cells_init,2);
    println("nnodes=",size(grid.coords4nodes,1));
    println("ncells=",size(grid.nodes4cells,1));

    println("Solving Stokes problem with Taylor--Hood element...");
    FE_velocity = FiniteElements.get_P2VectorFiniteElement(grid,false);
    FE_pressure = FiniteElements.get_P1FiniteElement(grid,false);
    ndofs_velocity = FE_velocity.ndofs;
    ndofs_pressure = FE_pressure.ndofs;
    ndofs_total = ndofs_velocity + ndofs_pressure;
    println("ndofs_velocity=",ndofs_velocity);
    println("ndofs_pressure=",ndofs_pressure);
    println("ndofs_total=",ndofs_total);
    val4coords = zeros(Base.eltype(grid.coords4nodes),ndofs_total);
    residual = solveStokesProblem!(val4coords,1,volume_data!,exact_velocity!,grid,FE_velocity,FE_pressure,4);
    println("solver residual = " * string(residual));
    integral4cells = zeros(size(grid.nodes4cells,1),1);
    integrate!(integral4cells,eval_L2_interpolation_error!(exact_pressure!, val4coords[ndofs_velocity+1:end], FE_pressure), grid, 2*FE_pressure.polynomial_order);
    L2error_pressure = sqrt(abs(sum(integral4cells)));
    println("pressure_error = " * string(L2error_pressure));
    integral4cells = zeros(size(grid.nodes4cells,1),2);
    integrate!(integral4cells,eval_L2_interpolation_error!(exact_velocity!, val4coords[1:ndofs_velocity], FE_velocity), grid, 2*FE_velocity.polynomial_order, 2);
    L2error_velocity = sqrt(abs(sum(integral4cells[:])));
    println("velocity_error = " * string(L2error_velocity));
   
    return L2error_velocity + L2error_pressure <= eps(100/residual)
end

  
function TimeStokesOperatorTH()
  grid = load_test_grid(2);
  ncells::Int = size(grid.nodes4cells,1);
  println("ncells=",ncells);
  Grid.ensure_volume4cells!(grid);
  dim=2
  
  
  FE_velocity = FiniteElements.get_P2VectorFiniteElement(grid,false);
  FE_pressure = FiniteElements.get_P1FiniteElement(grid,false);
  ndofs4cell = size(FE_velocity.dofs4cells,2) + size(FE_pressure.dofs4cells,2)
  aa = Vector{typeof(grid.coords4nodes[1])}(undef, ndofs4cell^2*ncells);
  ii = Vector{Int64}(undef, ndofs4cell^2*ncells);
  jj = Vector{Int64}(undef, ndofs4cell^2*ncells);
  
  println("\n Stokes-Matrix with exact gradients");
  @time FESolveStokes.StokesOperator4FE!(aa,ii,jj,grid,FE_velocity,FE_pressure);
  
  M1 = sparse(ii,jj,aa);
  
  println("\n Stokes-Matrix with ForwardDiff gradients");
  FE_velocity = FiniteElements.get_P2VectorFiniteElement(grid,true);
  FE_pressure = FiniteElements.get_P1FiniteElement(grid,true);
  @time FESolveStokes.StokesOperator4FE!(aa,ii,jj,grid,FE_velocity,FE_pressure);
  M2 = sparse(ii,jj,aa);
  show(norm(M1-M2))
end

end
