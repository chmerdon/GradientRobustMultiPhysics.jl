using Grid
using Quadrature
using SparseArrays
using FiniteElements
using FESolveCommon
using FESolvePoisson
using ForwardDiff
using Triangulate
ENV["MPLBACKEND"]="tkagg"
using PyPlot

function get_line_grid(maxarea)
    return Grid.Mesh{Float64}(Array{Float64,2}(Array{Float64,2}([0,1,2]')'),Array{Int64,2}([1 2;2 3]),ceil(log2(1/maxarea)));
end

function main()
fem = "P1"
#fem = "P2"

maxlevel = 13
use_FDgradients = false
show_plots = true
show_convergence_history = true
use_problem = "quartic"; f_order = 2; u_order = 4;
#use_problem = "cubic"; f_order = 1; u_order = 3;
#use_problem = "quadratic"; f_order = 0; u_order = 2;
#use_problem = "linear"; f_order = 0; u_order = 1;

# define problem data


function exact_solution(problem)
    function closure(x)
        if problem == "quartic"
            return x[1]^4 - x[1]^3 +2*x[1]^2 - 3*x[1]
        elseif problem == "cubic"
            return -x[1]^3
        elseif problem == "quadratic"
            return -x[1]^2
        elseif problem == "linear"
            return x[1]
        end
    end   
end    

function volume_data!(problem)
    hessian = [0.0]
    return function closure(result, x)  
        # compute Laplacian of exact solution
        u(x) = exact_solution(problem)(x)
        hessian = ForwardDiff.hessian(u,x)
        result[1] = -hessian[1]
    end    
end

function wrap_solution(result,x)
    result[1] = exact_solution(use_problem)(x)
end    

L2error = zeros(Float64,maxlevel)
L2errorBA = zeros(Float64,maxlevel)
ndofs = zeros(Int64,maxlevel)
               
for level = 1 : maxlevel


maxarea = 2.0^(-level)
grid = get_line_grid(maxarea)
Grid.show(grid)

if fem == "P1"
    @time FE = FiniteElements.getP1FiniteElement(grid,1);
elseif fem == "P2"
    FE = FiniteElements.getP2FiniteElement(grid,1);
end  
ensure_nodes4faces!(grid);
ensure_volume4cells!(grid);
FiniteElements.show(FE)
ndofs[level] = FiniteElements.get_ndofs(FE);

println("Solving Poisson problem...");    
val4dofs = FiniteElements.createFEVector(FE);
residual = solvePoissonProblem!(val4dofs,volume_data!(use_problem),wrap_solution,grid,FE,f_order + FiniteElements.get_polynomial_order(FE));
println("  solve residual = " * string(residual))

# compute velocity best approximation
println("Solving L2 bestapproximation problem...");    
val4dofsBA = FiniteElements.createFEVector(FE);
residual = computeBestApproximation!(val4dofsBA,"L2",wrap_solution,wrap_solution,grid,FE,u_order + FiniteElements.get_polynomial_order(FE))
println("  solve residual = " * string(residual));

# compute errors
integral4cells = zeros(size(grid.nodes4cells,1),1);
integrate!(integral4cells,eval_L2_interpolation_error!(wrap_solution, val4dofs, FE), grid, 2*maximum([u_order,FiniteElements.get_polynomial_order(FE)]), 1);
L2error[level] = sqrt(abs(sum(integral4cells)));
println("L2_error = " * string(L2error[level]));
integrate!(integral4cells,eval_L2_interpolation_error!(wrap_solution, val4dofsBA, FE), grid, 2*maximum([u_order,FiniteElements.get_polynomial_order(FE)]), 1);
L2errorBA[level] = sqrt(abs(sum(integral4cells)));
println("L2_error_BA = " * string(L2errorBA[level]));


# plot
if (show_plots) && (level == maxlevel)
    nodevals = FESolveCommon.eval_at_nodes(val4dofs,FE);
    pygui(true)
    PyPlot.figure(1)
    I = sortperm(grid.coords4nodes[:])
    PyPlot.plot(grid.coords4nodes[I],nodevals[I])
    PyPlot.title("Poisson Problem Solution")
    #show()
end    
end

if (show_convergence_history)
    PyPlot.figure(2)
    PyPlot.loglog(ndofs,L2error,"-o")
    PyPlot.loglog(ndofs,L2errorBA,"-o")
    PyPlot.loglog(ndofs,ndofs.^(-1),"--",color = "gray")
    PyPlot.loglog(ndofs,ndofs.^(-2),"--",color = "gray")
    PyPlot.loglog(ndofs,ndofs.^(-3),"--",color = "gray")
    PyPlot.legend(("L2 error","L2 error BA","O(h)","O(h^2)","O(h^3)"))
    PyPlot.title("Convergence history (fem=" * fem * " problem=" * use_problem * ")")
    ax = PyPlot.gca()
    ax.grid(true)
end 


end


main()
