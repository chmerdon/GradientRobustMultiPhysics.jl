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

function triangulate_lshape(maxarea)
    triin=Triangulate.TriangulateIO()
    triin.pointlist=Matrix{Cdouble}([-1 -1;
                     0 -1;
                     0 0;
                     1 0;
                     1 1;
                     -1 1]')
    triin.segmentlist=Matrix{Cint}([1 2 ; 2 3 ; 3 4 ; 4 5 ; 5 6 ; 6 1 ]')
    triin.segmentmarkerlist=Vector{Int32}([1, 2, 3, 4, 5, 6])
    (triout, vorout)=triangulate("pQa" * string(maxarea), triin)
    
    return Grid.Mesh{Float64}(Array{Float64,2}(triout.pointlist'),Array{Int64,2}(triout.trianglelist'));
end


function main()

fem = "CR"
#fem = "P1"
#fem = "P2"
maxlevel = 5
use_FDgradients = false
show_plots = true
show_convergence_history = true
use_problem = "cubic"; f_order = 1; u_order = 3;
#use_problem = "quadratic"; f_order = 0; u_order = 2;
#use_problem = "linear"; f_order = 0; u_order = 1;

# define problem data


function exact_solution(problem)
    function closure(x)
        if problem == "cubic"
            return -x[1]^3 - x[2]^3
        elseif problem == "quadratic"
            return -x[1]^2 - x[2]^2
        elseif problem == "linear"
            return x[1] + x[2]
        end
    end   
end    

function volume_data!(problem)
    hessian = [0.0 0.0;0.0 0.0]
    return function closure(result, x)  
        # compute Laplacian of exact solution
        u(x) = exact_solution(problem)(x)
        hessian = ForwardDiff.hessian(u,x)
        result[1] = -(hessian[1] + hessian[4])
    end    
end

function boundary_data!(problem)
    function closure(result, x)
        result[1] = exact_solution(problem)(x);
    end    
end

function wrap_solution(result,x)
    result[1] = exact_solution(use_problem)(x)
end    

L2error = zeros(Float64,maxlevel)
L2errorBA = zeros(Float64,maxlevel)
ndofs = zeros(Int64,maxlevel)
               
for level = 1 : maxlevel

println("Loading grid by triangle...");
maxarea = 4.0^(-level)
grid = triangulate_lshape(maxarea)

println("maxarea=, maxarea");
println("nnodes=",size(grid.coords4nodes,1));
println("ncells=",size(grid.nodes4cells,1));

println("Solving Poisson problem...");
#ensure_volume4cells!(grid);
#ensure_nodes4faces!(grid);
#show(grid.volume4cells)

if fem == "P1"
    FE = FiniteElements.get_P1FiniteElement(grid,use_FDgradients);
elseif fem == "CR"
    FE = FiniteElements.get_CRFiniteElement(grid,use_FDgradients);
elseif fem == "P2"
    FE = FiniteElements.get_P2FiniteElement(grid,use_FDgradients);
end    
ndofs[level] = FE.ndofs;
val4dofs = zeros(Base.eltype(grid.coords4nodes),FE.ndofs);
    
residual = solvePoissonProblem!(val4dofs,volume_data!(use_problem),boundary_data!(use_problem),grid,FE,f_order + FE.polynomial_order);
println("solve residual = " * string(residual))

# compute velocity best approximation
val4dofsBA = zeros(Base.eltype(grid.coords4nodes),FE.ndofs);
residual = computeBestApproximation!(val4dofsBA,"L2",wrap_solution,wrap_solution,grid,FE,u_order+FE.polynomial_order)
println("residual = " * string(residual));

# compute errors
integral4cells = zeros(size(grid.nodes4cells,1),1);
integrate!(integral4cells,eval_L2_interpolation_error!(wrap_solution, val4dofs, FE), grid, 2*maximum([u_order,FE.polynomial_order]), 1);
L2error[level] = sqrt(abs(sum(integral4cells)));
println("L2_error = " * string(L2error[level]));
integrate!(integral4cells,eval_L2_interpolation_error!(wrap_solution, val4dofsBA, FE), grid, 2*maximum([u_order,FE.polynomial_order]), 1);
L2errorBA[level] = sqrt(abs(sum(integral4cells)));
println("L2_error_BA = " * string(L2errorBA[level]));


# plot
if (show_plots) && (level == maxlevel)
    # evaluate velocity and pressure at grid points
    nodevals = zeros(size(grid.coords4nodes,1))
    ndofs4node = zeros(size(grid.coords4nodes,1))
    temp = 0;
    for cell = 1 : size(grid.nodes4cells,1)
        for node = 1 : size(grid.nodes4cells,2)
            for dof = 1 : size(FE.dofs4cells,2)
                temp = FE.bfun_ref[dof]([0,0], grid, cell);
                temp *= val4dofs[FE.dofs4cells[cell, dof]];
                nodevals[grid.nodes4cells[cell,node]] += temp
            end
            ndofs4node[grid.nodes4cells[cell,node]] +=1
        end    
    end
    
    # average
    nodevals ./= ndofs4node
    
    pygui(true)
    PyPlot.figure(1)
    PyPlot.plot_trisurf(view(grid.coords4nodes,:,1),view(grid.coords4nodes,:,2),nodevals,cmap=get_cmap("ocean"))
    PyPlot.title("Poisson Problem Solution")
    #show()
end    
end


println("\n L2 error");
show(L2error)
println("\n L2 error BA");
show(L2errorBA)

if (show_convergence_history)
    PyPlot.figure(2)
    PyPlot.loglog(ndofs,L2error,"-o")
    PyPlot.loglog(ndofs,L2errorBA,"-o")
    PyPlot.loglog(ndofs,ndofs.^(-1/2),"--",color = "gray")
    PyPlot.loglog(ndofs,ndofs.^(-1),"--",color = "gray")
    PyPlot.loglog(ndofs,ndofs.^(-3/2),"--",color = "gray")
    PyPlot.legend(("L2 error","L2 error BA","O(h)","O(h^2)","O(h^3)"))
    PyPlot.title("Convergence history (fem=" * fem * " problem=" * use_problem * ")")
    ax = PyPlot.gca()
    ax.grid(true)
end 


end


main()
