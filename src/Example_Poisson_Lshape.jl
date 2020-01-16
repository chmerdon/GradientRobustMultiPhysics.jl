using Grid
using Quadrature
using SparseArrays
using FiniteElements
using FESolveCommon
using FESolvePoisson
using ForwardDiff
using Triangulate
using Printf

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
    println(string(maxarea))
    (triout, vorout)=triangulate("pQa$(@sprintf("%.16f", maxarea))", triin)
    
    return Grid.Mesh{Float64}(Array{Float64,2}(triout.pointlist'),Array{Int64,2}(triout.trianglelist'));
end

function main()

# CHOOSE A FEM
#fem = "CR"
#fem = "P1"
fem = "MINI"
#fem = "P2"

# CHOOSE A PROBLEM
use_problem = "cubic"; f_order = 1; u_order = 3;
#use_problem = "quadratic"; f_order = 0; u_order = 2;
#use_problem = "linear"; f_order = 0; u_order = 1;

# FURTHER PARAMETERS
maxlevel = 5
show_plots = true
show_convergence_history = true


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
Grid.show(grid)

if fem == "P1"
    FE = FiniteElements.getP1FiniteElement(grid,1);
elseif fem == "MINI"
    FE = FiniteElements.getMINIFiniteElement(grid,1);
elseif fem == "CR"
    FE = FiniteElements.getCRFiniteElement(grid,1);
elseif fem == "P2"
    FE = FiniteElements.getP2FiniteElement(grid,1);
end    
FiniteElements.show(FE)
ndofs[level] = FiniteElements.get_ndofs(FE);

println("Solving Poisson problem...");    
val4dofs = FiniteElements.createFEVector(FE);
residual = solvePoissonProblem!(val4dofs,volume_data!(use_problem),wrap_solution,grid,FE,f_order + FiniteElements.get_polynomial_order(FE));
println("  solve residual = " * string(residual))

# compute velocity best approximation
println("Solving L2 best approximation problem...");
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
if (show_plots) && (level == maxlevel) && ndofs[level] < 5000
    nodevals = FESolveCommon.eval_at_nodes(val4dofs,FE);
    pygui(true)
    PyPlot.figure(1)
    PyPlot.plot_trisurf(view(grid.coords4nodes,:,1),view(grid.coords4nodes,:,2),nodevals,cmap=get_cmap("ocean"))
    PyPlot.title("Poisson Problem Solution")
    #show()
end    
end

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
