using Grid
using Quadrature
using SparseArrays
using FiniteElements
using FESolvePoisson
ENV["MPLBACKEND"]="tkagg"
using PyPlot


function main()
# define problem data

function volume_data!(result, x)
    fill!(result, 1.0)
end

function boundary_data!(result,x,xref)
    fill!(result, 0.0);
end

# define grid
coords4nodes_init = [-1 -1;
                     0 -1;
                     0 0;
                     1 0;
                     1 1;
                     0 1;
                     -1 1;
                     -1 0;
                     -1//2 -1//2;
                     -1//2 1//2;
                     1//2 1//2];
nodes4cells_init = [1 2 9;
                    2 3 9;
                    3 8 9;
                    8 1 9;
                    8 3 10;
                    3 6 10;
                    6 7 10;
                    7 8 10;
                    3 4 11;
                    4 5 11;
                    5 6 11;
                    6 3 11];
               
println("Loading grid...");
@time grid = Grid.Mesh{Float64}(coords4nodes_init,nodes4cells_init,3);


println("nnodes=",size(grid.coords4nodes,1));
println("ncells=",size(grid.nodes4cells,1));

println("Solving Poisson problem...");
#ensure_volume4cells!(grid);
#show(grid.volume4cells)


FE = FiniteElements.get_P2FiniteElement(grid,true);
val4coords = zeros(Base.eltype(grid.coords4nodes),size(FE.coords4dofs,1));
    
solvePoissonProblem!(val4coords,volume_data!,boundary_data!,grid,FE,2);
#show(val4coords[1:10]);

# plot
if size(grid.coords4nodes,1) < 5000
    pygui(true)
    PyPlot.figure(1)
    PyPlot.plot_trisurf(view(FE.coords4dofs,:,1),view(FE.coords4dofs,:,2),val4coords,cmap=get_cmap("ocean"))
    PyPlot.title("Poisson Problem Solution")
    #show()
end    
end


main()
