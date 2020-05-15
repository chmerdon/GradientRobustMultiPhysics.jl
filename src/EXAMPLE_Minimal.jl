using FEXGrid
using ExtendableGrids
using FiniteElements
using FEAssembly
ENV["MPLBACKEND"]="qt5agg"
using PyPlot

function main()

    # load mesh and refine
    xgrid = simplexgrid([0.0,1.0],[0.0,1.0])
    for j=1:5
        xgrid = uniform_refine(xgrid)
    end    

    # solve something
    function exact_function!(result,x)
        result[1] = x[1]^2+x[2]^2
    end
    FE = FiniteElements.getH1P2FiniteElement(xgrid,1)
    Solution = FEVector{Float64}("L2-Bestapproximation",FE)
    L2bestapproximate!(Solution[1], exact_function!; bonus_quadorder = 4, boundary_regions = [], verbosity = 3)
        
    # plot
    PyPlot.figure(1)
    xgrid = split_grid_into(xgrid,Triangle2D) #ensures that CellNodes is Array and not Variable
    ExtendableGrids.plot(xgrid, Solution[1][1:size(xgrid[Coordinates],2)]; Plotter = PyPlot)
end

main()