using FEXGrid
using ExtendableGrids
using FiniteElements
using FEAssembly
using PDETools
ENV["MPLBACKEND"]="qt5agg"
using PyPlot

function main()

    # load mesh and refine
    xgrid = simplexgrid([0.0,1.0],[0.0,1.0])
    for j=1:6
        xgrid = uniform_refine(xgrid)
    end
    
    # create Finite Element and FEVectors
    FE = FiniteElements.getHdivRT0FiniteElement(xgrid) # lowest order Raviart-Thomas
    #FE = FiniteElements.getH1P1FiniteElement(xgrid,2) # two-dimensional P1-Courant
    Solution1 = FEVector{Float64}("L2-Bestapproximation direct",FE)
    Solution2 = FEVector{Float64}("L2-Bestapproximation via PDEDescription",FE)

    # define function to be L2-bestapproximated
    function exact_function!(result,x)
        result[1] = x[1]^2+x[2]^2
        result[2] = -x[1]^2 + 1.0
    end

    # FIRST WAY: directly via a shortcut
    L2bestapproximate!(Solution1[1], exact_function!; bonus_quadorder = 2, boundary_regions = [], verbosity = 3)
        
    # SECOND WAY: via PDE description mechanic
    MyLHS = Array{Array{AbstractPDEOperator,1},2}(undef,1,1)
    MyLHS[1,1] = [ReactionOperator(DoNotChangeAction(2))]
    MyRHS = Array{Array{AbstractPDEOperator,1},1}(undef,1)
    MyRHS[1] = [RhsOperator(Identity, [exact_function!], 2, 2; bonus_quadorder = 2)]
    MyBoundary = BoundaryOperator(2,1) # empty, no boundary conditions
    L2BAP = PDEDescription("L2-Bestapproximation problem",MyLHS,MyRHS,[MyBoundary])
    solve!(Solution2,L2BAP; verbosity = 2)

    # uncomment next line to check that both are identical
    # println("difference = $(sum((Solution1.entries - Solution2.entries).^2))");

    # plot
    PyPlot.figure(1)
    nnodes = size(xgrid[Coordinates],2)
    nodevals = zeros(Float64,2,nnodes)
    nodevalues!(nodevals,Solution2[1],FE)
    xgrid = split_grid_into(xgrid,Triangle2D) #ensures that CellNodes is Array and not Variable
    ExtendableGrids.plot(xgrid, nodevals[1,:]; Plotter = PyPlot)
end

main()