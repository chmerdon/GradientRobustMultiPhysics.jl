using FEXGrid
using ExtendableGrids
using FiniteElements
using FEAssembly
using PDETools
ENV["MPLBACKEND"]="qt5agg"
using PyPlot

function main()

    verbosity = 2 # <-- increase/decrease this number to get more/less printouts on what is happening

    # define some (vector-valued) function (to be L2-bestapproximated in this example)
    function exact_function!(result,x)
        result[1] = x[1]^2+x[2]^2
        result[2] = -x[1]^2 + 1.0
    end
    # define its divergence (used in error calculation)
    function exact_divergence!(result,x)
        result[1] = 2*x[1]
    end

    # load mesh and refine
    xgrid = simplexgrid([0.0,1.0],[0.0,1.0])
    for j=1:6
        xgrid = uniform_refine(xgrid)
    end
    
    # create Finite Element Space and FEVectors: typeof(FE) carries information about the FiniteElement, while full FE carries links to dofmaps etc.
    FE = FiniteElements.getHdivRT0FiniteElement(xgrid) # lowest order Raviart-Thomas
    #FE = FiniteElements.getH1P1FiniteElement(xgrid,2) # two-dimensional P1-Courant
    Solution1 = FEVector{Float64}("L2-Bestapproximation direct",FE)
    Solution2 = FEVector{Float64}("L2-Bestapproximation via PDEDescription",FE)


    # FIRST WAY: directly via a shortcut
    L2bestapproximate!(Solution1[1], exact_function!; bonus_quadorder = 2, boundary_regions = [0], verbosity = verbosity)

        
    # SECOND WAY: via PDE description mechanic
    MyLHS = Array{Array{AbstractPDEOperator,1},2}(undef,1,1)
    MyLHS[1,1] = [ReactionOperator(DoNotChangeAction(2))]
    MyRHS = Array{Array{AbstractPDEOperator,1},1}(undef,1)
    MyRHS[1] = [RhsOperator(Identity, [exact_function!], 2, 2; bonus_quadorder = 2)]
    MyBoundary = BoundaryOperator(2,1) # leave it like that = no boundary conditions
    append!(MyBoundary, [1,2,3,4], BestapproxDirichletBoundary; data = exact_function!, bonus_quadorder = 2)
    L2BAP = PDEDescription("L2-Bestapproximation problem",MyLHS,MyRHS,[MyBoundary])
    # solve the described PDE
    solve!(Solution2,L2BAP; verbosity = verbosity)
    

    # check that both are identical
    println("\nDifference = $(sum((Solution1.entries - Solution2.entries).^2))");
    # calculate L2 error and L2 divergence error
    L2ErrorEvaluator = L2ErrorIntegrator(exact_function!, Identity, 2, 2; bonus_quadorder = 2)
    L2DivergenceErrorEvaluator = L2ErrorIntegrator(exact_divergence!, Divergence, 2, 1; bonus_quadorder = 1)
    println("L2error(Id) = $(sqrt(evaluate(L2ErrorEvaluator,Solution1[1])))")
    println("L2error(div) = $(sqrt(evaluate(L2DivergenceErrorEvaluator,Solution1[1])))")
        
    # evaluate/interpolate function at nodes and plot
    PyPlot.figure(1)
    nnodes = size(xgrid[Coordinates],2)
    nodevals = zeros(Float64,2,nnodes)
    nodevalues!(nodevals,Solution2[1],FE)
    ExtendableGrids.plot(xgrid, nodevals[1,:]; Plotter = PyPlot)
end

main()