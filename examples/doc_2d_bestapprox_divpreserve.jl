#= 

# 2D Divergence-Preserving L2-Bestapproximation
([source code](SOURCE_URL))

This example computes the L2-bestapproximation of some given vector-valued function into the lowest-order Raviart-Thomas space. It also
preserves the divergence of the functions in the sense that the divergence of the approximation equals the piecewise integral mean of the exact divergence.
Afterwards the L2 error (also of the divergence error) is computed and the solution is plotted (using PyPlot)

=#

push!(LOAD_PATH, "../src")
using ExtendableGrids
ENV["MPLBACKEND"]="qt5agg"
using PyPlot
using GradientRobustMultiPhysics


## define some vector field that should be approximated
function exact_function!(result,x)
    result[1] = x[1]^3+x[2]^2
    result[2] = -x[1]^2 + x[2] + 1
end
## define its divergence
function exact_divergence!(result,x)
    result[1] = 3*x[1]*x[1] + 1
end

## everything is wrapped in a main function
function main()

    ## generate a unit square mesh and refine
    xgrid = simplexgrid([0.0,1.0],[0.0,1.0])
    xgrid = uniform_refine(xgrid,4)
    
    ## setup a bestapproximation problem via a predefined prototype
    Problem = L2BestapproximationProblem(exact_function!, 2, 2; bestapprox_boundary_regions = [1,2,3,4], bonus_quadorder = 3)

    ## add a new unknown (Lagrange multiplier that handles the divergence constraint)
    ## here 1 is the number of components (it is scalar-valued)
    ## and 2 is the dimension (it lives in 2D)
    add_unknown!(Problem,1,2; unknown_name = "Lagrange multiplier for divergence", equation_name = "divergence constraint")
    add_operator!(Problem, [1,2], LagrangeMultiplier(Divergence))

    ## add the right-hand side data for the constraint and inspect the defined problem
    add_rhsdata!(Problem, 2, RhsOperator(Identity, [0], exact_divergence!, 2, 1; bonus_quadorder = 2))
    Base.show(Problem)

    ## choose some (inf-sup stable) finite element types
    FEType = [HDIVBDM1{2}, L2P0{1}]
    FES = [FESpace{FEType[1]}(xgrid),FESpace{FEType[2]}(xgrid)]

    ## create a solution vector and solve the problem
    Solution = FEVector{Float64}("L2-Bestapproximation",FES)
    solve!(Solution, Problem; verbosity = 1)

    ## calculate L2 error and L2 divergence error
    L2ErrorEvaluator = L2ErrorIntegrator(exact_function!, Identity, 2, 2; bonus_quadorder = 3)
    L2DivergenceErrorEvaluator = L2ErrorIntegrator(exact_divergence!, Divergence, 2, 1; bonus_quadorder = 2)
    println("\nL2error(Id) = $(sqrt(evaluate(L2ErrorEvaluator,Solution[1])))")
    println("L2error(div) = $(sqrt(evaluate(L2DivergenceErrorEvaluator,Solution[1])))")
        
    ## plot the vector field
    PyPlot.figure("|u|")
    nodevals = zeros(Float64,2,size(xgrid[Coordinates],2))
    nodevalues!(nodevals,Solution[1],FES[1])
    ExtendableGrids.plot(xgrid, sqrt.(nodevals[1,:].^2 + nodevals[2,:].^2); Plotter = PyPlot, isolines = 5)
    quiver(xgrid[Coordinates][1,:],xgrid[Coordinates][2,:],nodevals[1,:],nodevals[2,:])

    ## plot the divergence
    PyPlot.figure("divergence(u)")
    nodevals = zeros(Float64,4,size(xgrid[Coordinates],2))
    nodevalues!(nodevals,Solution[1],FES[1],Divergence)
    ExtendableGrids.plot(xgrid, nodevals[1,:]; Plotter = PyPlot)
end

main()