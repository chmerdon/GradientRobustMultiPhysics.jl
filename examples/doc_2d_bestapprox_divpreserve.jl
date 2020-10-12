#= 

# 2D Divergence-Preserving L2-Bestapproximation
([source code](SOURCE_URL))

This example computes the L2-bestapproximation of some given vector-valued function into the lowest-order Raviart-Thomas space. It also
preserves the divergence of the function in the sense that the divergence of the approximation equals the piecewise integral mean of the exact divergence.
Afterwards the L2 error (also of the divergence) is computed and the solution is plotted.

=#

module Example_2DBestapproxDivpreserve

using GradientRobustMultiPhysics
using ExtendableGrids

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
function main(; verbosity = 1, Plotter = nothing)

    ## generate a unit square mesh and refine
    xgrid = simplexgrid([0.0,1.0],[0.0,1.0])
    xgrid = uniform_refine(xgrid,4)
    
    ## setup a bestapproximation problem via a predefined prototype
    Problem = L2BestapproximationProblem(exact_function!, 2, 2; bestapprox_boundary_regions = [1,2,3,4], bonus_quadorder = 3)

    ## add a new unknown (Lagrange multiplier that handles the divergence constraint)
    ## here 1 is the number of components (it is scalar-valued) and 2 is the space dimension
    add_unknown!(Problem,1,2; unknown_name = "Lagrange multiplier for divergence", equation_name = "divergence constraint")
    add_operator!(Problem, [1,2], LagrangeMultiplier(Divergence))

    ## add the right-hand side data for the constraint and inspect the defined problem
    add_rhsdata!(Problem, 2, RhsOperator(Identity, [0], exact_divergence!, 2, 1; bonus_quadorder = 2))
    Base.show(Problem)

    ## choose some (inf-sup stable) finite element types
    #FEType = [HDIVBDM1{2}, L2P0{1}]
    FEType = [HDIVRT1{2}, L2P1{1}]
    FES = [FESpace{FEType[1]}(xgrid),FESpace{FEType[2]}(xgrid)]

    ## create a solution vector and solve the problem
    Solution = FEVector{Float64}("L2-Bestapproximation",FES)
    solve!(Solution, Problem; verbosity = verbosity)

    ## calculate L2 error and L2 divergence error
    L2ErrorEvaluator = L2ErrorIntegrator(exact_function!, Identity, 2, 2; bonus_quadorder = 3)
    L2DivergenceErrorEvaluator = L2ErrorIntegrator(exact_divergence!, Divergence, 2, 1; bonus_quadorder = 2)
    println("\nL2error(Id) = $(sqrt(evaluate(L2ErrorEvaluator,Solution[1])))")
    println("L2error(div) = $(sqrt(evaluate(L2DivergenceErrorEvaluator,Solution[1])))")
       
    ## plot
    GradientRobustMultiPhysics.plot(Solution, [1,1], [Identity, Divergence]; Plotter = Plotter, verbosity = verbosity, use_subplots = true)
end

end