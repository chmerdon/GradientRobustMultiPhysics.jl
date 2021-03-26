#= 

# 2D Finite Elements on Faces
([source code](SOURCE_URL))

This code demonstrates the novel feature of finite element spaces on faces by providing AT = ON_FACES in the finite element space constructor. It is used here to solve a bestapproximation into an Hdiv-conforming space
by using a broken Hdiv space and setting the normal jumps on interior faces to zero by using a Lagrange multiplier on the faces of the grid (a broken H1-conforming space).
Then the solution is compared to the solution of the same problem using the continuous Hdiv-conforming space.

=#

module Example_2DFaceElements

using GradientRobustMultiPhysics

## problem data
function exact_function!(result,x::Array{<:Real,1})
    result[1] = x[1]^3+x[2]
    result[2] = x[2] + 1
    return nothing
end

## everything is wrapped in a main function
function main(; Plotter = nothing, verbosity = 0)

    ## set log level
    set_verbosity(verbosity)
    
    ## choose initial mesh
    xgrid = uniform_refine(grid_unitsquare(Triangle2D),3)

    ## define bestapproximation problem
    user_function = DataFunction(exact_function!, [2,2]; name = "u_exact", dependencies = "X", quadorder = 3)
    Problem = L2BestapproximationProblem(user_function; name = "constrained L2-bestapproximation problem", bestapprox_boundary_regions = [])

    ## we want to use a broken space and give the constraint of no normal jumps on interior faces
    ## in form of a Lagrange multiplier, since there is no NormalFluxDisc{Jump} operator yet,
    ## we have to use the full identity and multiply the normal vector in an action
    add_unknown!(Problem; unknown_name = "LM face jumps", equation_name = "face jump constraint")
    add_operator!(Problem, [1,2], LagrangeMultiplier(NormalFluxDisc{Jump}; AT = ON_IFACES))
    ## the diagonal operator sets the Lagrange multiplier on all face boundary regions to zero
    add_operator!(Problem, [2,2], DiagonalOperator("Diag(1)", 1.0, true, [1,2,3,4]))

    ## choose some (inf-sup stable) finite element types
    ## first space is the Hdiv element
    ## second will be used for the Lagrange multiplier space on faces
    FEType = [HDIVRT1{2}, H1P1{1}]
    FES = [FESpace{FEType[1]}(xgrid; broken = true),FESpace{FEType[2], ON_FACES}(xgrid; broken = true)]

    ## solve
    Solution = FEVector{Float64}(["u_h (Hdiv-broken)", "LM face jumps"],FES)
    solve!(Solution, Problem)

    ## plot
    GradientRobustMultiPhysics.plot(xgrid, [Solution[1]], [Identity]; Plotter = Plotter)
    
    ## solve again with Hdiv-continuous element
    ## to see that we get the same result
    Problem = L2BestapproximationProblem(user_function; bestapprox_boundary_regions = [])
    FES = FESpace{FEType[1]}(xgrid)

    ## solve
    Solution2 = FEVector{Float64}("u_h (Hdiv-cont.)",FES)
    solve!(Solution2, Problem)

    ## calculate L2 error of both solutions and their difference
    L2ErrorEvaluator = L2ErrorIntegrator(Float64, user_function, Identity)
    L2DiffEvaluator = L2DifferenceIntegrator(Float64, 2, Identity)
    println("\tL2error(Hdiv-broken) = $(sqrt(evaluate(L2ErrorEvaluator,Solution[1])))")
    println("\tL2error(Hdiv-cont.) = $(sqrt(evaluate(L2ErrorEvaluator,Solution2[1])))")
    println("\tL2error(difference) = $(sqrt(evaluate(L2DiffEvaluator,[Solution[1], Solution2[1]])))")
end

end

#=
### Output of default main() run
=#
Example_2DFaceElements.main()