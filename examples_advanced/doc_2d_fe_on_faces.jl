#= 

# 2D FaceGrid
([source code](SOURCE_URL))

This code demonstrates the novel feature of finite element spaces on faces by providing AT = ON_FACES in the finite element space constructor. It is used here to solve a bestapproximation into an Hdiv-conforming space
by using a broken Hdiv space and setting the normal jumps on interior faces to zero by using a Lagrange multiplier on the faces of the grid (a broken H1-conforming space).

=#

module Example_2DFaceElements

using GradientRobustMultiPhysics

## problem data
function exact_function!(result,x::Array{<:Real,1})
    result[1] = x[1]
    result[2] = x[2]
    return nothing
end

## everything is wrapped in a main function
## default argument trigger P1-FEM calculation, you might also want to try H1P2{1,2}
function main(; Plotter = nothing, verbosity = 1, nlevels = 6, FEType = H1P1{1}, testmode = false)

    ## choose initial mesh and get face grid
    xgrid = uniform_refine(grid_unitsquare(Triangle2D),3)

    ## define bestapproximation on face grid
    ## be careful with boundary conditions (not reasonable on face grid)
    user_function = DataFunction(exact_function!, [2,2]; name = "u_exact", dependencies = "X", quadorder = 3)
    Problem = L2BestapproximationProblem(user_function; bestapprox_boundary_regions = [])

    ## we want to use a broken space and give the constraint of no normal jumps
    ## in form of a Lagrange multiplier, since there is no NormalFluxDisc{Jump} operator yet,
    ## we have to use the full identity and multiply the normal vector in an action
    xFaceNormals = xgrid[FaceNormals]
    function LMaction_kernel(result, input, item)
        ## compute normal flux
        result[1] = input[1] * xFaceNormals[1,item] + input[2] * xFaceNormals[2,item]
    end
    LMaction = Action(Float64, ActionKernel(LMaction_kernel, [1,2]; dependencies = "I", quadorder = 0))
    add_unknown!(Problem; unknown_name = "Lagrange multiplier for face jumps", equation_name = "face jump constraint")
    add_operator!(Problem, [1,2], LagrangeMultiplier(IdentityDisc{Jump}; AT = ON_IFACES, action = LMaction))
    add_operator!(Problem, [2,2], DiagonalOperator("Diag(1)", 1.0, true, [1,2,3,4]))

    ## choose some (inf-sup stable) finite element types
    FEType = [HDIVBDM1{2}, H1P1{1}]
    FES = [FESpace{FEType[1]}(xgrid; broken = true),FESpace{FEType[2], ON_FACES}(xgrid; broken = true)]

    ## solve
    Solution = FEVector{Float64}("discrete solution",FES)
    solve!(Solution, Problem; verbosity = verbosity)

    ## calculate L2 error
    L2ErrorEvaluator = L2ErrorIntegrator(Float64, user_function, Identity)
    println("\nL2error(Id) = $(sqrt(evaluate(L2ErrorEvaluator,Solution[1])))")

    ## plot
    GradientRobustMultiPhysics.plot(Solution, [1], [Identity]; Plotter = Plotter, verbosity = verbosity)
end

end

#=
### Output of default main() run
=#
Example_2DFaceElements.main()