
#= 

# 2D Convection-Diffusion-Problem
([source code](SOURCE_URL))

This example computes the solution of some convection-diffusion problem
```math
-\nu \Delta u + \mathbf{\beta} \cdot \nabla u = f \quad \text{in } \Omega
```
with some diffusion coefficient  ``\nu``, some vector-valued function  ``\mathbf{\beta}`` and inhomogeneous Dirichlet boundary data.

We prescribe an analytic solution and check the L2 and H1 error convergence of the method on a series of uniformly refined meshes.
We also compare with the error of a simple nodal interpolation and plot the solution and the norm of its gradient.

For small ``\nu``, the convection term dominates and pollutes the accuracy of the method. For demonstration some
simple gradient jump (interior penalty) stabilisation is added to improve things.

=#

module Example_2DConvectionDiffusion

using GradientRobustMultiPhysics
using ExtendableGrids
using Printf


## problem data and expected exact solution
function exact_solution!(result,x::Array{<:Real,1})
    result[1] = x[1]*x[2]*(x[1]-1)*(x[2]-1) + x[1]
end    
function exact_solution_gradient!(result,x::Array{<:Real,1})
    result[1] = x[2]*(2*x[1]-1)*(x[2]-1) + 1
    result[2] = x[1]*(2*x[2]-1)*(x[1]-1)
end    
function beta!(result)
    result[1] = 1
    result[2] = 0
end
function exact_solution_rhs!(diffusion)
    function closure(result,x::Array{<:Real,1})
        ## diffusion part
        result[1] = -diffusion*(2*x[2]*(x[2]-1) + 2*x[1]*(x[1]-1))
        ## convection part (beta * grad(u))
        result[1] += x[2]*(2*x[1]-1)*(x[2]-1) + 1
        return nothing
    end
end    

## everything is wrapped in a main function
function main(; verbosity = 1, Plotter = nothing, diffusion = 1e-5, stabilisation = 2e-2, nlevels = 6)
    
    ## load a mesh of the unit square (this one has triangles and quads in it)
    ## it also has four boundary regions (1 = bottom, 2 = right, 3 = top, 4 = left)
    ## used below to prescribe the boundary data
    xgrid = grid_unitsquare_mixedgeometries(); # initial grid

    ## negotiate data functions to the package
    user_function = DataFunction(exact_solution!, [1,2]; dependencies = "X", quadorder = 4)
    user_function_gradient = DataFunction(exact_solution_gradient!, [2,2]; dependencies = "X", quadorder = 3)
    user_function_rhs = DataFunction(exact_solution_rhs!(diffusion), [1,2]; dependencies = "X", quadorder = 3)
    user_function_convection = DataFunction(beta!, [2,2]; dependencies = "", quadorder = 0)

    ## choose a finite element type, here we choose a second order H1-conforming one
    FEType = H1P2{1,2}

    ## create PDE description = start with Poisson problem and add convection operator to block [1,1] and change equation name
    Problem = PoissonProblem(2; diffusion = diffusion)
    add_operator!(Problem, [1,1], ConvectionOperator(Float64,user_function_convection,1); equation_name = "convection diffusion equation")

    ## add right-hand side data to equation 1 (there is only one in this example)
    add_rhsdata!(Problem, 1, RhsOperator(Identity, [0], user_function_rhs))

    ## add boundary data to unknown 1 (there is only one in this example)
    ## on boundary regions where the solution is linear only need to be interpolated
    ## on boundary regions where the solution is zero only homoegeneous boundary conditions are needed
    add_boundarydata!(Problem, 1, [1,3], BestapproxDirichletBoundary; data = user_function)
    add_boundarydata!(Problem, 1, [2], InterpolateDirichletBoundary; data = user_function)
    add_boundarydata!(Problem, 1, [4], HomogeneousDirichletBoundary)

    # add a gradient jump (interior penalty) stabilisation for dominant convection
    if stabilisation > 0
        ## first we define an item-dependent action kernel...
        xFaceVolumes::Array{Float64,1} = xgrid[FaceVolumes]
        function stabilisation_kernel(result, input, item)
            for j = 1 : length(input)
                result[j] = input[j] * stabilisation * xFaceVolumes[item]^2
            end
            return nothing
        end
        stab_action_kernel = ActionKernel(stabilisation_kernel, [2,2]; name = "stabilisation action kernel", dependencies = "I", quadorder = 0)
        ## ... which generates an action...
        stab_action = Action(Float64,stab_action_kernel)
        ## ... which is given to an bilinear form constructor
        JumpStabilisation = AbstractBilinearForm("[grad(u)] [grad(v)]", GradientDisc{Jump}, GradientDisc{Jump}, stab_action; AT = ON_IFACES)
        add_operator!(Problem, [1,1], JumpStabilisation)
    end

    ## finally we have a look at the problem
    show(Problem)

    ## define ItemIntegrators for L2/H1 error computation and some arrays to store the errors
    L2ErrorEvaluator = L2ErrorIntegrator(Float64, user_function, Identity)
    H1ErrorEvaluator = L2ErrorIntegrator(Float64, user_function_gradient, Gradient)
    L2error = []; H1error = []; L2errorInterpolation = []; H1errorInterpolation = []; NDofs = []

    ## refinement loop over levels
    for level = 1 : nlevels

        ## uniform mesh refinement
        if (level > 1) 
            xgrid = uniform_refine(xgrid)
        end

        ## generate FESpace
        FES = FESpace{FEType}(xgrid)
        xFaceVolumes = xgrid[FaceVolumes] # update xFaceVolumes used in stabilisation definition

        ## solve PDE
        Solution = FEVector{Float64}("Problem solution",FES)
        solve!(Solution, Problem; verbosity = verbosity)
        push!(NDofs,length(Solution.entries))

        ## interpolate
        Interpolation = FEVector{Float64}("Interpolation",FES)
        interpolate!(Interpolation[1], user_function; verbosity = verbosity)

        ## compute L2 and H1 error
        append!(L2error,sqrt(evaluate(L2ErrorEvaluator,Solution[1])))
        append!(L2errorInterpolation,sqrt(evaluate(L2ErrorEvaluator,Interpolation[1])))
        append!(H1error,sqrt(evaluate(H1ErrorEvaluator,Solution[1])))
        append!(H1errorInterpolation,sqrt(evaluate(H1ErrorEvaluator,Interpolation[1])))
        
        if (level == nlevels)
            ## print error history
            println("\n         |   L2ERROR   |   L2ERROR")
            println("   NDOF  |   SOLUTION  |   INTERPOL");
            for j=1:nlevels
                @printf("  %6d |",NDofs[j]);
                @printf(" %.5e |",L2error[j])
                @printf(" %.5e\n",L2errorInterpolation[j])
            end
            println("\n         |   H1ERROR   |   H1ERROR")
            println("   NDOF  |   SOLUTION  |   INTERPOL");
            for j=1:nlevels
                @printf("  %6d |",NDofs[j]);
                @printf(" %.5e |",H1error[j])
                @printf(" %.5e\n",H1errorInterpolation[j])
            end

            ## plot
            GradientRobustMultiPhysics.plot(Solution, [1,1], [Identity, Gradient]; Plotter = Plotter, verbosity = verbosity, use_subplots = true)
        end    
    end    
end

end