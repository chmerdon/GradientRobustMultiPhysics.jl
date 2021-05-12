#= 

# 203 : Convection-Diffusion-Problem 2D
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

module Example203_ConvectionDiffusion2D

using GradientRobustMultiPhysics
using ExtendableGrids

## problem data and expected exact solution
function exact_solution!(result,x::Array{<:Real,1})
    result[1] = x[1]*x[2]*(x[1]-1)*(x[2]-1) + x[1]
end    
function exact_solution_gradient!(result,x::Array{<:Real,1})
    result[1] = x[2]*(2*x[1]-1)*(x[2]-1) + 1
    result[2] = x[1]*(2*x[2]-1)*(x[1]-1)
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
function main(; verbosity = 0, Plotter = nothing, diffusion = 1e-5, stabilisation = 2e-2, nlevels = 5)

    ## set log level
    set_verbosity(verbosity)
    
    ## load a mesh of the unit square (this one has triangles and quads in it)
    ## it also has four boundary regions (1 = bottom, 2 = right, 3 = top, 4 = left)
    xgrid = grid_unitsquare_mixedgeometries(); # initial grid

    ## negotiate data functions to the package
    user_function = DataFunction(exact_solution!, [1,2]; name = "u", dependencies = "X", quadorder = 4)
    user_function_gradient = DataFunction(exact_solution_gradient!, [2,2]; name = "∇(u)", dependencies = "X", quadorder = 3)
    user_function_rhs = DataFunction(exact_solution_rhs!(diffusion), [1,2]; name = "f", dependencies = "X", quadorder = 3)
    user_function_convection = DataFunction([1,0]; name = "β")

    ## choose a finite element type, here we choose a second order H1-conforming one
    FEType = H1P2{1,2}

    ## create PDE description
    Problem = PDEDescription("convection-diffusion problem")
    add_unknown!(Problem; unknown_name = "u", equation_name = "convection-diffusion equation")
    add_operator!(Problem, [1,1], LaplaceOperator(diffusion))
    add_operator!(Problem, [1,1], ConvectionOperator(user_function_convection,1))

    ## add right-hand side data to equation 1 (there is only one in this example)
    add_rhsdata!(Problem, 1, RhsOperator(Identity, [0], user_function_rhs))

    ## add boundary data to unknown 1 (there is only one in this example)
    ## on boundary regions where the solution is linear the data only needs to be interpolated
    ## on boundary regions where the solution is zero homoegeneous boundary conditions can be used
    add_boundarydata!(Problem, 1, [1,3], BestapproxDirichletBoundary; data = user_function)
    add_boundarydata!(Problem, 1, [2], InterpolateDirichletBoundary; data = user_function)
    add_boundarydata!(Problem, 1, [4], HomogeneousDirichletBoundary)

    ## add a gradient jump (interior penalty) stabilisation for dominant convection
    if stabilisation > 0
        ## first we define an item-dependent action kernel...
        xFaceVolumes::Array{Float64,1} = xgrid[FaceVolumes]
        function stabilisation_kernel(result, input, item)
            for j = 1 : length(input)
                result[j] = input[j] * stabilisation * xFaceVolumes[item]^2
            end
            return nothing
        end
        ## ... which generates an action
        stab_action = Action(Float64,stabilisation_kernel, [2,2]; name = "stabilisation action", dependencies = "I", quadorder = 0 )
        ## ... which is given to a bilinear form constructor
        JumpStabilisation = AbstractBilinearForm([Jump(Gradient), Jump(Gradient)], stab_action; AT = ON_IFACES, name = "s |F|^2 [∇(u)]⋅[∇(v)]")
        add_operator!(Problem, [1,1], JumpStabilisation)
    end

    ## finally we have a look at the defined problem
    @show Problem

    ## define ItemIntegrators for L2/H1 error computation and some arrays to store the errors
    L2ErrorEvaluator = L2ErrorIntegrator(Float64, user_function, Identity)
    H1ErrorEvaluator = L2ErrorIntegrator(Float64, user_function_gradient, Gradient)
    Results = zeros(Float64,nlevels,4); NDofs = zeros(Int,nlevels)

    ## refinement loop over levels
    for level = 1 : nlevels

        ## uniform mesh refinement
        xgrid = uniform_refine(xgrid)
        xFaceVolumes = xgrid[FaceVolumes] # update xFaceVolumes used in stabilisation definition

        ## generate FESpace and solution vector
        FES = FESpace{FEType}(xgrid)
        Solution = FEVector{Float64}("u_h",FES)

        ## solve PDE
        solve!(Solution, Problem)

        ## interpolate (just for comparison)
        Interpolation = FEVector{Float64}("I(u)",FES)
        interpolate!(Interpolation[1], user_function)

        ## compute L2 and H1 errors and save data
        NDofs[level] = length(Solution.entries)
        Results[level,1] = sqrt(evaluate(L2ErrorEvaluator,Solution[1]))
        Results[level,2] = sqrt(evaluate(L2ErrorEvaluator,Interpolation[1]))
        Results[level,3] = sqrt(evaluate(H1ErrorEvaluator,Solution[1]))
        Results[level,4] = sqrt(evaluate(H1ErrorEvaluator,Interpolation[1]))
        
        ## plot
        GradientRobustMultiPhysics.plot(xgrid, [Solution[1], Solution[1]], [Identity, Gradient]; Plotter = Plotter)
    end    

    ## print/plot convergence history
    print_convergencehistory(NDofs, Results; X_to_h = X -> X.^(-1/2), ylabels = ["|| u - u_h ||", "|| u - Iu ||", "|| ∇(u - u_h) ||", "|| ∇(u - Iu) ||"])
    plot_convergencehistory(NDofs, Results; add_h_powers = [2,3], X_to_h = X -> X.^(-1/2), Plotter = Plotter, ylabels = ["|| u - u_h ||", "|| u - Iu ||", "|| ∇(u - u_h) ||", "|| ∇(u - Iu) ||"])
end

end