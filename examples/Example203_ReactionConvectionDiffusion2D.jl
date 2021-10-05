#= 

# 203 : Reaction-Convection-Diffusion-Problem 2D
([source code](SOURCE_URL))

This example computes the solution of some convection-diffusion problem
```math
-\nu \Delta u + \mathbf{\beta} \cdot \nabla u + \alpha u = f \quad \text{in } \Omega
```
with some diffusion coefficient  ``\nu``, some vector-valued function  ``\mathbf{\beta}``, some scalar-valued function ``\alpha`` and inhomogeneous Dirichlet boundary data.

We prescribe an analytic solution with ``\mathbf{\beta} := (1,0)`` and ``\alpha = 0.1`` and check the L2 and H1 error convergence of the method on a series of uniformly refined meshes.
We also compare with the error of a simple nodal interpolation and plot the solution and the norm of its gradient.

For small ``\nu``, the convection term dominates and pollutes the accuracy of the method. For demonstration some
simple gradient jump (interior penalty) stabilisation is added to improve things.

=#

module Example203_ReactionConvectionDiffusion2D

using GradientRobustMultiPhysics
using ExtendableGrids

## coefficient functions
const β = DataFunction([1,0]; name = "β")
const α = DataFunction([0.01]; name = "α")

## problem data and expected exact solution
function exact_solution!(result,x::Array{<:Real,1})
    result[1] = x[1]*x[2]*(x[1]-1)*(x[2]-1) + x[1]
end    
function exact_solution_gradient!(result,x::Array{<:Real,1})
    result[1] = x[2]*(2*x[1]-1)*(x[2]-1) + 1
    result[2] = x[1]*(2*x[2]-1)*(x[1]-1)
end    
function exact_solution_rhs!(ν)
    eval_alpha = zeros(Float64,1)
    eval_beta = zeros(Float64,2)
    function closure(result,x::Array{<:Real,1})
        ## diffusion part
        result[1] = -ν*(2*x[2]*(x[2]-1) + 2*x[1]*(x[1]-1))
        ## convection part (beta * grad(u))
        eval_data!(eval_beta, β, x, 0)
        result[1] += eval_beta[1] * (x[2]*(2*x[1]-1)*(x[2]-1) + 1)
        result[1] += eval_beta[2] * (x[1]*(2*x[2]-1)*(x[1]-1))
        ## reaction part (alpha*u)
        eval_data!(eval_alpha, α, x, 0)
        result[1] += eval_alpha[1] * (x[1]*x[2]*(x[1]-1)*(x[2]-1) + x[1])
        return nothing
    end
end    

## custom bilinearform that can assemble the full PDE operator
function ReactionConvectionDiffusionOperator(α, β, ν)
    eval_alpha = zeros(Float64,1)
    eval_beta = zeros(Float64,2)
    function action_kernel!(result, input,x)
        ## input = [u,∇u] as a vector of length 3
        eval_data!(eval_beta, β, x, 0)
        eval_data!(eval_alpha, α, x, 0)
        result[1] = eval_alpha[1] * input[1] + eval_beta[1] * input[2] + eval_beta[2] * input[3]
        result[2] = ν * input[2]
        result[3] = ν * input[3]
        ## result will be multiplied with [v,∇v]
        return nothing
    end
    action = Action{Float64}( ActionKernel(action_kernel!, [3,3]; dependencies = "X", quadorder = max(α.quadorder,β.quadorder)))
    return AbstractBilinearForm([OperatorPair{Identity,Gradient},OperatorPair{Identity,Gradient}], action; name = "ν(∇u,∇v) + (αu + β⋅∇u, v)", transposed_assembly = true)
end

## everything is wrapped in a main function
function main(; verbosity = 0, Plotter = nothing, ν = 1e-5, τ = 2e-2, nlevels = 5)

    ## set log level
    set_verbosity(verbosity)
    
    ## load a mesh of the unit square (this one has triangles and quads in it)
    ## it also has four boundary regions (1 = bottom, 2 = right, 3 = top, 4 = left)
    xgrid = grid_unitsquare_mixedgeometries(); # initial grid

    ## negotiate data functions to the package
    u = DataFunction(exact_solution!, [1,2]; name = "u", dependencies = "X", quadorder = 4)
    ∇u = DataFunction(exact_solution_gradient!, [2,2]; name = "∇(u)", dependencies = "X", quadorder = 3)
    f = DataFunction(exact_solution_rhs!(ν), [1,2]; name = "f", dependencies = "X", quadorder = 3)

    ## choose a finite element type, here we choose a second order H1-conforming one
    FEType = H1P2{1,2}

    ## create PDE description
    Problem = PDEDescription("reaction-convection-diffusion problem")
    add_unknown!(Problem; unknown_name = "u", equation_name = "reaction-convection-diffusion equation")
    add_operator!(Problem, [1,1], ReactionConvectionDiffusionOperator(α,β,ν))
    add_rhsdata!(Problem, 1, RhsOperator(Identity, [0], f))

    ## add boundary data to unknown 1 (there is only one in this example)
    ## on boundary regions where the solution is linear the data only needs to be interpolated
    ## on boundary regions where the solution is zero homoegeneous boundary conditions can be used
    add_boundarydata!(Problem, 1, [1,3], BestapproxDirichletBoundary; data = u)
    add_boundarydata!(Problem, 1, [2], InterpolateDirichletBoundary; data = u)
    add_boundarydata!(Problem, 1, [4], HomogeneousDirichletBoundary)

    ## add a gradient jump (interior penalty) stabilisation for dominant convection
    if τ > 0
        ## first we define an item-dependent action kernel...
        xFaceVolumes::Array{Float64,1} = xgrid[FaceVolumes]
        stab_action = Action{Float64}((result,input,item) -> (result .= input .* xFaceVolumes[item]^2), [2,2]; name = "stabilisation action", dependencies = "I", quadorder = 0 )
        JumpStabilisation = AbstractBilinearForm([Jump(Gradient), Jump(Gradient)], stab_action; AT = ON_IFACES, factor = τ, name = "τ |F|^2 [∇(u)]⋅[∇(v)]")
        add_operator!(Problem, [1,1], JumpStabilisation)
    end

    ## finally we have a look at the defined problem
    @show Problem

    ## define ItemIntegrators for L2/H1 error computation and some arrays to store the errors
    L2ErrorEvaluator = L2ErrorIntegrator(Float64, u, Identity)
    H1ErrorEvaluator = L2ErrorIntegrator(Float64, ∇u, Gradient)
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
        interpolate!(Interpolation[1], u)

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