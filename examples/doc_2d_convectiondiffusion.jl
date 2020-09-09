
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

=#

module Example_2DConvectionDiffusion

using GradientRobustMultiPhysics
using ExtendableGrids
using Printf


## problem data and expected exact solution
function exact_solution!(result,x)
    result[1] = x[1]*x[2]*(x[1]-1)*(x[2]-1) + x[1]
end    
function exact_solution_gradient!(result,x)
    result[1] = x[2]*(2*x[1]-1)*(x[2]-1) + 1.0
    result[2] = x[1]*(2*x[2]-1)*(x[1]-1)
end    
function beta!(result,x)
    result[1] = 1.0
    result[2] = 0.0
end
function exact_solution_rhs!(diffusion)
    function closure(result,x)
        ## diffusion part
        result[1] = -diffusion*(2*x[2]*(x[2]-1) + 2*x[1]*(x[1]-1))
        ## convection part (beta * grad(u))
        result[1] += x[2]*(2*x[1]-1)*(x[2]-1) + 1.0
    end
end    

## everything is wrapped in a main function
function main(; verbosity = 1, Plotter = nothing)

    #####################################################################################
    #####################################################################################
    
    ## load a mesh of the unit square (this one has triangles and quads in it)
    ## it also has four boundary regions (1 = bottom, 2 = right, 3 = top, 4 = left)
    ## used below to prescribe the boundary data
    xgrid = grid_unitsquare_mixedgeometries(); # initial grid
    
    ## set problem parameters
    diffusion = 1
    nlevels = 7 # number of refinement levels

    ## choose a finite element type
    #FEType = H1P1{1} # P1-Courant
    FEType = H1P2{1,2} # P2
    #FEType = H1CR{1} # Crouzeix-Raviart

    #####################################################################################    
    #####################################################################################

    ## create PDE description = start with Poisson problem and add convection operator to block [1,1] and change equation name
    ConvectionDiffusionProblem = PoissonProblem(2; diffusion = diffusion)
    add_operator!(ConvectionDiffusionProblem, [1,1], ConvectionOperator(beta!,2,1); equation_name = "convection diffusion equation")

    ## add right-hand side data to equation 1 (there is only one in this example)
    add_rhsdata!(ConvectionDiffusionProblem, 1, RhsOperator(Identity, [0], exact_solution_rhs!(diffusion), 2, 1; bonus_quadorder = 3))

    ## add boundary data to unknown 1 (there is only one in this example)
    ## on boundary regions where the solution is linear only need to be interpolated
    ## on boundary regions where the solution is zero only homoegeneous boundary conditions are needed
    add_boundarydata!(ConvectionDiffusionProblem, 1, [1,3], BestapproxDirichletBoundary; data = exact_solution!, bonus_quadorder = 2)
    add_boundarydata!(ConvectionDiffusionProblem, 1, [2], InterpolateDirichletBoundary; data = exact_solution!)
    add_boundarydata!(ConvectionDiffusionProblem, 1, [4], HomogeneousDirichletBoundary)

    ## finally we have a look at the problem
    show(ConvectionDiffusionProblem)

    ## define ItemIntegrators for L2/H1 error computation and some arrays to store the errors
    L2ErrorEvaluator = L2ErrorIntegrator(exact_solution!, Identity, 2, 1; bonus_quadorder = 4)
    H1ErrorEvaluator = L2ErrorIntegrator(exact_solution_gradient!, Gradient, 2, 2; bonus_quadorder = 3)
    L2error = []; H1error = []; L2errorInterpolation = []; H1errorInterpolation = []; NDofs = []

    ## refinement loop over levels
    for level = 1 : nlevels

        ## uniform mesh refinement
        if (level > 1) 
            xgrid = uniform_refine(xgrid)
        end

        ## generate FESpace
        FES = FESpace{FEType}(xgrid)

        ## solve PDE
        Solution = FEVector{Float64}("Problem solution",FES)
        solve!(Solution, ConvectionDiffusionProblem; verbosity = verbosity)
        push!(NDofs,length(Solution.entries))

        ## interpolate
        Interpolation = FEVector{Float64}("Interpolation",FES)
        interpolate!(Interpolation[1], exact_solution!; bonus_quadorder = 4, verbosity = verbosity)

        ## compute L2 and H1 error
        append!(L2error,sqrt(evaluate(L2ErrorEvaluator,Solution[1])))
        append!(L2errorInterpolation,sqrt(evaluate(L2ErrorEvaluator,Interpolation[1])))
        append!(H1error,sqrt(evaluate(H1ErrorEvaluator,Solution[1])))
        append!(H1errorInterpolation,sqrt(evaluate(H1ErrorEvaluator,Interpolation[1])))
        
        ## plot final solution and print error history
        if (level == nlevels)
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

            ## split grid into triangles for the plotter (that only can handle triangles atm)
            if Plotter != nothing
                xgrid = split_grid_into(xgrid,Triangle2D)
                nnodes = size(xgrid[Coordinates],2)
                nodevals = zeros(Float64,2,nnodes)

                nodevalues!(nodevals,Solution[1],FES)
                ExtendableGrids.plot(xgrid, view(nodevals,1,:); Plotter = Plotter, label = "u")

                nodevalues!(nodevals,Solution[1],FES, Gradient)
                ExtendableGrids.plot(xgrid, view(sqrt.(sum(nodevals.^2, dims = 1)),:); Plotter = Plotter, label = "|grad(u)|")
            end
        end    
    end    
end

end