### Divergence-preserving L2-Bestapproximation

This example is an extended version of [Minimal Example: L2-Bestapproximation](@ref) that adds a side constraint that preserves the divergence of the approximated function.


We first define some (vector-valued) function (to be L2-bestapproximated in this example) and its divergence

```@example EXAMPLE_Minimal
function exact_function!(result,x)
    result[1] = x[1]^2+x[2]^2
    result[2] = -x[1]^2 + 1.0
end
function exact_divergence!(result,x)
    result[1] = 2*x[1]
end
```

Then, we generate a mesh and refine in a bit

```@example EXAMPLE_Minimal
xgrid = simplexgrid([0.0,1.0],[0.0,1.0])
for j=1:6
    xgrid = uniform_refine(xgrid)
end
```

Next, we define the L2-Bestapproximation problem via a PDEDEscription prototype and add the divergence constraint by adding another unknown for the Lagrange multiplier

```@example EXAMPLE_Minimal
Problem = L2BestapproximationProblem(exact_function!, 2, 2; bestapprox_boundary_regions = [1,2,3,4], bonus_quadorder = 2)
add_unknown!(Problem,1,2)
add_operator!(Problem, [1,2], LagrangeMultiplier(Divergence))
add_rhsdata!(Problem, 2, RhsOperator(Identity, [exact_divergence!], 2, 1; bonus_quadorder = 1))
show(Problem)
```

To discretise, let us choose some lowest-order Raviart-Thomas finite element space and piecewise constants for the Lagrange multiplier.

```@example EXAMPLE_Minimal
FEType = [HDIVRT0{2}, L2P0{1}]
FES = [FESpace{FEType[1]}(xgrid),FESpace{FEType[2]}(xgrid)]
```

Finally, we create a FEVector for the FESpace and solve the problem into it

```@example EXAMPLE_Minimal
Solution = FEVector{Float64}("L2-Bestapproximation",FES)
solve!(Solution, Problem)
```

The L2 error of our discrete solution and its divergence can be calculated by

```@example EXAMPLE_Minimal
L2ErrorEvaluator = L2ErrorIntegrator(exact_function!, Identity, 2, 2; bonus_quadorder = 2)
L2DivergenceErrorEvaluator = L2ErrorIntegrator(exact_divergence!, Divergence, 2, 1; bonus_quadorder = 1)
println("\nL2error(Id) = $(sqrt(evaluate(L2ErrorEvaluator,Solution[1])))")
println("L2error(div) = $(sqrt(evaluate(L2DivergenceErrorEvaluator,Solution[1])))")
```

Interpolating and evaluating yields a plot via ExtendableGrids.plot (here using the PyPlot Plotter)

```@example EXAMPLE_Minimal
PyPlot.figure(1)
nodevals = zeros(Float64,2,size(xgrid[Coordinates],2))
nodevalues!(nodevals,Solution[1],FES[1])
ExtendableGrids.plot(xgrid, nodevals[1,:]; Plotter = PyPlot)
```
