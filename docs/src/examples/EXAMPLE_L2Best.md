### Minimal Example: L2-Bestapproximation

This example is a slightly shorter version of EXAMPLE_Minimal.jl


We first define some (vector-valued) function (to be L2-bestapproximated in this example)

```@example EXAMPLE_Minimal
function exact_function!(result,x)
    result[1] = x[1]^2+x[2]^2
    result[2] = -x[1]^2 + 1.0
end
```

Then, we generate a mesh and refine in a bit

```@example EXAMPLE_Minimal
xgrid = simplexgrid([0.0,1.0],[0.0,1.0])
for j=1:6
    xgrid = uniform_refine(xgrid)
end
```

Next, we define the L2-Bestapproximation problem via a PDEDEscription prototype and investigate it

```@example EXAMPLE_Minimal
Problem = L2BestapproximationProblem(exact_function!, 2, 2; bestapprox_boundary_regions = [1,2,3,4], bonus_quadorder = 2)
show(Problem)
```

To discretise, let us choose some lowest-order Raviart-Thomas finite element space. (You can also try and replace FEType by the vector-valued P1 element FEType = H1P2{2,2})

```@example EXAMPLE_Minimal
FEType = HDIVRT0{2}
FES = FESpace{FEType}(xgrid)
```

Finally, we create a FEVector for the FESpace and solve the problem into it

```@example EXAMPLE_Minimal
Solution = FEVector{Float64}("L2-Bestapproximation",FES)
solve!(Solution, Problem)
```

We also can calculate the L2 error of our discrete solution by

```@example EXAMPLE_Minimal
L2ErrorEvaluator = L2ErrorIntegrator(exact_function!, Identity, 2, 2; bonus_quadorder = 2)
println("\nL2error(Id) = $(sqrt(evaluate(L2ErrorEvaluator,Solution[1])))")
```

Interpolating and evaluating yields a plot via ExtendableGrids.plot (here using the PyPlot Plotter)

```@example EXAMPLE_Minimal
PyPlot.figure(1)
nodevals = zeros(Float64,2,size(xgrid[Coordinates],2))
nodevalues!(nodevals,Solution[1],FES)
ExtendableGrids.plot(xgrid, nodevals[1,:]; Plotter = PyPlot)
```
