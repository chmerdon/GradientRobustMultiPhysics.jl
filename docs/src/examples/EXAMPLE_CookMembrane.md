### Linear Elasticity: Cook-Membrane

This example is a shortened version of EXAMPLE_CookMembrane.jl.


We first load the CookMembrane mesh (predefined inside the package) and refine it a bit

```@example EXAMPLE_CookMembrane
    xgrid = testgrid_cookmembrane() # initial simplex grid
    for j=1:4
        xgrid = uniform_refine(xgrid)
    end
```

Next is the function for the Neumann data on the right side of the Cook membrane (upwards pulling force)

```@example EXAMPLE_CookMembrane
function neumann_force_right!(result,x)
    result[1] = 0.0
    result[2] = 10.0
end
```

Further, there are some material parameters

```@example EXAMPLE_CookMembrane
elasticity_modulus = 1000; # elasticity modulus
poisson_number = 1//3; # Poisson number
shear_modulus = (1/(1+poisson_number))*elasticity_modulus;
lambda = (poisson_number/(1-2*poisson_number))*shear_modulus;
```

With this we generate the PDE description via the LinearElasticity prototype

```@example EXAMPLE_CookMembrane
LinElastProblem = LinearElasticityProblem(2; shearmodulus = shear_modulus, lambda = lambda)
```

and add Neumann boundary and Dirichlet data via

```@example EXAMPLE_CookMembrane
add_rhsdata!(LinElastProblem, 1,  RhsOperator(Identity, neumann_force_right!, 2, 2; regions = [3], on_boundary = true, bonus_quadorder = 0))
add_boundarydata!(LinElastProblem, 1, [1], HomogeneousDirichletBoundary)
```

For the discretisation, let us choose a quadratic finite element for the FESpace

```@example EXAMPLE_CookMembrane
FEType = H1P2{2,2} # P2
FES = FESpace{FEType}(xgrid)
```

Finally, we solve the problem by

```@example EXAMPLE_CookMembrane
Solution = FEVector{Float64}("Displacement",FES)
solve!(Solution, LinElastProblem)
```

In the following three plots we plot the triangulation, the absolute value of the displacement and the (by a factor 4) displaced mesh

```@example EXAMPLE_CookMembrane
PyPlot.figure(1)
xgrid = split_grid_into(xgrid,Triangle2D)
ExtendableGrids.plot(xgrid, Plotter = PyPlot)
PyPlot.figure(2)
nnodes = size(xgrid[Coordinates],2)
nodevals = zeros(Float64,3,nnodes)
nodevalues!(nodevals,Solution[1],FES)
nodevals[3,:] = sqrt.(nodevals[1,:].^2 + nodevals[2,:].^2)
ExtendableGrids.plot(xgrid, nodevals[3,:]; Plotter = PyPlot)
xgrid[Coordinates] = xgrid[Coordinates] + 4*nodevals[[1,2],:]
PyPlot.figure(3)
ExtendableGrids.plot(xgrid, Plotter = PyPlot)
```
