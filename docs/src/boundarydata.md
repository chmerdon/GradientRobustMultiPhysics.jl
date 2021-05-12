# Boundary Data

## Dirichlet Boundary Data

BoundaryOperators carry the boundary data for each unknown. Each regions can have a different AbstractBoundaryType and an associated [Data Function](@ref). This data function than will now if it depends on
space or time variables and will assemble itself accordingly.


| AbstractBoundaryType                | Subtypes                                 | causes                                                                  |
| :---------------------------------- | :--------------------------------------- | :---------------------------------------------------------------------- |
| DirichletBoundary                   |                                          |                                                                         |
|                                     | BestapproxDirichletBoundary              | computation of Dirichlet data by bestapproximation along boundary faces |
|                                     | InterpolateDirichletBoundary             | computation of Dirichlet data by interpolation along boundary faces     |
|                                     | HomogeneousDirichletBoundary             | zero Dirichlet data on all dofs                                         |


```@docs
BoundaryOperator
add_boundarydata!
```

## Other Boundary Data

NeumannBoundary can be implemented as a RhsOperator with AT = ON_BFACES

PeriodicBoundary can be implemented as a CombineDofs <: AbstractGlobalConstraint

SymmetryBoundary can be implemented by penalisation as a AbstractBilinearForm on AT = ON_BFACES and specified boundary regions with operator NormalFlux and some penalty factor.
