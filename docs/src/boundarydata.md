# Boundary Data

## Dirichlet Boundary Data

BoundaryDatas carry the boundary data for each unknown in certain regions. Each regions can have a different AbstractBoundaryType and an associated [Data Function](@ref). This data function than will now if it depends on
space or time variables and will assemble itself accordingly.


| AbstractBoundaryType                | Subtypes                                 | causes                                                                  |
| :---------------------------------- | :--------------------------------------- | :---------------------------------------------------------------------- |
| DirichletBoundary                   |                                          |                                                                         |
|                                     | BestapproxDirichletBoundary              | computation of Dirichlet data by bestapproximation along boundary faces |
|                                     | InterpolateDirichletBoundary             | computation of Dirichlet data by interpolation along boundary faces     |
|                                     | HomogeneousDirichletBoundary             | zero Dirichlet data on all dofs                                         |


```@docs
BoundaryData
add_boundarydata!
```

#### Remarks
- Neumann boundary data can be implemented via a RhsOperator with AT = ON_BFACES and specified boundary regions
- Periodic boundary data can be implemented via CombineDofs (see ExampleA10)
- Symmetry boundary can be implemented by penalisation as an BilinearForm on AT = ON_BFACES and specified boundary regions with operator NormalFlux and some penalty factor.
- InterpolateDirichletBoundary and HomogeneousDirichletBoundary allow for a mask that allows to apply the boundary data only to certain components (epxerimental feature)