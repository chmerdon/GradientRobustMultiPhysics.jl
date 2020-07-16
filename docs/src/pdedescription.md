
# PDE description

PDEs are described as a set of operators arranged in a matrix. The number of rows of this matrix is the number of partial differential equations in the system. The number of columns is the number of unknowns. PDEoperators are independent of any finite element space and hence allows a closer description of the continuous level. There are several prototype PDEs that can be used as a point of departure.

```@autodocs
Modules = [JUFELIA]
Pages = ["PDEDescription.jl"]
Order   = [:type, :function]
```

## PDE Operators


```@autodocs
Modules = [JUFELIA]
Pages = ["PDEOperators.jl"]
Order   = [:type, :function]
```


## Global Constraints

```@autodocs
Modules = [JUFELIA]
Pages = ["PDEGlobalConstraints.jl"]
Order   = [:type, :function]
```


## Boundary Data

BoundaryOperators carry the boundary data for each unknown. Each regions can have a different AbstractBoundaryType. 

So far only DirichletBoundaryData is possible, as most other types can be implemented differently:
- NeumannBoundary can be implemented as a RhsOperator with on_boundary = true
- PeriodicBoundarycan be implemented as a CombineDofs <: AbstractGlobalConstraint
- SymmetryBoundary can be implemented as a RHS AbstractBilinearForm on BFaces and specified regions with operator NormalFlux + MultiplyScalarAction(penalty).

```@autodocs
Modules = [JUFELIA]
Pages = ["PDEBoundaryData.jl"]
Order   = [:type, :function]
```