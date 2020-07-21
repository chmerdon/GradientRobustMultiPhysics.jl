
# PDE description

PDEs are described as a set of operators arranged in a matrix. The number of rows of this matrix is the number of partial differential equations in the system. The number of columns is the number of unknowns. PDEoperators are independent of any finite element space and hence allows a closer description of the continuous level. There are several prototype PDEs documented on the [PDE Prototypes](@ref) page that can be used as a point of departure.

```@autodocs
Modules = [GradientRobustMultiPhysics]
Pages = ["PDEDescription.jl"]
Order   = [:type, :function]
```

## PDE Operators

The PDE consists of PDEOperators characterising some feature of the model (like friction, convection, exterior forces etc.), they describe the continuous weak form of the PDE.

```@autodocs
Modules = [GradientRobustMultiPhysics]
Pages = ["PDEOperators.jl"]
Order   = [:type, :function]
```

## Function Operators

FunctionOperators are building blocks for the weak form and define the operations that should be applied to the trial and test functions inside some PDEOperator. Below is a list of available FunctionOperators. Especially note the Reconstruction...-Operators that allow to evaluate some reconstructed version of a trial or test function and so allows e.g. gradient-robust discretisations with classical ansatz spaces.

```@autodocs
Modules = [GradientRobustMultiPhysics]
Pages = ["FEBasisEvaluator.jl"]
Order   = [:type, :function]
```


## Global Constraints

GlobalConstraints are additional constraints that the user does not wish to implement as a global Lagrange multiplier because it e.g. causes a dense row in the system matrix and therefore may destroy the performance of the sparse matrix routines. Such a constraint may be a fixed integral mean. Another application are periodic boundary conditions or glued-together quantities in different regions of the grid. Here a CombineDofs constraint may help.

```@autodocs
Modules = [GradientRobustMultiPhysics]
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
Modules = [GradientRobustMultiPhysics]
Pages = ["PDEBoundaryData.jl"]
Order   = [:type, :function]
```