# Global Constraints

GlobalConstraints are additional constraints that the user does not wish to implement as a global Lagrange multiplier because it e.g. causes a dense row in the system matrix and therefore may destroy the performance of the sparse matrix routines. Such a constraint may be a fixed integral mean. Another application are periodic boundary conditions or glued-together quantities in different regions of the grid. Here a CombineDofs constraint may help.

```@autodocs
Modules = [GradientRobustMultiPhysics]
Pages = ["globalconstraints.jl"]
Order   = [:type, :function]
```

```@docs
add_constraint!
```
