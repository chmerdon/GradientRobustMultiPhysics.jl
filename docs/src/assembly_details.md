
# Assembly Details

The assembly of an operator is essentially based on a combination of [Assembly Types](@ref) and [Assembly Patterns](@ref). The assembly type mainly allows to choose the geometry information needed for providing quadrature and dof handling. The assembly pattern then basically evaluates the
function operators and action for the ansatz and test functions and does the quadrature-weighted accumulation into matrices or vectors that represent the operators.


## Assembly Types

The following assembly patterns are available.

| Assembly Type    | Description                                   |
| :--------------- | :-------------------------------------------- |
| ON_CELLS         | assembles over the cells of the mesh          |
| ON_FACES         | assembles over all faces of the mesh          |
| ON_IFACES (*)    | assembles over the interior faces of the mesh |
| ON_BFACES        | assembles over the boundary faces of the mesh |

!!! note
    (*) = still experimental, might have some issues


## Assembly Patterns

Each Pattern comes with a number of arguments/quantities with associated [Function Operators](@ref) as well as one of the [Assembly Types](@ref) that states whether the form is evaluated over CELLS, FACES order BFACES (see above). Moreover, patterns can have [Abstract Actions](@ref) that allow to make the evaluations parameter-, region- and/or function-dependent. Each pattern then has two implementation that writes into FEMatrix or FEVector (where e.g. a subset of arguments is fixed).

The patterns are used by the assembly of the PDE operators defined in a [PDE Description](@ref). However, it is also possible for the user to use them directly, see e.g. the example [2D Commuting Interpolators](@ref).


```@autodocs
Modules = [GradientRobustMultiPhysics]
Pages = ["assemblypatterns.jl"]
Order   = [:type, :function]
```