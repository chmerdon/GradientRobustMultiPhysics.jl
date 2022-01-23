
# Assembly Patterns

The definition and assembly of an operator is essentially based on Assembly Patterns and a [Assembly Type](@ref) to choose the geometry information needed for providing quadrature and dof handling. The assembly pattern then basically evaluates the function operators and action for the ansatz and test functions and does the quadrature-weighted accumulation into matrices or vectors that represent the operators.

Each pattern comes with a number of arguments/quantities with associated [Function Operators](@ref) as well as one of the [Assembly Type](@ref) that states whether the form is evaluated over CELLS, FACES order BFACES (see above). Important note: this assembly type is relative to the grid of the first argument of the pattern. If this argument already lives ON_FACES and the pattern is also ON_FACES, it will ultimatively assemble on the faces of the faces (that are the edges of the grid with these faces). Moreover, patterns can have an [Action](@ref) that allow to make the evaluations parameter-, region- and/or function-dependent. Each pattern then has usually on to three implementation that writes into FEMatrix or FEVector (where e.g. a subset of arguments is fixed) or evaluates the pattern in the given FEVectorBlocks.

The patterns are used to assembly the PDE operators defined in a [PDE Description](@ref).

```@docs
GradientRobustMultiPhysics.AssemblyPattern{APT <: AssemblyPatternType, T <: Real, AT <: AssemblyType}
```

The following table lists all available assembly patterns, their constuctor names and how they can be used for assembly or evaluations.


| AssemblyPatternType | constructor        | evaluate | assembles into matrix | assembles into vector |
| :------------------ | :----------------- | :------: | :-------------------: | :-------------------: |
| APT_ItemIntegrator  | ItemIntegrator     |    yes   |           no          |          no           |
| APT_LinearForm      | LinearForm         |     no   |           no          |         yes           |
| APT_BilinearForm    | BilinearForm       |     no   |          yes          |          no           |
| APT_NonlinearForm   | NonlinearForm      |     no   |          yes          |         yes           |

Number in brackets denotes the number of fixed arguments needed for this assembly, (L) means that a current solution is needed (to evaluate the linearisation of the nonlinear form in this state).
Evaluations of the other AssemblyPatterns may be possible in a future update, but currently have to be performed by maintaining a duplicate of the pattern rewritten as an ItemIntegrator.


#### Constructor details

Below all assembly pattern types, constructor functions and evaluate/assembly functions are detailed. (For more on the ItemIntegrator also see [Item Integrators](@ref).)

```@autodocs
Modules = [GradientRobustMultiPhysics]
Pages = ["itemintegrator.jl","linearform.jl","bilinearform.jl","nonlinearform.jl"]
Order   = [:type, :function]
```

