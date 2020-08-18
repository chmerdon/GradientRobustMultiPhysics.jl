
# Assembly Patterns

Assembly is reduced to one of the following patterns. Each Pattern comes with a number of arguments/quantities with associated AbstractFunctionOperators as well as an AbstractAssemblyType that states whether the form is evaluated over CELLS, FACES order BFACES. Moreover, patterns can have [Abstract Actions](@ref) that allow to make the evaluations parameter-, region- and/or function-dependent. Each Pattern then allows different types of their assembly into FEVector or FEMatrix where e.g. a subset of arguments is fixed.

Usually the patterns are used by the assembly of a [PDE Description](@ref). However, it is also possible for the user to use them directly, see e.g. the example [2D Commuting Interpolators](@ref).


```@autodocs
Modules = [GradientRobustMultiPhysics]
Pages = ["assemblypatterns.jl"]
Order   = [:type, :function]
```