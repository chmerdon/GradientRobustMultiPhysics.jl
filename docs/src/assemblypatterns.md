
# Assembly Patterns

Assembly is reduced to one of the following patterns. Each Pattern comes with a number of arguments/quantities with associated AbstractFunctionOperators as well as an AbstractAssemblyType that states whether the form is evaluated over CELLS, FACES order BFACES. Moreover, patterns can have [Abstract Actions](@ref) that allow to make the evaluations parameter-, region- and/or function-dependent. Each Pattern then allows different types of their assembly into FEVector or FEMatrix where e.g. a subset of arguments is fixed.

Examples:
- the standard piecewise L2 scalar product for a given scalar-valued FESpace FES can be realized via
  Bilineaform{Float64,AssemblyTypeCELL}(FES,Identity,DoNotChangeAction(1))
- the standard piecewise H1 scalar product for a given scalar-valued FESpace FES in 2D and global viscosity coefficient mu can be realized via
  Bilineaform{Float64,AssemblyTypeCELL}(FES,Gradient,MultiplyScalarAction(mu,2))


```@autodocs
Modules = [JUFELIA]
Pages = ["AbstractAssemblyPattern.jl"]
Order   = [:type, :function]
```