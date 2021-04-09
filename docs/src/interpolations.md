
# Finite Element Interpolations

Each finite element has its standard interpolator that can be applied to some user-defined DataFunction. Instead of interpolating on the full cells, the interpolation can be restricted to faces or edges, by specifying one of the [Assembly Types](@ref) in the call.

```@autodocs
Modules = [GradientRobustMultiPhysics]
Pages = ["interpolations.jl"]
Order   = [:type, :function]
```
