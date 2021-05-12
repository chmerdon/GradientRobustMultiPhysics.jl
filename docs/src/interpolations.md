
# Finite Element Interpolations

## Standard interpolations

Each finite element has its standard interpolator that can be applied to some user-defined DataFunction. Instead of interpolating on the full cells, the interpolation can be restricted to faces or edges, by specifying an [Assembly Type](@ref) in the call.

```@docs
interpolate!
```

## Nodal evaluations

Usually, Plotters need nodal values, so there is a gengeric function that evaluates any finite element function at the nodes of the grids (possibly by averaging if discontinuous).


```@docs
nodevalues!
```