
# Finite Element Interpolations

## Standard interpolations

Each finite element has its standard interpolator that can be applied to some user-defined DataFunction. Instead of interpolating on the full cells, the interpolation can be restricted to faces or edges, by specifying an [Assembly Type](@ref) in the call. 

It is also possible to interpolate finite element functions on one grid onto a finite element function on another grid (experimental feature, does not work for all finite elements yet and shall be extended to
interpolations of operator evaluations as well in future).

```@docs
interpolate!
```

## Nodal evaluations

Usually, Plotters need nodal values, so there is a gengeric function that evaluates any finite element function at the nodes of the grids (possibly by averaging if discontinuous).


```@docs
nodevalues!
```