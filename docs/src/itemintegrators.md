# Item Integrators

Item integrators are one of the [Assembly Patterns](@ref)
that help to compute certain quantities of the Solution, like a posteriori errors estimators, norms, drag/lift coefficients or other statistics.

```@docs
ItemIntegrator
L2ErrorIntegrator
L2NormIntegrator
L2DifferenceIntegrator
```

## Evaluation

There are two possibilities to evaluate an ItemIntegrator, on each item (with evaluate!) or globally (with evaluate):

```@docs
evaluate!
evaluate
```

#### Noteworthy Examples

Examples 204 and A06 use ItemIntegrators for a posteriori error estimation and refinement indicators.

Example 224 uses ItemIntegrators to calculate drag and lift coefficients.

