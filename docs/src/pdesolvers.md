
# Fixed-Time Solvers


## Fixed-Time Solvers

If solve! is applied to a PDEDescription and a FEVector (that specifies the ansatz spaces for the unknowns) an investigation of the PDEDescription is performed that decides if the problem is nonlinear (and has to be solved by a fixed-point algorithm) or if it can be solved directly in one step.
Additionally the user can manually trigger subiterations that splits the fixed-point algorithm into substeps where
only subsets of the PDE equations are solved together.

```@docs
solve!
```


## Anderson acceleration

Fixpoint iterations my be accelerated by Anderson acceleration. Concepts and some theoretical background can be found in the reference below. Within this package, Anderson acceleration can be triggered by optional solver arguments: the user can specify the depth of the Anderson acceleration (anderson\_iterations), the damping within the Anderson iteration (anderson\_damping), the ids of the unknowns that should be included in the iteration (anderson\_unknowns) and the convergence metric (anderson\_metric); also see above for a full list of optional solver arguments. In case of subiterations, the Anderson iteration will be called as a postprocessing after the final subiteration.

Reference:

"A Proof That Anderson Acceleration Improves the Convergence Rate in Linearly Converging Fixed-Point Methods (But Not in Those Converging Quadratically)",\
C. Evans, S. Pollock, L. Rebholz, and M. Xiao,\
SIAM J. Numer. Anal., 58(1) (2020),\
[>Journal-Link<](https://doi.org/10.1137/19M1245384)
