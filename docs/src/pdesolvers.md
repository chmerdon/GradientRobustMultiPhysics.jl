
# PDE Solvers


## Fixed-Time Solvers

If solve! is applied to a PDEDescription and a FEVector (that specifies the ansatz spaces for the unknowns) an investigation of the PDEDescription is performed that decides if the problem is nonlinear (and has to be solved by a fixed-point algorithm) or if it can be solved directly in one step.
Additionally the user can manually trigger subiterations that splits the fixed-point algorithm into substeps where only subsets of the PDE equations are solved together.

```@docs
solve!
```


## Time-Dependent Solvers

The structure TimeControlSolver can be used to setup a time-dependent solver that can be configured in a similar manner as the time-independent ones (subiterations, nonlinear iterations, linear solvers). As a TimeIntegrationRule so far only BackwardEuler is implemented.

Note that, the time-derivative is added by the TimeControlSolver and is in general not part of the PDEDescription (this is debatable and might change in the future).

```@docs
TimeControlSolver
advance!
```

Moreover there are two functions that advance the TimeControlSolver automatically until a given final time (advance\_until\_time!) is reached or until stationarity is reached (advance\_until\_stationarity!). As an experimental feature, one can add the module DifferentialEquations.jl as the first argument to these methods to let this module run the time integration.


```@docs
advance_until_time!
advance_until_stationarity!
```