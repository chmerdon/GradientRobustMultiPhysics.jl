
# Time-Dependent Solvers


## TimeControlSolver
The structure TimeControlSolver can be used to setup a time-dependent solver that can be configured in a similar manner as the time-independent ones (subiterations, nonlinear iterations, linear solvers). As a TimeIntegrationRule so far only BackwardEuler is implemented.

Note that, the time-derivative is added by the TimeControlSolver and is in general not part of the PDEDescription (this is debatable and might change in the future).

```@docs
TimeControlSolver
advance!
```

## Advancing a TimeControlSolver

Moreover there are two functions that advance the TimeControlSolver automatically until a given final time (advance\_until\_time!) is reached or until stationarity is reached (advance\_until\_stationarity!). As an experimental feature, one can add the module DifferentialEquations.jl as the first argument to these methods to let this module run the time integration.


```@docs
advance_until_time!
advance_until_stationarity!
```