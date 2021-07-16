
# Time-Dependent Solvers


## TimeControlSolver
The structure TimeControlSolver can be used to setup a time-dependent solver that can be configured in a similar manner as the time-independent ones (subiterations, nonlinear iterations, linear solvers). The following table lists the available TimeIntegrationRules:

| Time integration rule | order | Formula       
| :-------------------: | :---: | :-----------------------------------------------------------------------: |
| BackwardEuler         |   1   | ``(M^{n+1} + A^{n+1}) u^{n+1} = F^{n+1} + M^{n+1} u^n``                   |
| CrankNicolson         |   2   | ``(M^{n} + A^{n+1}) u^{n+1} = F^{n+1} + F^n + M^{n+1} u^n - A^n u^{n+1}`` |

Note that currently the time-derivative (M terms) is added by the TimeControlSolver in each integration step and is in general not part of the PDEDescription
(this might change in future). The default time derivative is a scaled (depends on the integration rule) mass matrix of the used finite element space, but the user can overwrite it via optional constructor arguments (experimental).

```@docs
TimeControlSolver
advance!
```

## Advancing a TimeControlSolver

There are two functions that advance the TimeControlSolver automatically until a given final time (advance\_until\_time!) is reached or until stationarity is reached (advance\_until\_stationarity!). As an experimental feature, one can add the module DifferentialEquations.jl as the first argument to these methods to let this module run the time integration (the native TimeIntegrationRule argument in the TimeControlSolver constuctor is ignored in this case).


```@docs
advance_until_time!
advance_until_stationarity!
```