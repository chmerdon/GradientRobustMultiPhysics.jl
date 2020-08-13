
# PDE Solvers

If solve! is applied to a PDEDescription and a FEVector (that specifies the ansatz spaces for the unknowns) an investigation of the PDEDescription is performed that decides if the problem is nonlinear (and has to be solved by a fixed-point algorithm) or if it can be solved directly in one step.
Additionally the user can manually trigger subiterations that splits the fixed-point algorithm into substeps where only subsets of the PDE equations are solved together.

Also, there is a preliminary time-dependent solver that can be setup in a similar manner and then performs the subiterations once in eachtimestep. As a TimeIntegrationRule so far only BackwardEuler is implemented.

```@autodocs
Modules = [GradientRobustMultiPhysics]
Pages = ["solvers.jl"]
Order   = [:type, :function]
```