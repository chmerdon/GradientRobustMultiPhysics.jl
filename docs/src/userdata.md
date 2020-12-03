
# User Data and Action Kernels

There is a variety of different user data, like scalar- and vector-valued constants, time-dependent data, region-dependent data or plain functions that
depend on the the space coordinates. To allow for flexible user-specified data, all functions have to be negotiated by the UserData interface that fixes
the order and number of dependencies.

## Data Functions

The simplest form of user data is called DataFunction and satisfies the interface

```@example
function data_function!(result,x,t)
    # definition of result, possibly dependent on
    # X = space coordinates
    # T = time
end
```

where the result vector carries the result of the function. Additionally, it can (but not has to) depend on the further inputs X (space coordinates) or T (time) DataFunctions can be used to define boundary data, right-hand side functions and can be interpolated by the finite element standard interpolations.

There are also ExtendedDataFunction that can additionally depend on R (region), I (item number) and L (local coordinates). The dependencies are stated via a string in the constructor that should be a substring of "XTRIL". However, extended data functions cannot be used everywhere.

```@docs
DataFunction
ExtendedDataFunction
```


## Action Kernels

Another for of user data are action kernel used to define an AbstractAction. Actions modify arguments of [Assembly Patterns](@ref) (usually all but the last one) and so allow parameter-depend assemblies. To use them, the user defines some kernel function for the action that has the interface

```@example
function action_kernel!(result,input,X,T,R,I,L)
    # result = modified input, e.g.
    # multiplication with some parameter that can depend on
    # X = space coordinates
    # T = time
    # R = region number
    # I = item number (cell, face or edge number depending on assembly type)
    # L = local coordinates on item reference domain
end
```

During assembly, input (in general) takes the role of all non-testfunction arguments and the result vector will be the one that is multiplied with the testfunctions. Additionally, the kernel function can depend on X, T, R, I and L as specified above. Again, the dependencies of an action_kernel are stated via a string in the constructor that should be a substring of "XTRIL". Note, that the avoidance of the X-dependency spares the computation of the global coordinates of
the quadrature points in the assembly loops. Moreover, time-dependency of an action kernel e.g. triggers reassembly of the associated PDEoperator in each time step.

```@docs
ActionKernel
NLActionKernel
```


## Actions

Actions are used by [Assembly Patterns](@ref) and consist of an action kernel plus some additional infratructure. To generate an action from an action kernel works via the following functions.

```@docs
Action
```

Moreover, there are some shortcut action constructors that can be used directly without defining an action kernel first.

```@docs
MultiplyScalarAction
DoNotChangeAction
```
