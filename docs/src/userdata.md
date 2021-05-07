
# User Data and Action Kernels

There is a variety of different user data, like scalar- and vector-valued constants, time-dependent data, region-dependent data or plain functions that
depend on the the space coordinates. Also dependency on the item number of the reference coordinates of the quadrature point in the quadrature item arr sometimes desireable.
To allow for flexible user-specified data, all functions have to be negotiated by the UserData interface that fixes
the order and number of the arguments in the interface via a user-given substring of "XTRIL" where each character stands for a dependency.
The following table explains the meaning of each character.


| Character          | Explanation                                                |
| :----------------: | :--------------------------------------------------------- | 
| X                  | depends on (vector-valued) space coordinates               | 
| T                  | depends on time coordinate                                 | 
| R                  | depends on region number                                   | 
| I                  | depends on item number                                     |
| L                  | depends on local coordinates in reference geometry of item |

Also note that all functions are expected to write their result into the first argument.


## Data Function

The simplest form of user data is called DataFunction which allows additional dependencies on space or time coordinates. The following tables lists all allowed substrings of "XTRIL"
and the expected interface of the function provided by the user.


| dependency string  | Expected interface                                                     |
| :----------------: | :--------------------------------------------------------------------- | 
| ""                 | function f!(result) ... end  (constant data)                           | 
| "X"                | function f!(result,x) ... end  (space-dependent data)                  | 
| "T"                | function f!(result,t) ... end  (time-dependent constant-in-space data) | 
| "XT"               | function f!(result,x,t) ... end  (space and time-dependent data)       |

DataFunctions can be used to define boundary data, right-hand side functions and can be interpolated by the finite element standard interpolations.

```@docs
DataFunction
```


## Extended Data Function

There are also ExtendedDataFunction that allow the additional dependencies R (region), I (item number) and L (local coordinates). The dependencies are stated via a string in the constructor that should be a substring of "XTRIL". However, extended data functions cannot be used everywhere.

```@docs
ExtendedDataFunction
```


## Action Kernel

Another form of user data are action kernels that are used to define an AbstractAction. Actions modify (usually a subset of) arguments of [Assembly Patterns](@ref) and so allow parameter-dependent assemblies. 
Also, a trilinear form always needs an action that holds instructions how to prepare the first two arguments such that it can be evaluated with the testfunction operator.
To use them, the user defines some kernel function for the action that has the interface. Kernel functions also allow the full range of dependencies, hence any substring of "XTRIL".

However, between the result argument and the further dependencies they get an input argument which (during assembly) carries the operator-evaluations of the arguments that go into the action. Hence the usual interface
of a action kernel function looks like this:

```@example
function action_kernel!(result,input,[X,T,R,I,L])
    # result = modified input, e.g.
    # multiplication with some parameter that can depend on
    # X = space coordinates
    # T = time
    # R = region number
    # I = item number (cell, face or edge number depending on assembly type)
    # L = local coordinates on item reference domain
end
```

During assembly, input (in general) takes the role of all non-testfunction arguments and the result vector will be the one that is multiplied with the testfunctions. Additionally, the kernel function can depend on X, T, R, I and L as specified above. Once again note, that time-dependency of the kernel function is inherited to the action and later to the whole PDEOperator and so triggers reassembly of the associated PDEoperator in each time step.

```@docs
ActionKernel
```


## Action

Actions are used by abstract user-defined PDEOperators and consist of an action kernel plus some additional infrastructure. To generate an action from an action kernel or directly from a function
works via the following constructors.

```@docs
Action
```

Moreover, there are some shortcut action constructors that can be used directly without defining an action kernel first.

```@docs
NoAction
MultiplyScalarAction
```


## NLAction Kernel

For the manual linearisation (=without automatic differentiation) of [Nonlinear Operators](@ref) assembly pattern, the user can also insert nonlinear action kernels that have a second input argument for the operator-evaluations of the current solution. But since this feature is still experimental, it will not explained in more detail, yet.

```@docs
NLActionKernel
```

