
# User Data and Actions

There is a variety of different user data, like scalar- and vector-valued constants, time-dependent data, region-dependent data or plain functions that
depend on the the space coordinates. Also dependency on the item number of the reference coordinates of the quadrature point in the quadrature item are sometimes desireable.
To allow for flexible user-specified data, all functions have to be negotiated by the UserData interface that fixes
the order and number of the arguments in the interface via a user-given substring of "XTIL" where each character stands for a dependency.
The following table explains the meaning of each character.


| Character          | Explanation                                                |
| :----------------: | :--------------------------------------------------------- | 
| X                  | depends on (vector-valued) space coordinates               | 
| T                  | depends on time coordinate                                 | 
| I                  | depends on item information (item nr, parent nr, region)   |
| L                  | depends on local coordinates in reference geometry of item |

Also note that all functions are expected to write their result into the first argument.


## Data Function

DataFunctions can be used to define boundary data, right-hand side functions and can be interpolated by the finite element standard interpolations. The have to be conform to the interface

```julia
function datafunction_kernel!(result,[X,T,I,L])
    # X = space coordinates
    # T = time
    # I = item information (vector with item number (w.r.t. AT), parent number and region  number)
    # L = local coordinates on item reference domain
end
```

```@docs
DataFunction
```

There are also derivatives defined for DataFunctions that generate another DataFunction where the derivative is calculated via ForwardDiff.

```@docs
DataFunction
∇
div
curl
Δ 
```


## Action

Actions are used by abstract user-defined PDEOperators and consist of an action kernel function of the interface

```julia
function action_kernel!(result,input,[X,T,I,L])
    # result = modified input, possibly depended on
    # X = space coordinates
    # T = time
    # I = item information (vector with item number (w.r.t. AT), parent number and region  number)
    # L = local coordinates on item reference domain
end
```

plus some additional infrastructure like expected dimensiona of result and input and further dependencies


```@docs
Action
```