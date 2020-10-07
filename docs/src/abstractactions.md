
# Abstract Actions


## General concept

Parameters (possibly region- or item-dependent) are handle by AbstractAction types and can be assigned to [Assembly Patterns](@ref). The action is applied to some input vector, that collects the evaluations of the function operators of (a well-defined subset of the) arguments of the assembly pattern, and then returns the modified result. Each AbstractAction implements the apply! interface

apply!(result,input,::AbstractAction,qp::Int)

that defines how the result vector is computed based on the input and the current quadrature point number.

Before it is applied to any input or quadrature point, an update! step is performed upon entry on any item which prepares the data for the evaluation (fixing item and region numbers or computing the global coordinates of the quadrature points). Details of that are hidden here, since it is usually not necessary for the user to interfere with that.

Below is a table with implemented action types and some short description what it can be used for. Click on the action name for more details.

| Action type                                  | What it does                                                                                            |
| :------------------------------------------- | :------------------------------------------------------------------------------------------------------ |
| [`DoNotChangeAction`](@ref)                  | does nothing  (result = input)                                                                          |
| [`MultiplyScalarAction`](@ref)               | multiplies argument (vector) with a scalar                                                              |
| [`RegionWiseMultiplyScalarAction`](@ref)     | multiplies argument (vector) with a scalar that depends on the region                                   |
| [`MultiplyVectorAction`](@ref)               | multiplies argument (vector) componentwise with a vector of same length                                 |
| [`RegionWiseMultiplyVectorAction`](@ref)     | multiplies argument (vector) componentwise with a region-depending vector of same length                |
| [`MultiplyMatrixAction`](@ref)               | matrix-vector multiplication, input is multiplied with the specified matrix                             |
| [`FunctionAction`](@ref)                     | f!(result, input) for some user-specified function with this interface                                  |
| [`XFunctionAction`](@ref)                    | f!(result, input, x) for some user-specified function (that also depends on x)                          |
| [`ItemWiseXFunctionAction`](@ref)            | f!(result, input, x, item) for some user-specified function (that also depends on x and item number)    |
| [`RegionWiseXFunctionAction`](@ref)          | f!(result, input, x, region) for some user-specified function (that also depends on x and region id)    |




```@autodocs
Modules = [GradientRobustMultiPhysics]
Pages = ["actions.jl"]
Order   = [:type, :function]
```