
# Abstract Actions


## General concept

Parameters (possibly region- or item-dependent) are handle by AbstractAction types and can be assigned to [Assembly Patterns](@ref). The action is applied to some input vector (usually the first argument of some assembly pattern) and then returns the modified result. Each AbstractAction implements the apply! interface

apply!(result,input,::AbstractAction,qp::Int)

that defines how the result vector is computed based on the input and the current quadrature point number.

Before it is applied to any input or quadraturepoint, an update! step is performed upon entry on any item which prepares the data for the evaluation (fixing item and region numbers or computing the global coordinates of the quadrature points). Details of that are hidden here, since it is usually not necessary for the user to interfere with that.




## Implemented Actions

Below each action, its constructor and the resulting apply-effect is explained.


### DoNotChangeAction

```@docs
DoNotChangeAction(resultdim::Int)
```

When applied to some input, the input vector is not changed, i.e.

```math
\texttt{result[j]} = \texttt{input[j]} \quad \text{for } \texttt{j = 1 : resultdim}
```



### MultiplyScalarAction

```@docs
MultiplyScalarAction(value::Real, resultdim::Int = 1)
```

When applied to some input, the input vector is multiplied element-wise with the value of 'value', i.e.

```math
\texttt{result[j]} = \texttt{input[j] * value} \quad \text{for } \texttt{j = 1 : resultdim}
```


### RegionWiseMultiplyScalarAction

```@docs
RegionWiseMultiplyScalarAction(value4region::Array{<:Real,1}, resultdim::Int = 1)
```

When applied to some input, the input vector is multiplied element-wise with the region-coressponding value of 'value4region', i.e.

```math
\texttt{result[j]} = \texttt{input[j] * value4region[region]} \quad \text{for } \texttt{j = 1 : resultdim}
```


### MultiplyVectorAction

```@docs
MultiplyVectorAction(values::Array{<:Real,1})
```

When applied to some input, the input vector is multiplied element-wise with the components of 'vector', i.e.

```math
\texttt{result[j]} = \texttt{input[j] * values[j]} \quad \text{for } \texttt{j = 1 : length(vector)}
```


### RegionWiseMultiplyVectorAction

```@docs
RegionWiseMultiplyVectorAction(values4region::Array{Array{<:Real,1},1}, resultdim::Int)
```

When applied to some input, the input vector is multiplied element-wise with the region-coressponding components of 'values4region', i.e.

```math
\texttt{result[j]} = \texttt{input[j] * values4region[region][j]} \quad \text{for } \texttt{j = 1 : length(vector4regions[region])}
```




### MultiplyMatrixAction

```@docs
MultiplyMatrixAction(matrix::Array{<:Real,2})
```

When applied to some input, the input vector is multiplied with the 'matrix', i.e.

```math
\texttt{result} = \texttt{matrix * input}
```

 

### FunctionAction

```@docs
FunctionAction(f::Function, resultdim::Int = 1, xdim::Int = 2; bonus_quadorder::Int = 0)
```

When applied to some input, the input vector is evaluated by the specified function, i.e.

```math
\texttt{function!(result, input)}
```

Note, that the function here cannot depend on x, item or region in this case. If further dependencies are needed check the other variants.



### XFunctionAction

```@docs
XFunctionAction(f::Function, resultdim::Int = 1, xdim::Int = 2; bonus_quadorder::Int = 0)
```

When applied to some input, the input vector is evaluated by the specified x-dependent function, i.e.

```math
\texttt{function!(result, input, x)}
```

Note, that for this action additional global coordinates for the quadrature points are computed in the update! step of this action. So this action causes some more computation time and should be avoided if the x-dependency is not needed.


### ItemWiseXFunctionAction

```@docs
ItemWiseXFunctionAction(f::Function, resultdim::Int = 1, xdim::Int = 2; bonus_quadorder::Int = 0)
```

When applied to some input, the input vector is evaluated by the specified x- and item-dependent function, i.e.

```math
\texttt{function!(result, input, x, item)}
```

### RegionWiseXFunctionAction

```@docs
RegionWiseXFunctionAction(f::Function, resultdim::Int = 1, xdim::Int = 2; bonus_quadorder::Int = 0)
```

When applied to some input, the input vector is evaluated by the specified x- and region-dependent function, i.e.

```math
\texttt{function!(result, input, x, region)}
```
