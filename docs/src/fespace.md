
# Finite Element Spaces and Arrays

This page describes the structure FESpace that acts as a finite element space on a given grid.
See [Implemented Finite Elements](@ref) for a list of available finite element types.

Moreover, there are special arrays FEVector and FEMatrix that carry coefficients and assembled operators.


## FESpace


```@docs
FESpace{AbstractFiniteElement}
eltype(::FESpace)
show(::IO, ::FESpace)
```


## FEVector and FEVectorBlock


```@autodocs
Modules = [GradientRobustMultiPhysics]
Pages = ["FEVector.jl"]
Order   = [:type, :function]
```


## FEMatrix and FEMatrixBlock


```@autodocs
Modules = [GradientRobustMultiPhysics]
Pages = ["FEMatrix.jl"]
Order   = [:type, :function]
```