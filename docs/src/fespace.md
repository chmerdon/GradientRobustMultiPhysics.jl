
# Finite Element Spaces and Arrays

This page describes the structure [FESpace](@ref) that acts as a finite element space on a given grid.
See [Implemented Finite Elements](@ref) for a list of available finite element types.

Moreover, there are special arrays [FEVector](@ref) and [FEMatrix](@ref) that carry coefficients and discretised PDEOperators.


## FESpace


```@docs
FESpace{AbstractFiniteElement}
eltype(::FESpace)
show(::IO, ::FESpace)
```


## FEVector

A FEVector consists of FEVectorBlocks that share a common one-dimensional arrays. Each block is associated to a FESpace and can only write into a region of the common array specified by offsets. It also acts as a one-dimensional AbstractArray itself.


```@autodocs
Modules = [GradientRobustMultiPhysics]
Pages = ["fevector.jl"]
Order   = [:type, :function]
```


## FEMatrix

A FEMatrix consists of FEMatrixBlocks that share a common ExtendableSparseMatrix. Each block is associated to two FESpaces and can only write into a submatrix of the common sparse matrix specified by offsets. It also acts as a two-dimensional AbstractArray itself.

```@autodocs
Modules = [GradientRobustMultiPhysics]
Pages = ["fematrix.jl"]
Order   = [:type, :function]
```