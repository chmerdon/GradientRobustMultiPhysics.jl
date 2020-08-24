
# Function Operators

FunctionOperators are building blocks for the weak form and define the operations that should be applied to the trial and test functions (and their discrete representatives) inside some PDEOperator. Below is a list of currently available FunctionOperators. Note, that not all operators an be applied to all finite element types.


| Function operator                          | Description                                   |
| :----------------------------------------- | :-------------------------------------------- |
| [`Identity`](@ref)                         | identity operator                             |
| [`FaceJumpIdentity`](@ref)                 | jumps of identity operator over faces         |
| [`NormalFlux`](@ref)                       | normal flux (function times normal)           | 
| [`Gradient`](@ref)                         | gradient/Jacobian operator                    |
| [`SymmetricGradient`](@ref)                | symmetric part of the gradient                |
| [`Divergence`](@ref)                       | divergence operator                           |
| [`CurlScalar`](@ref)                       | curl operator 1D to 2D (rotated gradient)     |
| [`ReconstructionIdentity{FEType}`](@ref)   | reconstruction operator into specified FEType |
| [`ReconstructionDivergence{FEType}`](@ref) | divergence of FEType reconstruction operator  |
|                                            |                                               |


!!! note

    Especially note the operators ReconstructionIdentity{FEType} and ReconstructionDivergence{FEType} that allow to evaluate some
    reconstructed version of a vector-valued testfunction that maps its discrete divergence to the divergence and so allows e.g. gradient-robust discretisations with classical non divergence-conforming ansatz spaces. So far such operators are available for the vector-valued Crouzeix-Raviart and Bernardi--Raugel finite element types.


```@autodocs
Modules = [GradientRobustMultiPhysics]
Pages = ["functionoperators.jl"]
Order   = [:type, :function]
```

