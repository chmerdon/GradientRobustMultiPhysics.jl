
# Function Operators

FunctionOperators are building blocks for the weak form and define the operations that should be applied to the trial and test functions (and their discrete representatives) inside some PDEOperator. Below is a list of currently available FunctionOperators. Note, that not all operators an be applied to all finite element types.


| Function operator                          | Description                                   |
| :----------------------------------------- | :-------------------------------------------- |
| [`Identity`](@ref)                         | identity                                      |
| [`IdentityComponent{c}`](@ref)             | identity of c-th component                    |
| [`IdentityDisc{Jump}`](@ref)               | jumps of identity (only over faces)           |
| [`NormalFlux`](@ref)                       | normal flux (function times normal)           | 
| [`Gradient`](@ref)                         | gradient/Jacobian                             |
| [`GradientDisc{Jump}`](@ref)               | jumps of gradient/Jacobian (only over faces)  |
| [`SymmetricGradient`](@ref)                | symmetric part of the gradient                |
| [`Divergence`](@ref)                       | divergence                                    |
| [`CurlScalar`](@ref)                       | curl operator 1D to 2D (rotated gradient)     |
| [`Curl2D`](@ref)                           | curl operator 2D to 1D                        |
| [`ReconstructionIdentity{FEType}`](@ref)   | reconstruction operator into specified FEType |
| [`ReconstructionDivergence{FEType}`](@ref) | divergence of FEType reconstruction operator  |
| [`ReconstructionGradient{FEType}`](@ref)   | gradient of FEType reconstruction operator    |
|                                            |                                               |


!!! note

    Especially note the operators Reconstruction...{FEType} operators that allow to evaluate operators of some
    reconstructed version of a vector-valued testfunction that maps its discrete divergence to the divergence and so allows e.g. gradient-robust discretisations with classical non divergence-conforming ansatz spaces. So far such operators are available for the vector-valued Crouzeix-Raviart and Bernardi--Raugel finite element types.


```@autodocs
Modules = [GradientRobustMultiPhysics]
Pages = ["functionoperators.jl"]
Order   = [:type, :function]
```

