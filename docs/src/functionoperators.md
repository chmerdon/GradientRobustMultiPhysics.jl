
# Function Operators

FunctionOperators are building blocks for the weak form and define the operations that should be applied to the trial and test functions (and their discrete representatives) inside some PDEOperator. Below is a list of currently available FunctionOperators. Note, that not all operators an be applied to all finite element types.


| Function operator                                    | Description                                             |
| :--------------------------------------------------- | :------------------------------------------------------ |
| [`Identity`](@ref)                                   | identity                                                |
| [`IdentityDisc{Jump}`](@ref)                         | jumps of identity (only over faces)                     |
| [`IdentityDisc{Average}`](@ref)                      | average of identity (only over faces)                   |
| [`IdentityComponent{c}`](@ref)                       | identity of c-th component                              |
| [`NormalFlux`](@ref)                                 | normal flux (function times normal)                     |
| [`TangentFlux`](@ref)                                | tangent flux (function times tangent)                   | 
| [`Gradient`](@ref)                                   | gradient/Jacobian (as a vector)                         |
| [`GradientDisc{Jump}`](@ref)                         | jumps of gradient/Jacobian (only over faces)            |
| [`GradientDisc{Average}`](@ref)                      | average of gradient/Jacobian (only over faces)          |
| [`SymmetricGradient`](@ref)                          | symmetric part of the gradient                          |
| [`Divergence`](@ref)                                 | divergence                                              |
| [`CurlScalar`](@ref)                                 | curl operator 1D to 2D (rotated gradient)               |
| [`Curl2D`](@ref)                                     | curl operator 2D to 1D                                  |
| [`Curl3D`](@ref)                                     | curl operator 3D to 3D                                  |
| [`Hessian`](@ref)                                    | Hesse matrix = all 2nd order derivatives (as a vector)  |
| [`Laplacian`](@ref)                                  | Laplace Operator                                        |
| [`ReconstructionIdentity{FEType}`](@ref)             | reconstruction operator into specified FEType           |
| [`ReconstructionIdentityDisc{FEType,Jump}`](@ref)    | jump of reconstruction operator (over faces)            |
| [`ReconstructionIdentityDisc{FEType,Average}`](@ref) | average of reconstruction operator (over faces)         |
| [`ReconstructionDivergence{FEType}`](@ref)           | divergence of FEType reconstruction operator            |
| [`ReconstructionGradient{FEType}`](@ref)             | gradient of FEType reconstruction operator              |
| [`ReconstructionGradientDisc{FEType,Jump}`](@ref)    | jump of reconstruction operator gradient (over faces)   |
| [`ReconstructionGradientDisc{FEType,Average}`](@ref) | average of reconstruction operator gadient (over faces) |
|                                                      |                                                         |


!!! note

    Especially note the operators Reconstruction...{FEType} operators that allow to evaluate operators of some
    reconstructed version of a vector-valued testfunction that maps its discrete divergence to the divergence and so allows e.g. gradient-robust discretisations with classical non divergence-conforming ansatz spaces. So far such operators are available for the vector-valued Crouzeix-Raviart and Bernardi--Raugel finite element types.

!!! note

    As each finite element type is transformed differently from the reference domain to the general domain,
    the evaluation of each function operator has to be implemented for each finite element class.
    Currently, not every function operator works in any dimension and for any finite element. More evaluations
    are added as soon as they are needed (and possibly upon request).


