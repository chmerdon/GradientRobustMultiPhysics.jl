
# Function Operators

FunctionOperators are building blocks for the weak form and define the operations that should be applied to the trial and test functions (and their discrete representatives) inside some PDEOperator. Below is a list of currently available FunctionOperators. Note, that not all operators an be applied to all finite element types.


| Function operator                                    | Description                                             |
| :--------------------------------------------------- | :------------------------------------------------------ |
| [`Identity`](@ref)                                   | identity                                                |
| [`IdentityComponent{c}`](@ref)                       | identity of c-th component                              |
| [`NormalFlux`](@ref)                                 | normal flux (function times normal)                     |
| [`TangentFlux`](@ref)                                | tangent flux (function times tangent)                   | 
| [`Gradient`](@ref)                                   | gradient/Jacobian (as a vector)                         |
| [`SymmetricGradient`](@ref)                          | symmetric part of the gradient                          |
| [`Divergence`](@ref)                                 | divergence                                              |
| [`CurlScalar`](@ref)                                 | curl operator 1D to 2D (rotated gradient)               |
| [`Curl2D`](@ref)                                     | curl operator 2D to 1D                                  |
| [`Curl3D`](@ref)                                     | curl operator 3D to 3D                                  |
| [`Hessian`](@ref)                                    | Hesse matrix = all 2nd order derivatives (as a vector)  |
| [`Laplacian`](@ref)                                  | Laplace Operator                                        |
| [`ReconstructionIdentity{FEType}`](@ref)             | reconstruction operator into specified FEType           |
| [`ReconstructionDivergence{FEType}`](@ref)           | divergence of FEType reconstruction operator            |
| [`ReconstructionGradient{FEType}`](@ref)             | gradient of FEType reconstruction operator              |


!!! note

    Especially note the operators Reconstruction...{FEType} operators that allow to evaluate operators of some
    reconstructed version of a vector-valued testfunction that maps its discrete divergence to the divergence and so allows e.g. gradient-robust discretisations with classical non divergence-conforming ansatz spaces. So far such operators are available for the vector-valued Crouzeix-Raviart and Bernardi--Raugel finite element types.

!!! note

    As each finite element type is transformed differently from the reference domain to the general domain,
    the evaluation of each function operator has to be implemented for each finite element class.
    Currently, not every function operator works in any dimension and for any finite element. More evaluations
    are added as soon as they are needed (and possibly upon request).


## Jumps and Averages

If one of the operators above is evaluted ON_FACES for a finite element that is not continuous there, the code usual will crash or produce weird results. However, some operators can be transformed into a
Jump- or Average operator and then either the jumps or the average of this operator along the face is assembled. The operator Jump(Identity) for example gives the jump of the
identity evaluation on both sides of the face.

```@docs
Jump
Average
```

!!! note

    Currently this feature is only available for assembly on faces (2D and 3D) and certain function operators like Identity, Gradient, ReconstructionIdentity, ReconstructionGradient, NormalFlux, TangentFlux, but more
    are added as soon as they are needed (and possibly upon request).