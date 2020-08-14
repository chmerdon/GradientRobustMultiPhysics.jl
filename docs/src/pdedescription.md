
# PDE description

Although a more manually low-level assembly of your problem is possible, it is advised to describe it in the form of a PDEDescription
that has similarities with the weak form of your problem and in general does not need any information on the discretisation at this point.

The following flow chart summarises the assemble process that is run during the solve process. The green parts can be modified/specified by the user, the rest is handled automatically. For details on steering the solver see [PDE Solvers](@ref)

![Assembly Flowchart](assembly_flowchart.png) 


A PDEDescription is as a set of PDEOperators arranged in a quadratic n by n matrix. Every matrix row refers to one equation and the positioning of the PDEOperators (e.g. a bilinerform) immediately sets the information which unknowns have to be used to evaluate the operator. Also 
nonlinear PDEOperator are possible where extra information on the further involved uknowns have to be specified.
UserData is also assigned to the PDEDescription depending on their type. Operator coefficients are assigned directly to the PDEOperators (in form of AbstractActions), right-hand side data is assigned to the right-hand side array of PDEOperators and boundary data is assigned to the BoundaryOperators of the PDEDescription. Additionaly global constraints (like a global zero integral mean) can be assigned as a GlobalConstraint.

Several add...! funcions allow to extend the problems at any stage. There are several prototype PDEs documented on the [PDE Prototypes](@ref) page that can be used as a point of departure. Below is a list of functions that allows to generate, extend or inspect a PDEDEscription.


```@autodocs
Modules = [GradientRobustMultiPhysics]
Pages = ["pdedescription.jl"]
Order   = [:type, :function]
```

## PDE Operators

The PDE consists of PDEOperators characterising some feature of the model (like friction, convection, exterior forces etc.), they describe the continuous weak form of the PDE. The following table lists all available operators and physics-motivated constructors for them. Click on them to find out more details.


| PDEOperator subtype                 | Special constructors                 | Mathematically                                   |
| :---------------------------------: | :----------------------------------: | :----------------------------------------------: |
| [`AbstractBilinearForm`](@ref)      |                                      | ``(A(FO_1(u)),FO_2(v))``                         |
|                                     | [`LaplaceOperator`](@ref)            | ``(\kappa \nabla u,\nabla v)``                   |
|                                     | [`ReactionOperator`](@ref)           | ``(\alpha u, v)``                                |
|                                     | [`ConvectionOperator`](@ref)         | ``(\beta \cdot \nabla u, v)`` (beta is function) |
|                                     | [`HookStiffnessOperator2D`](@ref)    | ``(\mathbb{C} \epsilon(u),\epsilon(v))``         |
| [`AbstractTrilinearForm`](@ref)     |                                      | ``(A(FO_1(a),FO_2(u)),FO_3(v))``                 |
|                                     | [`ConvectionOperator`](@ref)         | ``(a \cdot \nabla u, v)`` (a is unknown)         |
| [`RhsOperator`](@ref)               |                                      | ``(f*FO(v))``                                    |

Legend: FO are placeholders for [Function Operators](@ref), and A stands for [Abstract Actions](@ref).

### Complete List and Details

```@autodocs
Modules = [GradientRobustMultiPhysics]
Pages = ["pdeoperators.jl"]
Order   = [:type, :function]
```

## Function Operators

FunctionOperators are building blocks for the weak form and define the operations that should be applied to the trial and test functions inside some PDEOperator. Below is a list of available FunctionOperators. 


| Function operator                          | Description                                   | 
| :----------------------------------------: | :-------------------------------------------: |
| [`Identity`](@ref)                         | identity operator                             |
| [`ReconstructionIdentity{FEType}`](@ref)   | reconstruction operator into specified FEType |
| [`NormalFlux`](@ref)                       | normal flux (function times normal)           |
| [`Gradient`](@ref)                         | gradient/Jacobian operator                    |
| [`SymmetricGradient`](@ref)                | symmetric part of the gradient                |
| [`Divergence`](@ref)                       | divergence operator                           |
| [`ReconstructionDivergence{FEType}`](@ref) | divergence of FEType reconstruction operator  |
| [`CurlScalar`](@ref)                       | curl operator 1D to 2D (rotated gradient)     |

!!! note

    Especially note the operators ReconstructionIdentity{FEType} and ReconstructionDivergence{FEType} that allow to evaluate some
    reconstructed version of a vector-valued testfunction that maps its discrete divergence to the divergence and so allows e.g. gradient-robust discretisations with classical non divergence-conforming ansatz spaces. So far such operators are available for the vector-valued Crouzeix-Raviart and Bernardi--Raugel finite element types.


```@autodocs
Modules = [GradientRobustMultiPhysics]
Pages = ["febasisevaluator.jl"]
Order   = [:type, :function]
```


## Global Constraints

GlobalConstraints are additional constraints that the user does not wish to implement as a global Lagrange multiplier because it e.g. causes a dense row in the system matrix and therefore may destroy the performance of the sparse matrix routines. Such a constraint may be a fixed integral mean. Another application are periodic boundary conditions or glued-together quantities in different regions of the grid. Here a CombineDofs constraint may help.

```@autodocs
Modules = [GradientRobustMultiPhysics]
Pages = ["globalconstraints.jl"]
Order   = [:type, :function]
```


## Boundary Data

BoundaryOperators carry the boundary data for each unknown. Each regions can have a different AbstractBoundaryType and an associated data function that satisfies the interface function data!(result,x) or function data!(result,x,t) if it is also time-dependent.

So far only DirichletBoundaryData is possible, as most other types can be implemented differently:
- NeumannBoundary can be implemented as a RhsOperator with on_boundary = true
- PeriodicBoundarycan be implemented as a CombineDofs <: AbstractGlobalConstraint
- SymmetryBoundary can be implemented as a RHS AbstractBilinearForm on BFaces and specified regions with operator NormalFlux + MultiplyScalarAction(penalty).

```@autodocs
Modules = [GradientRobustMultiPhysics]
Pages = ["boundarydata.jl"]
Order   = [:type, :function]
```