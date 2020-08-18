
# PDE Description

## Purpose

Although a more manually low-level assembly of your problem is possible, it is advised to describe it in the form of a PDEDescription
to get access to certain automated mechanisms (in particular concerning solvers). The PDEDescription has similarities with the weak form of your problem (without time derivatives that are added separately) and in general does not need any information on the discretisation at this point.

The following flow chart summarises the assemble process that is run during the solve process. The green parts can be modified/specified by the user, the rest is handled automatically. For details on steering the solver see [PDE Solvers](@ref)

![Assembly Flowchart](assembly_flowchart.png) 


```@docs
PDEDescription
Base.show(io::IO, PDE::PDEDescription)
```


## Creating/Extending a PDEDescription

Several add...! functions allow to extend the problems at any stage. There are several prototype PDEs documented on the [PDE Prototypes](@ref) page that can be used as a point of departure. Below is a list of functions that allows to initialise and extend a PDEDescription.

```@autodocs
Modules = [GradientRobustMultiPhysics]
Pages = ["pdedescription.jl"]
Order   = [:type, :function]
```

## PDE Operators

The PDE consists of PDEOperators characterising some feature of the model (like friction, convection, exterior forces etc.), they describe the continuous weak form of the PDE. The following table lists all available operators and physics-motivated constructors for them. Click on them to find out more details.


| PDEOperator subtype                 | Special constructors                 | Mathematically                                                         |
| :---------------------------------- | :----------------------------------- | :--------------------------------------------------------------------- |
| [`AbstractBilinearForm`](@ref)      |                                      | ``(\mathrm{A}(\mathrm{FO}_1(u)),\mathrm{FO}_2(v))``                    |
|                                     | [`LaplaceOperator`](@ref)            | ``(\kappa \nabla u,\nabla v)``                                         |
|                                     | [`ReactionOperator`](@ref)           | ``(\alpha u, v)``                                                      |
|                                     | [`ConvectionOperator`](@ref)         | ``(\beta \cdot \nabla u, v)`` (beta is function)                       |
|                                     | [`HookStiffnessOperator2D`](@ref)    | ``(\mathbb{C} \epsilon(u),\epsilon(v))``                               |
| [`AbstractTrilinearForm`](@ref)     |                                      | ``(\mathrm{A}(\mathrm{FO}_1(a),\mathrm{FO}_2(u)),\mathrm{FO}_3(v))``   |
|                                     | [`ConvectionOperator`](@ref)         | ``(a \cdot \nabla u, v)`` (a is registered unknown)                    |
| [`RhsOperator`](@ref)               |                                      | ``(f \cdot \mathrm{FO}(v))``                                           |

Legend: ``\mathrm{FO}``  are placeholders for [Function Operators](@ref), and ``\mathrm{A}`` stands for [Abstract Actions](@ref).


```@autodocs
Modules = [GradientRobustMultiPhysics]
Pages = ["pdeoperators.jl"]
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