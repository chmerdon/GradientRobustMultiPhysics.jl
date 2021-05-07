# PDE Operators

## Purpose

The PDE consists of PDEOperators characterising some feature of the model (like friction, convection, exterior forces etc.), they describe the continuous weak form of the PDE. 


```@docs
GradientRobustMultiPhysics.PDEOperator
```

The following table lists all available operators and physics-motivated constructors for them. Click on them or scroll down to find out more details.

| Main constructors                   | Special constructors                     | Mathematically                                                                                                 |
| :---------------------------------- | :--------------------------------------- | :------------------------------------------------------------------------------------------------------------- |
| [`AbstractBilinearForm`](@ref)      |                                          | ``(\mathrm{A}(\mathrm{FO}_1(u)),\mathrm{FO}_2(v))`` or ``(\mathrm{FO}_1(u),\mathrm{A}(\mathrm{FO}_2(v)))``     |
|                                     | [`LaplaceOperator`](@ref)                | ``(\kappa \nabla u,\nabla v)``                                                                                 |
|                                     | [`ReactionOperator`](@ref)               | ``(\alpha u, v)``                                                                                              |
|                                     | [`LagrangeMultiplier`](@ref)             | ``(\mathrm{FO}_1(u), v)`` (automatically assembles 2nd transposed block)                                       |
|                                     | [`ConvectionOperator`](@ref)             | ``(\beta \cdot \nabla u, v)`` (beta is function)                                                               |
|                                     | [`HookStiffnessOperator2D`](@ref)        | ``(\mathbb{C} \epsilon(u),\epsilon(v))`` (also 1D or 3D variants exist)                                        |
| [`AbstractTrilinearForm`](@ref)     |                                          | ``(\mathrm{A}(\mathrm{FO}_1(a),\mathrm{FO}_2(u)),\mathrm{FO}_3(v))``                                           |
|                                     | [`ConvectionOperator`](@ref)             | ``((a \cdot \nabla) u, v)`` (a is registered unknown)                                                          |
|                                     | [`ConvectionRotationFormOperator`](@ref) | ``((a \times \nabla) u,v)`` (a is registered unknown, only 2D for now)                                         |
| [`GenerateNonlinearForm`](@ref)     |                                          | ``(\mathrm{NA}(\mathrm{FO}_1(u),...,\mathrm{FO}_{N-1}(u)),\mathrm{FO}_N(v))``                                  |
| [`RhsOperator`](@ref)               |                                          | ``(f \cdot \mathrm{FO}(v))``                                                                                   |

Legend: ``\mathrm{FO}``  are placeholders for [Function Operators](@ref), and ``\mathrm{A}`` stands for a (linear) [Actions](@ref) (that only expects the operator value of the finite element function as an input) and ``\mathrm{NA}`` stands for a (nonlinear) [Actions](@ref) (see [`GenerateNonlinearForm`](@ref) for details).


## Assembly Type

Many PDE operators need a specification of that decides to which parts of the mesh the PDEOperator is associated (e.g. cells, faces, bfaces, edges), this is prescribed via the AssemblyType.
The following assembly types are available. Additional to define where PDEOperators live and assemble, they can be also used as an argument for interpolation!.

| AssemblyType     | Description                                                      |
| :--------------- | :--------------------------------------------------------------- |
| AT_NODES         | interpolate at vertices of the mesh (only for H1-conforming FEM) |
| ON_CELLS         | assemble/interpolate over the cells of the mesh                  |
| ON_FACES         | assemble/interpolate over all faces of the mesh                  |
| ON_IFACES        | assemble/interpolate over the interior faces of the mesh         |
| ON_BFACES        | assemble/interpolate over the boundary faces of the mesh         |
| ON_EDGES (*)     | assemble/interpolate over all edges of the mesh (in 3D)          |
| ON_BEDGES (*)    | assemble/interpolate over the boundary edges of the mesh (in 3D) |

!!! note
    (*) = only reasonable in 3D and still experimental, might have some issues


```@autodocs
Modules = [GradientRobustMultiPhysics]
Pages = ["assemblytypes.jl"]
Order   = [:type, :function]
```


## Special Linear Operators

Below you find the special constructors of available common linear, bilinear and trilinear forms.

```@docs
LaplaceOperator
ReactionOperator
ConvectionOperator
ConvectionRotationFormOperator
HookStiffnessOperator1D
HookStiffnessOperator2D
HookStiffnessOperator3D
RhsOperator
```

## Custom Linear Operators

It is possible to define custom bilineraforms and trilinearforms by specifiyng [Function Operators](@ref) and (in case of bilinearform optionally) an [Action](@ref).

```@docs
AbstractBilinearForm
AbstractTrilinearForm
```

## Lagrange Multipliers

There is a special bilinearform intended to use for the assembly of Lagrange multipliers that automatically copies itself to the transposed block of the PDEdescription.

```@docs
LagrangeMultiplier
```


## Nonlinear Operators

Nonlinear Operators can be setup in two ways. The manual way requires the user to define an action with a nonlinear action kernel (see [Action Kernels](@ref)) that specifies the linearisation of the nonlinearity. The is also an automatic way where the user specifies a norml action kernel (where the input can be used nonlinearly) which is then automatically differentiated to generate the linearised action kernel, see below for details.

```@docs
GenerateNonlinearForm
```


## Other Operators

There are some more operators that do not fit into the structures above. Also, in the future, the goal is to open up the operator level for exterior code to setup operators that are assembled elsewhere.

```@docs
FVConvectionDiffusionOperator
DiagonalOperator
CopyOperator
```

