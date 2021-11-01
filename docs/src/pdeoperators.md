# PDE Operators

## Purpose

The PDEDescription consists of PDEOperators characterising some feature of the model (like friction, convection, exterior forces, optimality conditions etc.) and describe the continuous weak form of the PDE.
They can be separated roughly into two categories: linear operators and nonlinear operators.


```@docs
GradientRobustMultiPhysics.PDEOperator
```

The following table lists all available operators and available physics-motivated constructors for them (besides the abstract main constructor).
Click on them or scroll down to find out more details.

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
| [`NonlinearForm`](@ref)     |                                          | ``(\mathrm{NA}(\mathrm{FO}_1(u),...,\mathrm{FO}_{N-1}(u)),\mathrm{FO}_N(v))``                                  |
| [`RhsOperator`](@ref)               |                                          | ``(f \cdot \mathrm{FO}(v))`` or ``\mathrm{A}(\mathrm{FO}(v))``                                                 |

Legend: ``\mathrm{FO}``  are placeholders for [Function Operators](@ref), and ``\mathrm{A}`` stands for a (linear) [Action](@ref) (that only expects the operator value of the finite element function as an input) and ``\mathrm{NA}`` stands for a (nonlinear) [Action](@ref) (see [`NonlinearForm`](@ref) for details).


## Assembly Type

Many PDE operators need a specification that decides on which set of entities of the mesh (e.g. cells, faces, bfaces, edges) a PDEOperator lives and has to be assembled. This can be steered by the AssemblyType of ExtendableGrids.
The AssemblyType can be also used as an argument for [Finite Element Interpolations](@ref). The following AssemblyTypes are available. 

| AssemblyType     | Description                                                      |
| :--------------- | :--------------------------------------------------------------- |
| AT_NODES         | interpolate at vertices of the mesh (only for H1-conforming FEM) |
| ON_CELLS         | assemble/interpolate on the cells of the mesh                  |
| ON_FACES         | assemble/interpolate on all faces of the mesh                  |
| ON_IFACES        | assemble/interpolate on the interior faces of the mesh         |
| ON_BFACES        | assemble/interpolate on the boundary faces of the mesh         |
| ON_EDGES (*)     | assemble/interpolate on all edges of the mesh (in 3D)          |
| ON_BEDGES (*)    | assemble/interpolate on the boundary edges of the mesh (in 3D) |

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

### Examples

Below some examples for operators are given:

```julia
# Example 1 : div-div bilinearform with a factor λ (e.g. for divergence-penalisation)
operator = AbstractBilinearForm([Divergence,Divergence]; factor = λ, name = "λ (div(u),div(v))")

# Example 2 : Gradient jump stabilisation with an item-dependent action and a factor s (e.g. for convection stabilisation)
xFaceVolumes::Array{Float64,1} = xgrid[FaceVolumes]
function stabilisation_kernel(result, input, item)
    result .= input 
    result .*= xFaceVolumes[item]^2
end
action = Action{Float64}(stabilisation_kernel, [2,2]; dependencies = "I", quadorder = 0 )
operator = AbstractBilinearForm([Jump(Gradient), Jump(Gradient)], action; AT = ON_IFACES, factor = s, name = "s |F|^2 [∇(u)]⋅[∇(v)]")

```

## Lagrange Multipliers

There is a special bilinearform intended to use for the assembly of Lagrange multipliers that automatically copies itself to the transposed block of the PDEdescription.

```@docs
LagrangeMultiplier
```


## Nonlinear Operators

Nonlinear Operators can be setup in two ways. The manual way requires the user to define an action with a nonlinear action kernel (see [Action Kernel](@ref)) that specifies the linearisation of the nonlinearity. There is also an automatic way where the user specifies only a function (where the input can be used nonlinearly) which is then automatically differentiated to generate the linearised action kernel, see below for details.

```@docs
NonlinearForm
```


## Other Operators

There are some more operators that do not fit into the structures above. Also, in the future, the goal is to open up the operator level for exterior code to setup operators that are assembled elsewhere.

```@docs
FVConvectionDiffusionOperator
DiagonalOperator
CopyOperator
```

