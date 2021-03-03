
# Assembly Details

The assembly of an operator is essentially based on a combination of [Assembly Types](@ref) and [Assembly Patterns](@ref). The assembly type mainly allows to choose the geometry information needed for providing quadrature and dof handling. The assembly pattern then basically evaluates the
function operators and action for the ansatz and test functions and does the quadrature-weighted accumulation into matrices or vectors that represent the operators.


## Assembly Types

The following assembly types are available.

| AssemblyType     | Description                                           |
| :--------------- | :---------------------------------------------------- |
| ON_CELLS         | assembles over the cells of the mesh                  |
| ON_FACES         | assembles over all faces of the mesh                  |
| ON_IFACES        | assembles over the interior faces of the mesh         |
| ON_BFACES        | assembles over the boundary faces of the mesh         |
| ON_EDGES (*)     | assembles over all edges of the mesh (in 3D)          |
| ON_BEDGES (*)    | assembles over the boundary edges of the mesh (in 3D) |

!!! note
    (*) = only reasonable in 3D and still experimental, might have some issues


## Assembly Patterns

Each Pattern comes with a number of arguments/quantities with associated [Function Operators](@ref) as well as one of the [Assembly Types](@ref) that states whether the form is evaluated over CELLS, FACES order BFACES (see above). Important note: this assembly type is relative to the grid of the first argument of the pattern. If this argument already lives ON_FACES and the pattern is also ON_FACES, it will ultimatively assemble on the faces of the faces (that are the edges of the grid with these faces). Moreover, patterns can have [Abstract Actions](@ref) that allow to make the evaluations parameter-, region- and/or function-dependent. Each pattern then has usually on to three implementation that writes into FEMatrix or FEVector (where e.g. a subset of arguments is fixed) or evaluates the pattern in the given FEVectorBlocks.

The patterns are used by the assembly of the PDE operators defined in a [PDE Description](@ref). However, it is also possible for the user to use them directly, see e.g. the example [Commuting Interpolators (2D)](@ref).

The following table lists all available assembly patterns and how they can be used for assembly or evaluations.


| AssemblyPattern    | evaluate | assemble into matrix | assembled into vector |
| :----------------: | :------: | :------------------: | :-------------------: |
| ItemIntegrator     |    yes   |          no          |           no          |
| LinearForm         |     no   |          no          |          yes          |
| BilinearForm       |     no   |         yes          |          yes (1)      |
| TrilinearForm      |     no   |         yes (1)      |          yes (2)      |
| MultilinearForm    |     no   |          no          |          yes (N-1)    |
| NonlinearForm      |     no   |         yes (L)      |          yes (L)      |

Number in brackets denotes the number of fixed arguments needed for this assembly, (L) means that a current solution is needed to evaluate (to evaluate the linearisation of the nonlinear form in this state).
Evaluations of the other AssemblyPatterns will be possible in a future update, but currently have to be performed by maintaining a duplicate of the pattern rewritten as an ItemIntegrator.


#### ItemIntegrator

```@docs
ItemIntegrator
```


#### Linearform

```@docs
LinearForm
```


#### Bilinearform

```@docs
BilinearForm
SymmetricBilinearForm
```

#### Trilinearform

```@autodocs
TrilinearForm
```

#### Multilinearform

```@docs
MultilinearForm
```

#### Nonlinearform

```@docs
NonlinearForm
```


### Evaluate! & Assemble!

```@docs
evaluate!
assemble!
```