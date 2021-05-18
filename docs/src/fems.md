
# Implemented Finite Elements

This page describes the finite element type-tree and lists all implemented finite elements.



## The Finite Element Type-Tree

Finite elements are abstract type leaves in a type-tree. The complete tree looks like this:

```
AbstractFiniteElement
├─ AbstractH1FiniteElement
│  ├─ AbstractH1FiniteElementWithCoefficients
│  │  └─ H1BR
│  ├─ H1CR
│  ├─ H1MINI
│  ├─ H1P0
│  ├─ H1P1
│  ├─ H1P2
│  └─ H1P2B
│  └─ H1P3
├─ AbstractHcurlFiniteElement
│  └─ HCURLN0
└─ AbstractHdivFiniteElement
   ├─ HDIVBDM1
   ├─ HDIVRT0
   └─ HDIVRT1
```


#### Remarks
- each type depends on one or two parameters, the first one is always the number of components (ncomponents) that determines if the
  finite element is scalar- or veector-valued; some elements additionaly require the parameter edim <: Int if they are structurally different in different space dimensions
- each finite elements mainly comes with a set of basis functions in reference coordinates for each applicable AbstractElementGeometry and degrees of freedom maps for each [Assembly Type](@ref) (coded as a string)
- broken finite elements are possible via the broken switch in the [FESpace](@ref) constructor
- the type steers how the basis functions are transformed from local to global coordinates and how FunctionOperators are evaluated
- depending on additional continuity properties of the element types more basis function sets are defined:
    - AbstractH1FiniteElements additionally have evaluations of nonzero basisfunctions on faces/bfaces
    - AbstractHdivFiniteElements additionally have evaluations of nonzero normalfluxes of basisfunctions on faces/bfaces
    - AbstractHcurlFiniteElements additionally have evaluations of nonzero tangentfluxes of basisfunctions on edges/bedges
- each finite element has its own implemented standard interpolation interpolate! (see [Finite Element Interpolations](@ref)) that can be applied to a [Data Function](@ref), below it is shortly described what this means for each finite element


## List of implemented Finite Elements

The following table lists all curently implemented finite elements. Click on them to find out more details.


| H1 finite elements | Hdiv finite elements | Hcurl finite elements |
| :----------------: | :------------------: | :-------------------: |
| [`H1P0`](@ref)     | [`HDIVRT0`](@ref)    | [`HCURLN0`](@ref)     |
| [`H1P1`](@ref)     | [`HDIVBDM1`](@ref)   |                       |
| [`H1MINI`](@ref)   | [`HDIVRT1`](@ref)    |                       |
| [`H1CR`](@ref)     |                      |                       |
| [`H1BR`](@ref)     |                      |                       |
| [`H1P2`](@ref)     |                      |                       |
| [`H1P2B`](@ref)    |                      |                       |
| [`H1P3`](@ref)     |                      |                       |



## H1-conforming finite elements

### P0 finite element

Piecewise constant finite element that has one degree of freedom on each cell of the grid. (It is masked as a H1-conforming finite element, because it uses the same transformations.)

The interpolation of a given function into this space preserves the cell integrals.

```@docs
H1P0
```

### P1 finite element

The lowest-order current finite element that has a degree of freedom on each vertex of the grid. On simplices the
basis functions coincide with the linear barycentric coordinates, on parallelepiped bi-linear functions are used
(also known as Q1 element).

The interpolation of a given function into this space performs point evaluations at the nodes.

```@docs
H1P1
```


### MINI finite element

The mini finite element adds cell bubles to the P1 element that are e.g. beneficial to define inf-sup stable finite element pairs for the Stokes problem.

The interpolation of a given function into this space performs point evaluations at the nodes and preserves its
cell integral.

```@docs
H1MINI
```

### Bernardi-Raugel (BR) finite element

The Bernardi-Raugel adds normal-weighted face bubbles to the P1 finite element and therefore is only available
as a vector-valued element.

The interpolation of a given function into this space performs point evaluations at the nodes and preserves face integrals of its normal flux.

```@docs
H1BR
```


### P2 finite element

The P2 finite element method on simplices equals quadratic polynomials. On the Triangle2D shape the degrees of freedom
are associated with the three vertices and the three faces of the triangle. On the Tetrahedron3D shape the degrees of freedom are associated with the four verties and the six edges. On Parallelogram2D cubic Q2 element functions are used.

The interpolation of a given function into this space performs point evaluations at the nodes and preserves its face/edge integrals in 2D/3D.

```@docs
H1P2
```

### P2B finite element

The P2B finite element adds additional cell bubles (in 2D and 3D) and face bubbles (only in 3D) that are e.g. used to define inf-sup stable finite element pairs for the Stokes problem.

The interpolation of a given function into this space performs point evaluations at the nodes and preserves its cell and face integrals in 2D and also edge integrals in 3D.

```@docs
H1P2B
```

### P3 finite element

The P3 finite element method on simplices equals cubic polynomials. On the Triangle2D shape the degrees of freedom
are associated with the three vertices, the three faces (double dof) of the triangle and the cell itself (one cell bubble).

The interpolation of a given function into this space performs point evaluations at the nodes and preserves cell and face integrals in 2D.

```@docs
H1P3
```

### Crouzeix-Raviart (CR) finite element

The Crouzeix-Raviart element associates one lowest-order function with each face. On the Triangle2D shape, the basis function of a face is one minus two times the nodal basis function of the opposite node. 

The interpolation of a given function into this space preserves its face integrals.

```@docs
H1CR
```



## Hdiv-conforming finite elements

These Raviart-Thomas and Brezzi-Douglas-Marini finite elements of lower order and their standard interpolations are available:

```@docs
HDIVRT0
HDIVBDM1
HDIVRT1
```

## Hcurl-conforming finite elements

So far only the lowest order Nedelec element is available in 2D and 3D. On Triangle2D it has one degree of freedom for each face (i.e. the rotated RT0 element), on Tetrahedron3D it has one degree of freedom associated to each of the six edges.

Its standard interpolation of a given functions preserves its tangential face/edge integrals.

```@docs
HCURLN0
```

