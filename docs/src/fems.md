
# Implemented Finite Elements

This page describes the finite element type-tree and lists all implemented finite elements.



## The Finite Element Type-Tree

Finite elements are abstract type leaves in a type-tree. Its root and first layer look like this

AbstractFiniteElements
- AbstractH1FiniteElement
- AbstractHdivFiniteElement
- AbstractHcurlFiniteElement
- AbstractL2FiniteElement


Remarks:
- each finite elements mainly comes with a set of basis functions in reference coordinates for each applicable AbstractElementGeometry and a generator for the degrees of freedom map of a [`FESpace`](@ref)
- the type steers how the basis functions are transformed from local to global coordinates and how FunctionOperators are evaluated by FEBasisEvaluator.jl
- depending on additional continuity properties of the element types more basis function sets are defined:
    - AbstractH1FiniteElements additionally have evaluations of nonzero basisfunctions on faces/bfaces
    - AbstractHdivFiniteElements additionally have evaluations of nonzero normalfluxes of basisfunctions on faces/bfaces


## List of implemented Finite Elements

The following table lists all curently implemented finite elements. Click on them to find out more details.


| H1 finite elements | Hdiv finite elements | Hcurl finite elements | L2 finite elements |
| :----------------: | :------------------: | :-------------------: | :----------------: |
| [`H1P1`](@ref)     | [`HDIVRT0`](@ref)    | [`HCURLN0`](@ref)     | [`L2P0`](@ref)     |
| [`H1MINI`](@ref)   | [`HDIVBDM1`](@ref)   |                       | [`L2P1`](@ref)     |
| [`H1P2`](@ref)     | [`HDIVRT1`](@ref)    |                       |                    |
| [`H1BR`](@ref)     |                      |                       |                    |
| [`H1CR`](@ref)     |                      |                       |                    |



## Details

#### H1-conforming finite elements
```@docs
H1P1
H1MINI
H1P2
H1CR
H1BR
```

#### Hdiv-conforming finite elements

```@docs
HDIVRT0
HDIVBDM1
HDIVRT1
```

#### Hcurl-conforming finite elements

```@docs
HCURLN0
```

#### L2-conforming finite elements

```@docs
L2P0
L2P1
```
