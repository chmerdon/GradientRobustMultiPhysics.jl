# GradientRobustMultiPhysics.jl

This package offers finite element methods for multiphysics problems in Julia that focus on the preservation of structural and qualitative properties, in particular the gradient-robustness property for the discretisation of (nearly) incompressible flows and resulting qualitative properties in coupled processes. The code therefore offers several classical and novel non-standard finite element discretisations to play and compare with in these applications and a toolkit to setup multi-physics problems by defining PDE systems and fixed-point iterations to solve them.

The implementation is based on ExtendableGrids.jl that allows to have unstructured grids with mixed element geometries in it, e.g. triangles and quads in the same mesh. Generic quadrature rules of arbitrary order for intervals, triangles and parallelograms are also provided.

!!! note

    The focus is (at least currently) not on maxed-out efficiency or parallel computing. Also, this package is still in an early development stage with a limited number of applications and interfaces and features might change (especially for time-dependent PDEs) in future updates.


!!! note

    Currently, 3D functionality is very limited, but will be improved in future.


## What is gradient-robustness?

Gradient-robustness describes discretisations that exactly balance gradient forces and the momentum balance. In the case of the incompressible Navier--Stokes equations this means that the discrete velocity does not depend on the exact pressure. Divergence-conforming finite element metods have this property but are usually expensive and difficult to contruct. However, also non-divergence-conforming classical finite element methods can be made pressure-robust with the help of reconstruction operators applied to testfuntions in certain terms of the momentum balance.

Recently gradient-robustness was also shown to produce well-balanced schemes e.g. in the context of (nearly) compressible flows.

Todo: add some references


## Getting started

The general work-flow is as follows:

1. Describe your PDE with the help of the PDEDescription (possibly based on one of the [PDE Prototypes](@ref)). Additional parameters or non-constant parameters or non-standard terms (like stabilisations) in the weak form can be added manually afterwards.
2. Generate a mesh (possibly using one of the constructors by ExtendableGrid.jl) and assign boundary and right-hand side data matching the boundary regions and regions in the mesh.
3. Define finite element ansatz spaces for the unknowns of your PDE system.
4. Solve by using solve! or via a TimeControlSolver and advance! if the PDE system is time-dependent.

Please have a look at the Examples in the examples subfolder or the documented examples here, see [Examples Overview](@ref).
