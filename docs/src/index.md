# GradientRobustMultiPhysics.jl

This package offers finite element methods for multiphysics problems in Julia that focus on the preservation of structural and qualitative properties, in particular the gradient-robustness property for the discretisation of constrained vector-valued quantities (like nearly incompressible flows) and coupled processes. The code therefore offers several classical and novel non-standard finite element discretisations to play and compare with in these applications and a toolkit to setup multi-physics problems by defining PDE systems and fixed-point iterations to solve them.

The implementation is based on ExtendableGrids.jl that allows to have unstructured grids with mixed element geometries in it, e.g. triangles and quads in the same mesh. Generic quadrature rules of arbitrary order for intervals, triangles and parallelograms are also provided.

!!! note

    The focus is (at least currently) not on maxed-out efficiency or parallel GPU computing. Also, this package is still in an early development stage with a limited number of applications and interfaces and features might change in future updates.


## Getting started

The general work-flow is as follows:

1. Describe your PDE with the help of the PDEDEscription (possibly based on one of the prototypes). Additional parameters or non-constant parameters or non-standard terms (like stabilisations) in the weak form can be added manually afterwards.
2. Generate a mesh (possibly using one of the constructors by ExtendableGrid.jl) and assign boundary and right-hand side data matching the boundary regions and regions in the mesh.
3. Define finite element ansatz spaces for the unknowns of your PDE system.
4. Solve by using solve! or via a TimeControlSolver and advance! if the PDE system is time-dependent.

Please have a look at the Examples in the examples subfolder. The smallest one is EXAMPLE_Minimal.jl.