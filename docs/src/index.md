# PACKAGE-NAME

This package focusses on finite element methods in Julia that focus on the preservation of structural and qualitative properties, in particular the gradient-robustness property for discretisation of the (incompressible or compressible) Navier--Stokes equations and mass-conservation for densities that are transported in the flow. The code therefore offers several classical and novel non-standard finite element discretisations to play with in these applications.
However, the code also allows to solve other PDE systems and is extendable to other finite element methods, see the examples to get some impression.

Moreover, the implementation is based on the ExtendableGrids package that allows to have unstructured grids with mixed element geometries in it, e.g. triangles and quads in the same mesh. Generic quadrature rules of arbitrary order for intervals, triangles and parallelograms are also provided.

Note, that the focus is (at least currently) not on maxed-out efficiency or parallel computing. Also, this package is still in an early development stage and interfaces and features might change.