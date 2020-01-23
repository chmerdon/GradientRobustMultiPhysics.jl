# juliaFE
FiniteElements for Julia
=========================

Learning julia by implementing finite elements

Usage examples:
- 1D Poisson problem: see Example_Poisson_Line.jl
- 2D Poisson problem: see Example_Poisson_Lshape.jl
- 2D Stokes problem: see Example_Stokes_Square.jl


Dependencies on other Julia packages:
- ExtendableSparse
- Triangulate
- DiffResults
- FowardDiff


Features/Limitations:
- type-treed FiniteElements module into
    H1 elements (so far P1,P2,MINI,BR,CR)
    L2 elements (so far P0, provisorically masked as a H1 element)
    Hdiv elements (so far RT0 (but needs more testing and features))
    Hcurl elements (in future)
- running solver for Poisson problems, L2 bestapproximation with Dirichlet boundary data, Stokes problem
- own Mesh class (so far only for 1D and 2D meshes into intervals and triangles)
- own Quadrature class (with generic quadrature formulas for intervals and triangles)


Next Goals:
- improve assembly and features for Hdiv FiniteElement (esp. boundary and face data handling)
- type-treed geometry elements (Point, Line, Triangle, Tetrahedron,...) linked
to objects nodes4cells of Mesh to choose correct transformation by multiple dispatch
- sparse-matrices to save nodes4cells etc. to allow (in a far future) for different elements in Mesh and easier
adjacency information (to build them up in 3D)


