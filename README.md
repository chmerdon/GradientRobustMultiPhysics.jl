# juliaFE
FiniteElements for Julia
=========================

Learning julia by implementing finite elements

Usage examples:
- 1D Poisson problem: see Example_Poisson_Line.jl
- 2D Poisson problem: see Example_Poisson_Lshape.jl
- 2D Stokes problem: see Example_Stokes_Square.jl, Example_Stokes_Square_probust.jl
- 2D Navier-Stokes problem: see Example_NavierStokes_Square.jl 


Dependencies on other Julia packages:
- ExtendableSparse
- Triangulate
- DiffResults
- FowardDiff


Features/Limitations:
- type-treed FiniteElements module into
    H1 elements (so far P1,P2,MINI,BR,CR)
    L2 elements (so far P0, provisorically masked as a H1 element)
    Hdiv elements (so far RT0, RT1)
    Hcurl elements (in future)
- running solver for Poisson problems, L2 bestapproximation with Dirichlet boundary data, (Navier-)Stokes problem
- pressure-robustness: Hdiv reconstruction for Stokes elements (so far BR/RT0)
- own Mesh class (so far only for 1D and 2D meshes into intervals and triangles)
- own Quadrature class (with generic quadrature formulas for intervals and triangles)


Next Goals:
- implement BDM1 and P2B-Stokes FEM (with and without RT1/BDM reconstruction)
- implement solver for compressible Stokes problem
- further improve steering by Grid.ElemTypes (Point, Line, Triangle, Tetrahedron,...)
to objects nodes4cells of Mesh to choose correct transformation by multiple dispatch
- sparse-matrices to save nodes4cells etc. to allow (in a far future) for different elements in Mesh and easier
adjacency information (to build them up in 3D)
- different boundary conditions (dependent on regions in the cell)


Known issues:
- Stokes solver slow for large number of dofs due to issue with ExtendableSparse that suffers from a non-sparse row/column in the matrix related to the pressure constraint


