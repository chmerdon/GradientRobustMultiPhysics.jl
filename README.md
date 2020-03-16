# juliaFE
FiniteElements for Julia
-------------------------

Learning julia by implementing finite elements

DEMO scripts:
- DEMO_Poisson1D.jl
- DEMO_Poisson2D.jl
- DEMO_HagenPoiseuille.jl
- DEMO_HdivBA_polynomials.jl
- DEMO_Stokes_p7vortex.jl
- DEMO_Stokes_p7vortex_probustness.jl
- DEMO_Stokes_polynomials.jl
- DEMO_GreshoVortex.jl
- DEMO_CompressibleStokes_stratifiednoflow.jl
- DEMO_CompressibleStokes_p7vortex.jl


Dependencies on other Julia packages:
- ExtendableSparse
- Triangulate
- DiffResults
- ForwardDiff
- BenchmarkTools


Features/Limitations:
- type-treed FiniteElements module into
    H1 elements (so far P1,P2,P2B,MINI,BR,CR)
    L2 elements (so far P0, P1disc, provisorically masked as a H1 element)
    Hdiv elements (so far RT0, RT1, BDM1)
    Hcurl elements (in future)
- running solver for Poisson problems, L2 bestapproximation with Dirichlet boundary data, (Navier-)Stokes problem, compressible Stokes problem (needs further testing)
- pressure-robustness: Hdiv reconstruction for Stokes elements (so far BR/RT0, BR/BDM1, CR/RT0, CR/BDM1)
- own Mesh class (so far only for 1D and 2D meshes into intervals and triangles)
- own Quadrature class (with generic quadrature formulas for intervals and triangles)


Next Goals:
- multiple solves with same matrix (as in transient and compressible Stokes solver) should reuse LU decomposition of sparse matrix
- further test solver for compressible Stokes problem and extend to transient Navier-Stokes
- fix interpolation for MINI and P2B element (cell bubble values depend on nodal values)
- implement tests for Hdiv elements and reconstruction operators
- implement tests for compressible Stokes problem
- implement RT1/BDM2 reconstruction for P2B-Stokes FEM
- add Pardiso solver (via Pardiso.jl)
- further improve steering by Grid.ElemTypes (Point, Line, Triangle, Tetrahedron,...)
to objects nodes4cells of Mesh to choose correct transformation by multiple dispatch
- sparse-matrices to save nodes4cells etc. to allow (in a far future) for different elements in Mesh and easier adjacency information (to build them up in 3D)


Known Issues:
- Navier-Stokes solve with CR element for a linear problem is not exact (Stokes solve is exact)



