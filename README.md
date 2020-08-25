# GradientRobustMultiPhysics.jl

finite element module for Julia focussing on gradient-robust finite element methods and multiphysics applications


### Features/Limitations:
- solve 1D, 2D and 3D problems in Cartesian coordinates (3D features very limited at the moment)
- type-treed FiniteElements (scalar or vector-valued)
    - H1 elements (so far P1, P2, MINI, CR, BR)
    - L2 elements (so far P0, P1)
    - Hdiv elements (so far RT0, BDM1, RT1)
    - Hcurl elements (in future)
- based on ExtendableGrids.jl, allowing mixed element geometries in the grid
- PDEDescription module and problem prototypes for easy problem description and discretisation setup
- PDEDescription recognizes nonlinear problem and automatically devises a fixpoint algorithm
- time-dependent solvers (very preliminary stage, only backward Euler for now)
- generic quadrature rules (on intervals, triangles, parallelograms, parallelepiped)
- includes reconstruction operators for gradient-robust Stokes discretisation (BR>RT0 or CR>RT0 in 2D and 3D, more in progress)
- export into vtk datafiles for external plotting


### EXAMPLES 
see subfolder examples, can be run by include("...")


### Dependencies on other Julia packages:
- ExtendableSparse.jl
- ExtendableGrids.jl
- ForwardDiff.jl
- DocStringExtensions.jl
- BenchmarkTools.jl
- Triangulate.jl (for runtests.jl and grid generation in some examples)
- PyPlot.jl (to run most of the examples with plots)
