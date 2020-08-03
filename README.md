# GradientRobustMultiPhysics.jl

finite element module for Julia focussing on gradient-robust finite element methods and multiphysics applications


### Features/Limitations:
- solve 1D, 2D and 3D problems in Cartesian coordinates (3D features very limited, though at the moment)
- type-treed FiniteElements (scalar or vector-valued)
    - H1 elements (so far P1, P2, MINI, CR)
    - H1 elements with coefficients (so far BR)
    - L2 elements (so far P0, P1)
    - Hdiv elements (so far RT0)
    - Hcurl elements (in future)
- based on ExtendableGrids.jl, allowing mixed element geometries in the grid
- PDEDescription module and problem prototypes for easy problem description and discretisations setup
- PDEDescription recognizes nonlinear problem and automatically devises a fixpoint algorithm
- time-dependent solvers (very preliminary stage, only backward Euler for now)
- generic quadrature rules (on intervals, triangles, parallelograms, parallelepiped)
- includes reconstruction operators for gradient-robust Stokes discretisation (so far only 2D BR>RT0 and CR>RT0)


### EXAMPLES (in subfolder examples, can be run by include("..."))
- minimal examples: EXAMPLE_Minimal.jl, EXAMPLE_1D.jl, EXAMPLE_3D.jl
- convection-diffusion problem: EXAMPLE_ConvectionDiffusion.jl
- linear elasticity: EXAMPLE_CookMembrane.jl, EXAMPLE_ElasticTire2.jl
- incompressible flows: EXAMPLE_Stokes.jl, EXAMPLE_Stokes_probust.jl, EXAMPLE_Karman.jl
- instationary flows: EXAMPLE_KarmanStreet_instationary.jl
- compressible flows; EAMPLE_CompressibleStokes.jl


### Dependencies on other Julia packages:
- ExtendableSparse.jl
- ExtendableGrids.jl
- ForwardDiff.jl
- DocStringExtensions.jl
- BenchmarkTools.jl
- Triangulate.jl (for runtests.jl and grid generation in some examples)
- PyPlot.jl (to run most of the examples with plots)
