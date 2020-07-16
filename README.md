# juliaFE
FiniteElements for Julia
-------------------------

yet another finite element module for Julia


Features/Limitations:
- type-treed FiniteElements (scalar or vector-valued)
    - H1 elements (so far P1, P2, MINI, CR)
    - H1 elements with coefficients (so far BR)
    - L2 elements (so far P0, P1)
    - Hdiv elements (so far RT0)
    - Hcurl elements (in future)
- new finite elements (inside these categories) can be added quite easily (one mainly has to specify basis functions on reference geometries and DofMaps)    
- handling of mixed element geometries in the grid via ExtendableGrids
- PDEDescription module and problem prototypes for easy problem description and discretisations setup
- PDEDescription recognizes nonlinear problem and automatically devises a fixpoint algorithm
- time-dependent solvers (very preliminary stage, only backward Euler for now)
- generic quadrature rules (on intervals, triangles and quads)
- includes reconstruction operators for gradient-robust Stokes discretisation (so far BR>RT0 and CR>RT0)
- no strong focus on parallelisation, maxed-out efficiency at the moment (but possibly later)


EXAMPLES (in subfolgder examples, can be run by include("..."))
- minimal example: EXAMPLE_Minimal.jl
- convection-diffusion problem: EXAMPLE_ConvectionDiffusion.jl
- linear elasticity: EXAMPLE_CookMembrane.jl, EXAMPLE_ElasticTire2.jl
- incompressible flows: EXAMPLE_Stokes.jl, EXAMPLE_Stokes_probust.jl
- compressible flows; EAMPLE_CompressibleStokes.jl


Dependencies on other Julia packages:
- ExtendableSparse
- ExtendableGrids
- ForwardDiff
- BenchmarkTools
- PyPlot (to run most of the examples, but otherwise not necessary)
