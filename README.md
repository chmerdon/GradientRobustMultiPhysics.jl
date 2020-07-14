# juliaFE
FiniteElements for Julia
-------------------------

Learning julia by implementing finite elements

EXAMPLE scripts
- Minimal: EXAMPLE_Minimal.jl
- Convection-Diffusion: EXAMPLE_ConvectionDiffusion.jl
- Elasticity: EXAMPLE_CookMembrane.jl, EXAMPLE_ElasticTire_.jl
- incompressible flows: EXAMPLE_Stokes.jl, EXAMPLE_Stokes_probust.jl
- compressible flows; EAMPLE_CompressibleStokes.jl

Dependencies on other Julia packages:
- ExtendableSparse
- ExtendableGrids
- VTKView (optional)
- ForwardDiff
- BenchmarkTools

Features/Limitations:
- type-treed FiniteElements
    H1 elements (so far P1, P2, MINI, CR)
    H1 elements with coefficients (so far BR)
    L2 elements (so far P0, P1)
    Hdiv elements (so far RT0)
    Hcurl elements (in future)
- PDEDescription module and problem prototypes for easy problem description and discretisations setup
- PDEDescription recognizes nonlinear problem and automatically devises a fixpoint algorihtm
- handling of mixed element geometries via ExtendableGrids and flexible operator assembly
- reconstruction operators for gradient-robust Stokes discretisation (so far BR>RT0 and CR>RT0)
- generic quadrature rules
