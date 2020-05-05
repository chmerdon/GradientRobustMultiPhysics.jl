# juliaFE
FiniteElements for Julia
-------------------------

Learning julia by implementing finite elements

EXAMPLE scripts
EXAMPLE_MixedElementGeometries

Dependencies on other Julia packages:
- ExtendableSparse
- ExtendableGrids
- VTKView (optional)
- ForwardDiff
- BenchmarkTools

Features/Limitations:
- type-treed FiniteElements
    H1 elements (so far P1)
    L2 elements (in future)
    Hdiv elements (in future)
    Hcurl elements (in future)
- handling of mixed element geometries via ExtendableGrids and flexible operator assembly
- generic quadrature rules
