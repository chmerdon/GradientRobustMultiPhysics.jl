[![Build status](https://github.com/chmerdon/GradientRobustMultiPhysics.jl/workflows/linux-macos-windows/badge.svg)](https://github.com/chmerdon/GradientRobustMultiPhysics.jl/actions)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://chmerdon.github.io/GradientRobustMultiPhysics.jl/stable/index.html)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://chmerdon.github.io/GradientRobustMultiPhysics.jl/dev/index.html)
[![DOI](https://zenodo.org/badge/229078096.svg)](https://zenodo.org/badge/latestdoi/229078096)


# GradientRobustMultiPhysics.jl

finite element module for Julia focussing on gradient-robust finite element methods and multiphysics applications, part of the meta-package [PDELIB.jl](https://github.com/WIAS-BERLIN/PDELib.jl)


### Features/Limitations:
- solves 1D, 2D and 3D problems in Cartesian coordinates
- uses type-treed FiniteElements (scalar or vector-valued)
    - H1 elements (so far P1, P2, P2B, MINI, CR, BR)
    - Hdiv elements (so far RT0, BDM1, RT1)
    - Hcurl elements (so far N0)
- finite elements can be broken (e.g. piecewise Hdiv) or live on faces or edges (experimental feature)
- grids by [ExtendableGrids.jl](https://github.com/j-fu/ExtendableGrids.jl), allows mixed element geometries in the grid (simplices and quads atm)
- PDEDescription module for easy and close-to-physics problem description and discretisation setup
- PDEDescription recognizes nonlinear operators and automatically devises fixed-point or Newton algorithms by automatic differentation (experimental feature)
- time-dependent solvers by own backward Euler implementation or via external module [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) (experimental)
- reconstruction operators for gradient-robust Stokes discretisations (BR>RT0/BDM1 in 2D/3D, or CR>RT0 in 2D, more in progress)
- plotting via [GridVisualize.jl](https://github.com/j-fu/GridVisualize.jl)
- export into csv or vtk datafiles possible for external plotting


### Installation
via Julia package manager in Julia 1.5 or above:

```@example
# latest stable version
(@v1.5) pkg> add GradientRobustMultiPhysics
# latest version
(@v1.5) pkg> add GradientRobustMultiPhysics#master
```

### EXAMPLES 
see [documentation](https://chmerdon.github.io/GradientRobustMultiPhysics.jl/stable/index.html)


### Dependencies on other Julia packages:

[ExtendableGrids.jl](https://github.com/j-fu/ExtendableGrids.jl)\
[GridVisualize.jl](https://github.com/j-fu/GridVisualize.jl)\
[ExtendableSparse.jl](https://github.com/j-fu/ExtendableSparse.jl)\
[DocStringExtensions.jl](https://github.com/JuliaDocs/DocStringExtensions.jl)\
[ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)\
[DiffResults.jl](https://github.com/JuliaDiff/DiffResults.jl)\
[StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl)
