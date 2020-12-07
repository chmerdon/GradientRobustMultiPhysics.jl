[![Build status](https://github.com/chmerdon/GradientRobustMultiPhysics.jl/workflows/linux-macos-windows/badge.svg)](https://github.com/chmerdon/GradientRobustMultiPhysics.jl/actions)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://chmerdon.github.io/GradientRobustMultiPhysics.jl/stable/index.html)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://chmerdon.github.io/GradientRobustMultiPhysics.jl/dev/index.html)
[![DOI](https://zenodo.org/badge/229078096.svg)](https://zenodo.org/badge/latestdoi/229078096)


# GradientRobustMultiPhysics.jl

finite element module for Julia focussing on gradient-robust finite element methods and multiphysics applications


### Features/Limitations:
- solve 1D, 2D and 3D problems in Cartesian coordinates
- type-treed FiniteElements (scalar or vector-valued)
    - H1 elements (so far P1, P2, P2B, MINI, CR, BR)
    - Hdiv elements (so far RT0, BDM1, RT1)
    - L2 elements (so far P0, P1)
    - Hcurl elements (so far N0)
- based on [ExtendableGrids.jl](https://github.com/j-fu/ExtendableGrids.jl), allowing mixed element geometries in the grid (simplices and quads atm)
- PDEDescription module and problem prototypes for easy problem description and discretisation setup
- PDEDescription recognizes nonlinear operators and automatically devises fixed-point or Newton algorithms (experimental)
- time-dependent solvers (only backward Euler for now)
- reconstruction operators for gradient-robust Stokes discretisation (BR>RT0 or CR>RT0 in 2D and 3D, more in progress)
- export into vtk datafiles for external plotting, internal plotting in 2D via ExtendableGrids.plot


### Installation
via Julia package manager in Julia 1.5 or above:

```@example
# latest stable version
(@v1.5) pkg> add GradientRobustMultiPhysics
# latest version
(@v1.5) pkg> add GradientRobustMultiPhysics#master
```

### EXAMPLES 
see documentation


### Dependencies on other Julia packages:

[ExtendableGrids.jl](https://github.com/j-fu/ExtendableGrids.jl)\
[ExtendableSparse.jl](https://github.com/j-fu/ExtendableSparse.jl)\
[DocStringExtensions.jl](https://github.com/JuliaDocs/DocStringExtensions.jl)\
[ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)\
[DiffResults.jl](https://github.com/JuliaDiff/DiffResults.jl)\
[BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl)
