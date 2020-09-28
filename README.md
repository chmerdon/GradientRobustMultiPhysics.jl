[![Build Status](https://travis-ci.com/chmerdon/GradientRobustMultiPhysics.jl.svg?branch=master)](https://travis-ci.com/github/chmerdon/GradientRobustMultiPhysics.jl)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://chmerdon.github.io/GradientRobustMultiPhysics.jl/stable/index.html)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://chmerdon.github.io/GradientRobustMultiPhysics.jl/dev/index.html)


# GradientRobustMultiPhysics.jl

finite element module for Julia focussing on gradient-robust finite element methods and multiphysics applications


### Features/Limitations:
- solve 1D, 2D and 3D problems in Cartesian coordinates (3D features very limited at the moment)
- type-treed FiniteElements (scalar or vector-valued)
    - H1 elements (so far P1, P2, MINI, CR, BR)
    - Hdiv elements (so far RT0, BDM1, RT1)
    - L2 elements (so far P0, P1)
    - Hcurl elements (in future)
- based on ExtendableGrids.jl, allowing mixed element geometries in the grid (simplices and quads only atm)
- PDEDescription module and problem prototypes for easy problem description and discretisation setup
- PDEDescription recognizes nonlinear operators and automatically devises fixed-point or Newton algorithms (experimental)
- time-dependent solvers (only backward Euler for now)
- reconstruction operators for gradient-robust Stokes discretisation (BR>RT0 or CR>RT0 in 2D and 3D, more in progress)
- export into vtk datafiles for external plotting


### EXAMPLES 
see documentation


### Dependencies on other Julia packages:

[ExtendableGrids.jl](https://github.com/j-fu/ExtendableGrids.jl)\
[ExtendableSparse.jl](https://github.com/j-fu/ExtendableSparse.jl)\
[DocStringExtensions.jl](https://github.com/JuliaDocs/DocStringExtensions.jl)\
[ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)\
[DiffResults.jl](https://github.com/JuliaDiff/DiffResults.jl)\
[BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl)
