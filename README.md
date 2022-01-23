[![Build status](https://github.com/chmerdon/GradientRobustMultiPhysics.jl/workflows/linux-macos-windows/badge.svg)](https://github.com/chmerdon/GradientRobustMultiPhysics.jl/actions)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://chmerdon.github.io/GradientRobustMultiPhysics.jl/stable/index.html)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://chmerdon.github.io/GradientRobustMultiPhysics.jl/dev/index.html)
[![DOI](https://zenodo.org/badge/229078096.svg)](https://zenodo.org/badge/latestdoi/229078096)


# GradientRobustMultiPhysics.jl

finite element module for Julia focussing on gradient-robust finite element methods and multiphysics applications, part of the meta-package [PDELIB.jl](https://github.com/WIAS-BERLIN/PDELib.jl)


### Features/Limitations:
- solves 1D, 2D and 3D problems in Cartesian coordinates
- several available finite elements (scalar and vector-valued, H1, Hdiv and Hcurl on different geometries), see [here](https://chmerdon.github.io/GradientRobustMultiPhysics.jl/stable/fems/) for a complete list
- finite elements can be broken (e.g. piecewise Hdiv) or live on faces or edges (experimental feature)
- grids by [ExtendableGrids.jl](https://github.com/j-fu/ExtendableGrids.jl) 
- PDEDescription module for easy and close-to-physics problem description (as variational equations) independent from the actual discretisation
- Newton terms for nonlinear operators are added automatically by automatic differentiation (experimental feature)
- solver can run fixed-point iterations between subsets of equations of the PDEDescription (possibly with Anderson acceleration)
- time-dependent problems can be integrated in time by internal backward Euler or Crank-Nicolson implementation or via the external module [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) (experimental)
- reconstruction operators for gradient-robust Stokes discretisations (BR->RT0/BDM1 or CR->RT0 in 2D/3D, and P2B->RT1/BDM2 in 2D, more to come)
- plotting via functionality of [GridVisualize.jl](https://github.com/j-fu/GridVisualize.jl)
- export into csv files or vtk files (via [WriteVTK.jl](https://github.com/jipolanco/WriteVTK.jl))


### Quick Example

The following minimal example demonstrates how to setup a Poisson problem.

```julia
using GradientRobustMultiPhysics
using ExtendableGrids

# build/load any grid (here: a uniform-refined 2D unit square into triangles)
xgrid = uniform_refine(grid_unitsquare(Triangle2D), 4)

# create empty PDE description
Problem = PDEDescription("Poisson problem")

# add unknown(s) (here: "u" that gets id 1 for later reference)
add_unknown!(Problem; unknown_name = "u", equation_name = "Poisson equation")

# add left-hand side PDEoperator(s) (here: only Laplacian with diffusion coefficient 1e-3)
add_operator!(Problem, [1,1], LaplaceOperator(1e-3))

# define right-hand side function (as a constant DataFunction, x and t dependency explained in documentation)
f = DataFunction([1]; name = "f")

# add right-hand side data (here: f = [1] in region(s) [1])
add_rhsdata!(Problem, 1, RhsOperator(Identity, f; regions = [1]))

# add boundary data (here: zero data for boundary regions 1:4)
add_boundarydata!(Problem, 1, [1,2,3,4], HomogeneousDirichletBoundary)

# discretise = choose FEVector with appropriate FESpaces
FEType = H1P2{1,2} # quadratic element with 1 component in 2D
Solution = FEVector("u_h",FESpace{FEType}(xgrid))

# inspect problem and Solution vector structure
@show Problem Solution

# solve
solve!(Solution, Problem)
```

### Other Examples

More extensive examples can be found in the [documentation](https://chmerdon.github.io/GradientRobustMultiPhysics.jl/stable/index.html)
and interactive [Pluto](https://github.com/fonsp/Pluto.jl) notebooks can be found in the subfolder examples/pluto for download.


### Installation
via Julia package manager in Julia 1.6 or above:

```@example
# latest stable version
(@v1.6) pkg> add GradientRobustMultiPhysics
# latest version
(@v1.6) pkg> add GradientRobustMultiPhysics#master
```


### Dependencies on other Julia packages:

[ExtendableGrids.jl](https://github.com/j-fu/ExtendableGrids.jl)\
[GridVisualize.jl](https://github.com/j-fu/GridVisualize.jl)\
[ExtendableSparse.jl](https://github.com/j-fu/ExtendableSparse.jl)\
[DocStringExtensions.jl](https://github.com/JuliaDocs/DocStringExtensions.jl)\
[ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)\
[DiffResults.jl](https://github.com/JuliaDiff/DiffResults.jl)\
[WriteVTK.jl](https://github.com/jipolanco/WriteVTK.jl)\
[StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl)
