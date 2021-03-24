#= 

# 2D Lid-driven cavity (Anderson-Iteration)
([source code](SOURCE_URL))

This example solves the lid-driven cavity problem where one seeks
a velocity ``\mathbf{u}`` and pressure ``\mathbf{p}`` of the incompressible Navier--Stokes problem
```math
\begin{aligned}
- \mu \Delta \mathbf{u} + (\mathbf{u} \cdot \nabla) \mathbf{u} + \nabla p & = 0\\
\mathrm{div}(u) & = 0
\end{aligned}
```
where ``\mathbf{u} = (1,0)`` along the top boundary of a square domain.

For small viscosities (where a Newton and a classical Picard iteration do not converge anymore),
Anderson acceleration might help (see https://arxiv.org/pdf/1810.08494.pdf) which can be tested with this script.

=#

module Example_2DLidDrivenCavityAnderson

using GradientRobustMultiPhysics
using ExtendableGrids
using Printf

## everything is wrapped in a main function
function main(; verbosity = 2, Plotter = nothing, viscosity = 5e-4, anderson_iterations = 10, maxResidual = 1e-10, maxIterations = 50)

    ## grid
    xgrid = uniform_refine(grid_unitsquare(Triangle2D), 4);

    ## choose one of these (inf-sup stable) finite element type pairs
    broken_p = false
    #FETypes = [H1P2{2,2}, H1P1{1}] # Taylor--Hood
    #FETypes = [H1P2B{2,2}, H1P1{1}]; broken_p = true # P2-bubble
    #FETypes = [H1CR{2}, H1P0{1}] # Crouzeix--Raviart
    #FETypes = [H1MINI{2,2}, H1P1{1}] # MINI element on triangles only
    FETypes = [H1BR{2}, H1P0{1}]; broken_p = true # Bernardi--Raugel
  
    #####################################################################################

    ## load Navier-Stokes problem prototype and assign data
    Problem = IncompressibleNavierStokesProblem(2; viscosity = viscosity, nonlinear = true, auto_newton = false)
    add_boundarydata!(Problem, 1, [1,2,4], HomogeneousDirichletBoundary)
    add_boundarydata!(Problem, 1, [3], BestapproxDirichletBoundary; data = DataFunction([1,0]))
    Base.@show Problem

    ## generate FESpaces
    FES = [FESpace{FETypes[1]}(xgrid), FESpace{FETypes[2]}(xgrid; broken = broken_p)]
    Solution = FEVector{Float64}(["velocity", "pressure"],FES)

    ## solve Stokes problem
    solve!(Solution, Problem; verbosity = verbosity, anderson_iterations = anderson_iterations, maxIterations = maxIterations, maxResidual = maxResidual)

    ## plot
    GradientRobustMultiPhysics.plot(xgrid, [Solution[1],Solution[2]], [Identity, Identity]; Plotter = Plotter, verbosity = verbosity)
end

end

#=
### Output of default main() run
=#
Example_2DLidDrivenCavityAnderson.main()