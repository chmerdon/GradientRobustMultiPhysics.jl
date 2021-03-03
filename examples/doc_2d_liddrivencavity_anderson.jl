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

## data
function boundary_data_top!(result)
    result[1] = 1;
    result[2] = 0;
end

## everything is wrapped in a main function
function main(; verbosity = 2, Plotter = nothing, viscosity = 5e-4, anderson_iterations = 10)

    ## grid
    xgrid = uniform_refine(grid_unitsquare(Triangle2D), 4);

    ## problem parameters
    maxIterations = 50  # termination criterion 1 for nonlinear mode
    maxResidual = 1e-10 # termination criterion 2 for nonlinear mode
    broken_p = false

    ## choose one of these (inf-sup stable) finite element type pairs
    #FETypes = [H1P2{2,2}, H1P1{1}] # Taylor--Hood
    #FETypes = [H1P2B{2,2}, H1P1{1}]; broken_p = true # P2-bubble
    #FETypes = [H1CR{2}, H1P0{1}] # Crouzeix--Raviart
    #FETypes = [H1MINI{2,2}, H1P1{1}] # MINI element on triangles only
    FETypes = [H1BR{2}, H1P0{1}]; broken_p = true # Bernardi--Raugel

    #####################################################################################    
    #####################################################################################

    ## negotiate data functions to the package
    user_function_bnd = DataFunction(boundary_data_top!, [2,2]; name = "u_bnd", dependencies = "", quadorder = 0)

    ## load Navier-Stokes problem prototype and assign data
    StokesProblem = IncompressibleNavierStokesProblem(2; viscosity = viscosity, nonlinear = true, auto_newton = false)
    add_boundarydata!(StokesProblem, 1, [1,2,4], HomogeneousDirichletBoundary)
    add_boundarydata!(StokesProblem, 1, [3], BestapproxDirichletBoundary; data = user_function_bnd)

    ## generate FESpaces
    FESpaceVelocity = FESpace{FETypes[1]}(xgrid)
    FESpacePressure = FESpace{FETypes[2]}(xgrid; broken = broken_p)
    Solution = FEVector{Float64}("Stokes velocity",FESpaceVelocity)
    append!(Solution,"Stokes pressure",FESpacePressure)

    ## set nonlinear options and Newton terms
    StokesProblem.LHSOperators[1,1][1].store_operator = true   
    Base.show(StokesProblem)

    ## solve Stokes problem
    solve!(Solution, StokesProblem; verbosity = verbosity, anderson_iterations = anderson_iterations, maxIterations = maxIterations, maxResidual = maxResidual)

    ## plot
    GradientRobustMultiPhysics.plot(Solution, [1,2], [Identity, Identity]; Plotter = Plotter, verbosity = verbosity)

end

end

#=
### Output of default main() run
=#
Example_2DLidDrivenCavityAnderson.main()