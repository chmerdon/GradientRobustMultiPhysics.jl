#= 

# A11 : Navier-Stokes Fixed-point iterations
([source code](SOURCE_URL))

This example computes the velocity ``\mathbf{u}`` and pressure ``\mathbf{p}`` of the incompressible Navier--Stokes problem
```math
\begin{aligned}
- \mu \Delta \mathbf{u} + (\mathbf{u} \¢dot \nabla) \mathbf{u} + \nabla p & = \mathbf{f}\\
\mathrm{div}(u) & = 0
\end{aligned}
```
with exterior force ``\mathbf{f}`` and some parameter ``\mu`` and inhomogeneous Dirichlet boundary data.

The convection term can be discretized in (at least) three different ways, leading to three different fixed-point iteration schemes:
- Newton iteration (when discretised as a NonlinearForm)
- Picard iteration (when discretised as a BilinearForm)
- fully explicit iteration (when discretised as a LinearForm)

This example script has a test case that checks that the result of all three iterations are the same.

=#

module ExampleA11_NavierStokesFixpointIterations

using GradientRobustMultiPhysics
using ExtendableGrids
using GridVisualize
using SimplexGridFactory
using Triangulate

## flow data for boundary condition, right-hand side and error calculation
function get_flowdata(μ)
    u = DataFunction((result,x) -> (
        result[1] = sin(2*pi*x[1])*sin(2*pi*x[2]);
        result[2] = cos(2*pi*x[1])*cos(2*pi*x[2]);
        ), [2,2]; dependencies = "X", name = "u", bonus_quadorder = 5)
    p = DataFunction((result,x) -> (result[1] = (cos(4*pi*x[1])-cos(4*pi*x[2])) / 4), [1,2]; dependencies = "X", name = "p")
    Δu, ∇p = Δ(u), ∇(p)
    f = DataFunction((result,x) -> (
        result .= -μ*Δu(x) + ∇p(x)
        ), [2,2]; dependencies = "X", name = "f", bonus_quadorder = 5)
    return u, f
end

## everything is wrapped in a main function
function main(;
    μ = 1e-1,               # viscosity
    Plotter = nothing,      # Plotter for visualization (e.g. PyPlot)
    iterationtype = 1,      # convection term discretisation (1 = Newton, 2 = Picard, 3 = just in right-hand side)
    verbosity = 0)

    ## set log level
    set_verbosity(verbosity)

    ## FEType (Hdiv-conforming)
    FETypes = [H1P2{2,2}, L2P1{1}] # Scott-Vogelius
    
    ## load exact flow data
    u, f = get_flowdata(μ)

    ## problem description
    Problem = PDEDescription("Oseen Equations")
    add_unknown!(Problem; equation_name = "momentum equation", unknown_name = "u")
    add_unknown!(Problem; equation_name = "incompressibility constraint", unknown_name = "p", algebraic_constraint = true)
    add_operator!(Problem, [1,1], LaplaceOperator(μ))
    add_operator!(Problem, [1,2], LagrangeMultiplier(Divergence))

    ## convection term discretised in 3 ways
    function convection_kernel(result, input)
        uh, ∇uh = view(input,1:2), view(input,3:6)
        result[1] = ∇uh[1]*uh[1] + ∇uh[2]*uh[2]
        result[2] = ∇uh[3]*uh[1] + ∇uh[4]*uh[2]
    end

    ## add convection term as chosen by iterationtype
    if iterationtype == 1 # Newton for c(u_h, u_h, v_h)
        add_operator!(Problem, 1, NonlinearForm(Identity, [Identity, Gradient], [1,1], convection_kernel, [2,6]))
    elseif iterationtype == 2 # Picard (adds c(u_old, u_h, v_h) on left-hand side)
        add_operator!(Problem, [1,1], BilinearForm([Gradient, Identity], [Identity], [1], Action(convection_kernel, [2,6]); transposed_assembly = true))
    elseif iterationtype == 3 # fully explicit (adds c(u_old, u_old, v_h) on right-hand side)
        add_rhsdata!(Problem, 1, LinearForm(Identity, [Identity, Gradient], [1,1], Action(convection_kernel, [2,6]); factor = -1))
    end

    ## add right-hand side data 
    add_rhsdata!(Problem, 1, LinearForm(Identity, f))

    ## add boundary data (fixes normal components of along boundary)
    add_boundarydata!(Problem, 1, [1,2,3,4], InterpolateDirichletBoundary; data = u)
    add_constraint!(Problem, FixedIntegralMean(2,0))

    ## show final problem description (without stabilizing terms)
    @show Problem

    ## get grid and barycentric refinement
    xgrid = barycentric_refine(uniform_refine(grid_unitsquare(Triangle2D), 3))

    ## generate FES spaces and solution vector
    FES = [FESpace{FETypes[1]}(xgrid), FESpace{FETypes[2]}(xgrid)]
    Solution = FEVector(["u_h", "p_h"],FES)

    ## solve
    solve!(Solution, Problem; skip_update = iterationtype == 3 ? -1 : 1, maxiterations = 20, target_residual = 1e-13, show_statistics = true)

    ## plot last solution and convergence hisotry
    p = GridVisualizer(; Plotter = Plotter, layout = (1,3), clear = true, resolution = (1500,500))
    scalarplot!(p[1,1], xgrid, view(nodevalues(Solution[1]; abs = true), 1, :), levels = 3, colorbarticks = 9, title = "u_h (abs + quiver)")
    vectorplot!(p[1,1], xgrid, evaluate(PointEvaluator(Solution[1], Identity)), spacing = 0.05, clear = false)
    scalarplot!(p[1,2], xgrid, view(nodevalues(Solution[2]),1,:), levels = 7, title = "p_h")

    return Solution
end

## checks if the solutions of all three iteration schemes are the same
function test(; μ = 1e-1)
    SolutionNLF = main(; μ = μ, iterationtype = 1)
    SolutionBLF = main(; μ = μ, iterationtype = 2)
    SolutionLF = main(; μ = μ, iterationtype = 3)
    distance1 = maximum(SolutionNLF.entries - SolutionBLF.entries)
    distance2 = maximum(SolutionNLF.entries - SolutionLF.entries)
    return max(distance1, distance2)
end

end