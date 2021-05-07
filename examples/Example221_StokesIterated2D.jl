#= 

# 221 : Stokes iterated penalty method 2D
([source code](SOURCE_URL))

This example computes a velocity ``\mathbf{u}`` and pressure ``\mathbf{p}`` of the incompressible Navier--Stokes problem
```math
\begin{aligned}
- \mu \Delta \mathbf{u} + (\mathbf{u} \cdot \nabla) \mathbf{u} + \nabla p & = \mathbf{0}\\
\mathrm{div}(u) & = 0
\end{aligned}
```
with some viscosity parameter ``\mu``.

Here we solve the simple Hagen-Poiseuille flow on the two-dimensional unit square domain with the iterated penalty method.
Given intermediate solution  ``\mathbf{u}`` and  ``p`` the next approximations are computed by the two equations

```math
\begin{aligned}
(\nabla \mathbf{u}^{next}, \nabla \mathbf{v}) + ((\mathbf{u}^{next} \cdot \nabla) \mathbf{u}^{next}, \mathbf{v}) + \epsilon (\mathrm{div}(\mathbf{u}) ,\mathrm{div}(v)) & = (\mathbf{f},\mathbf{v}) + (p,\mathrm{div}(\mathbf{v}))
&& \text{for all } \mathbf{v} \in \mathbf{V}\\
(p^{next},q) & = (p,q) - (\mathrm{div}(\mathbf{u}^{next}),q) && \text{for all } \mathbf{q} \in Q
\end{aligned}
```

This is done consecutively until the residual of both equations is small enough. (The convection term is linearised by auto-differentiated Newton terms.)
=#

module Example221_StokesIterated2D

using GradientRobustMultiPhysics
using ExtendableGrids
using Printf

## data for Hagen-Poiseuille flow
function exact_pressure!(viscosity)
    function closure(result,x::Array{<:Real,1})
        result[1] = viscosity*(-2*x[1]+1.0)
    end
end
function exact_velocity!(result,x::Array{<:Real,1})
    result[1] = x[2]*(1.0-x[2]);
    result[2] = 0.0;
end

## everything is wrapped in a main function
function main(; verbosity = 0, Plotter = nothing, nonlinear = false, div_penalty = 1e4, viscosity = 1.0)

    ## set verbosity level
    set_verbosity(verbosity)

    ## initial grid
    xgrid = uniform_refine(grid_unitsquare(Triangle2D),4);

    ## Taylor--Hood element
    FETypes = [H1P2{2,2}, H1P1{1}];

    ## negotiate data functions to the package
    user_function_velocity = DataFunction(exact_velocity!, [2,2]; name = "u", dependencies = "X", quadorder = 2)
    user_function_pressure = DataFunction(exact_pressure!(viscosity), [1,2]; name = "p", dependencies = "X", quadorder = 1)

    ## generate Stokes problem
    Problem = PDEDescription("NSE (iterated penalty)")
    add_unknown!(Problem; equation_name = "velocity update", unknown_name = "u")
    add_unknown!(Problem; equation_name = "pressure update", unknown_name = "p")
    add_constraint!(Problem, FixedIntegralMean(2,0))

    ## add boundary data
    add_boundarydata!(Problem, 1, [1,2,3,4], InterpolateDirichletBoundary; data = user_function_velocity)

    ## velocity update equation
    add_operator!(Problem, [1,1], LaplaceOperator(viscosity; store = true))
    add_operator!(Problem, [1,2], AbstractBilinearForm([Divergence, Identity]; name = "(div(v),p)", store = true, factor = -1))
    add_operator!(Problem, [1,1], ConvectionOperator(1, Identity, 2, 2; auto_newton = true))
    add_operator!(Problem, [1,1], AbstractBilinearForm([Divergence, Divergence]; name = "ϵ (div(u),div(v))", store = true, factor = div_penalty))

    ## pressure update equation
    BLF_MAMA = AbstractBilinearForm([Identity, Identity]; name = "(p,q)")
    add_operator!(Problem, [2,1], AbstractBilinearForm([Identity, Divergence]; name = "(q,div(u))", store = true, factor = div_penalty))
    add_operator!(Problem, [2,2], BLF_MAMA)
    add_rhsdata!(Problem, 2, restrict_operator(BLF_MAMA; fixed_arguments = [1], fixed_arguments_ids = [2]))

    @show Problem

    ## discretise and solve problem
    FES = [FESpace{FETypes[1]}(xgrid), FESpace{FETypes[2]}(xgrid; broken = true)]
    Solution = FEVector{Float64}(["u_h","p_h"],[FES[1],FES[2]])
    solve!(Solution, Problem; subiterations = [[1],[2]], maxiterations = 20, show_solver_config = true)

    ## calculate L2 error and L2 curl error
    L2ErrorEvaluatorV = L2ErrorIntegrator(Float64, user_function_velocity, Identity)
    L2ErrorEvaluatorP = L2ErrorIntegrator(Float64, user_function_pressure, Identity)
    println("|| u - u_h || = $(sqrt(evaluate(L2ErrorEvaluatorV,Solution[1])))")
    println("|| p - p_h || = $(sqrt(evaluate(L2ErrorEvaluatorP,Solution[2])))")

    ## plot
    GradientRobustMultiPhysics.plot(xgrid, [Solution[1], Solution[2]], [Identity, Identity]; Plotter = Plotter)
end
end