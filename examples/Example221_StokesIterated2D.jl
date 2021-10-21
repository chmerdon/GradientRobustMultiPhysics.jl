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

Here we solve the simple Hagen-Poiseuille flow on the two-dimensional unit square domain with the iterated penalty method
for the Bernardi--Raugel finite element method.
Given intermediate solutions  ``\mathbf{u}_h`` and  ``p_h`` the next approximations are computed by the two equations

```math
\begin{aligned}
(\nabla \mathbf{u}_h^{next}, \nabla \mathbf{v}_h) + ((\mathbf{u}_h^{next} \cdot \nabla) \mathbf{u}_h^{next}, \mathbf{v}_h) + \epsilon (\mathrm{div}_h(\mathbf{u}_h) ,\mathrm{div}_h(\mathbf{v}_h)) & = (\mathbf{f},\mathbf{v}_h) + (p_h,\mathrm{div}(\mathbf{v}_h))
&& \text{for all } \mathbf{v}_h \in \mathbf{V}_h\\
(p^{next}_h,q_h) & = (p_h,q_h) - (\mathrm{div}(\mathbf{u}_h^{next}),q_h) && \text{for all } q_h \in Q_h
\end{aligned}
```

This is done consecutively until the residual of both equations is small enough.
The convection term is linearised by auto-differentiated Newton terms.
The discrete divergence is computed via a RT0 reconstruction operator that preserves the disrete divergence.
(another way would be to compute B*inv(M)*B' where M is the mass matrix of the pressure and B is the matrix for the div-pressure block).
=#

module Example221_StokesIterated2D

using GradientRobustMultiPhysics
using ExtendableGrids
using ExtendableSparse
using GridVisualize

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
    xgrid = uniform_refine(grid_unitsquare(Triangle2D),4)

    ## Bernardi--Raugel element
    FETypes = [H1BR{2}, H1P0{1}]; PenaltyDivergence = ReconstructionDivergence{HDIVRT0{2}}

    ## FE spaces
    FES = [FESpace{FETypes[1]}(xgrid), FESpace{FETypes[2]}(xgrid; broken = true)]

    ## negotiate data functions to the package
    u = DataFunction(exact_velocity!, [2,2]; name = "u", dependencies = "X", quadorder = 2)
    p = DataFunction(exact_pressure!(viscosity), [1,2]; name = "p", dependencies = "X", quadorder = 1)

    ## generate Stokes problem
    Problem = PDEDescription("NSE (iterated penalty)")
    add_unknown!(Problem; equation_name = "velocity update", unknown_name = "u")
    add_unknown!(Problem; equation_name = "pressure update", unknown_name = "p")
    add_constraint!(Problem, FixedIntegralMean(2,0))

    ## add boundary data
    add_boundarydata!(Problem, 1, [1,2,3,4], InterpolateDirichletBoundary; data = u)

    ## velocity update equation
    add_operator!(Problem, [1,1], LaplaceOperator(viscosity; store = true))
    add_operator!(Problem, [1,2], AbstractBilinearForm([Divergence, Identity]; name = "(div(v),p)", store = true, factor = -1))
    add_operator!(Problem, [1,1], ConvectionOperator(1, Identity, 2, 2; auto_newton = true))

    ## add penalty for discrete divergence
    add_operator!(Problem, [1,1], AbstractBilinearForm([PenaltyDivergence, PenaltyDivergence]; name = "ϵ (div_h(u),div_h(v))", store = true, factor = div_penalty))

    ## pressure update equation
    PressureMAMA = AbstractBilinearForm([Identity, Identity]; name = "(p,q)", store = true)
    add_operator!(Problem, [2,1], AbstractBilinearForm([Identity, Divergence]; name = "(q,div(u))", store = true, factor = div_penalty))
    add_operator!(Problem, [2,2], PressureMAMA)
    add_rhsdata!(Problem, 2, restrict_operator(PressureMAMA; fixed_arguments = [1], fixed_arguments_ids = [2]))

    ## show and solve problem
    @show Problem
    Solution = FEVector{Float64}(["u_h","p_h"],[FES[1],FES[2]])
    solve!(Solution, Problem; subiterations = [[1],[2]], maxiterations = 20, show_solver_config = true)

    ## calculate L2 error
    L2ErrorEvaluatorV = L2ErrorIntegrator(Float64, u, Identity)
    L2ErrorEvaluatorP = L2ErrorIntegrator(Float64, p, Identity)
    println("|| u - u_h || = $(sqrt(evaluate(L2ErrorEvaluatorV,Solution[1])))")
    println("|| p - p_h || = $(sqrt(evaluate(L2ErrorEvaluatorP,Solution[2])))")

    ## plot
    p=GridVisualizer(;Plotter=Plotter,layout=(1,2),clear=true,resolution=(1000,500))
    scalarplot!(p[1,1],xgrid,view(Solution.entries,1:num_nodes(xgrid)),levels=0)
    PE = PointEvaluator(Solution[1], Identity)
    vectorplot!(p[1,1],xgrid,evaluate(PE);Plotter=Plotter, spacing = [0.2,0.05], clear = false, title = "u (abs + quiver)")
    nodevals = zeros(Float64,1,num_nodes(xgrid))
    nodevalues!(nodevals, Solution[2], Identity)
    scalarplot!(p[1,2],xgrid,view(nodevals,1,:); Plotter=Plotter, title = "p")
end
end