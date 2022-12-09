#= 

# 219 : Axisymmetric Stokes (2.5D)
([source code](SOURCE_URL))

This example solves the 3D Hagen-Poiseuille flow via 
the 2.5D exisymmetric formulation of the Stokes problem
that seeks a velocity ``\mathbf{u} = (u_z, u_r)``
and pressure ``p`` such that
```math
\begin{aligned}
- \mu\left(\partial^2_r + r^{-1} \partial_r + \partial^2_z - r^{-2} \right) u_r
+ \partial_r p & = \mathbf{f}_r\\
- \mu\left(\partial^2_r + r^{-1} \partial_r + \partial^2_z \right) u_z
+ \partial_z p & = \mathbf{f}_z\\
(\partial_r + r^{-1})u_r + \partial_z u_z & = 0
\end{aligned}
```
with exterior force ``\mathbf{f}`` and some viscosity parameter ``\mu``.

The axisymmetric formulation assumes that the velocity in some
3D-domain, that is obtained by rotation of a 2D domain ``\Omega``,
only depends on the distance ``r`` to the rotation axis and the
``z``-coordinate tangential to the x-axis, but not on the angular coordinate
of the cylindric coordinates.
The implementation employs ``r``-dependent bilinear forms and a
Cartesian grid for the 2D ``(z,r)`` domain that is assumed to be rotated
around the ``r=0``-axis.

This leads to the weak formulation
```math
\begin{aligned}
a(u,v) + b(p,v) & = (f,v) \\
         b(q,u) & = 0
\end{aligned}
```
with the bilinear forms
```math
\begin{aligned}
a(u,v) := \int_{\Omega} \left( \nabla u : \nabla v + r^{-2} u_r v_r \right) r dr dz
b(q,v) := \int_{\Omega} q \left( \mathrm{div}(v) + r^{-1} u_r \right) r dr dz
\end{aligned}
```
where the usual Cartesian differential operators can be used.


=#

module Example219_AxisysmmetricStokes25D

using GradientRobustMultiPhysics
using ExtendableGrids
using GridVisualize

## data for axisymmetric Hagen-Poiseuille flow
function exact_pressure!(μ)
    function closure(result,x)
        result[1] = μ*(-2*x[1]+1.0)
    end
end
function exact_velocity!(result,x)
    result[1] = (1.0-x[2]^2);
    result[2] = 0.0;
end

## custom bilinearform a for the axisymmetric Stokes problem
function ASLaplaceOperator(μ)
    function action_kernel!(result, input, x)
        r = x[2]
        ## input = [u,∇u] as a vector of length 6
        result[1] = 0
        result[2] = μ/r * input[2]
        result[3] = μ*r * input[3]
        result[4] = μ*r * input[4]
        result[5] = μ*r * input[5]
        result[6] = μ*r * input[6]
        ## result will be multiplied with [v,∇v]
        return nothing
    end
    action = Action(action_kernel!, [6,6]; dependencies = "X", bonus_quadorder = 4)
    return BilinearForm([OperatorPair{Identity,Gradient},OperatorPair{Identity,Gradient}], action; name = "ν(∇#A,∇#T) + (α#A + β⋅∇#A, #T)", transposed_assembly = true)
end

## custom bilinearform b for the axisymmetric Stokes problem
function ASPressureOperator()
    function action_kernel!(result, input, x)
        r = x[2]
        ## input = [u,divu] as a vector of length 3
        result[1] = -(r*input[3] + input[2])
        ## result will be multiplied with [q]
        return nothing
    end
    action = Action(action_kernel!, [1,3]; dependencies = "X", bonus_quadorder = 4)
    return BilinearForm([OperatorPair{Identity,Divergence},Identity], action; transposed_assembly = false)
end
function ASPressureOperatorTransposed()
    function action_kernel!(result, input, x)
        r = x[2]
        ## input = [q] as a vector of length 1
        result[1] = 0
        result[2] = -input[1]
        result[3] = -r*input[1]
        ## result will be multiplied with [v,divv]
        return nothing
    end
    action = Action(action_kernel!, [3,1]; dependencies = "X", bonus_quadorder = 4)
    return BilinearForm([Identity,OperatorPair{Identity,Divergence}], action; transposed_assembly = false)
end

## everything is wrapped in a main function
function main(; verbosity = 0, Plotter = nothing, μ = 1)

    ## set log level
    set_verbosity(verbosity)

    ## negotiate data functions to the package
    u = DataFunction(exact_velocity!, [2,2]; name = "u", dependencies = "X", bonus_quadorder = 2)
    p = DataFunction(exact_pressure!(μ), [1,2]; name = "p", dependencies = "X", bonus_quadorder = 1)

    ## grid
    xgrid = uniform_refine(grid_unitsquare(Triangle2D), 5);

    ## finite element type
    FETypes = [H1P2{2,2}, H1P1{1}] # Taylor--Hood

    ## problem description
    Problem = PDEDescription("axisymmetric Stokes")
    add_unknown!(Problem; unknown_name = "u", equation_name = "momentum balance")
    add_unknown!(Problem; unknown_name = "p", equation_name = "continuity equation")
    add_operator!(Problem, [1,1], ASLaplaceOperator(μ))
    add_operator!(Problem, [1,2], ASPressureOperator())
    add_operator!(Problem, [2,1], ASPressureOperatorTransposed())

    add_constraint!(Problem, FixedIntegralMean(2,0))
    add_boundarydata!(Problem, 1, [1,2,4], BestapproxDirichletBoundary; data = u)
    add_boundarydata!(Problem, 1, [3], HomogeneousDirichletBoundary)
    @show Problem

    ## generate FESpaces
    FES = [FESpace{FETypes[1]}(xgrid), FESpace{FETypes[2]}(xgrid)]
    Solution = FEVector(FES)

    ## solve
    solve!(Solution, Problem)

    ## plot
    p = GridVisualizer(; Plotter = Plotter, layout = (1,2), clear = true, resolution = (1000,500))
    scalarplot!(p[1,1],xgrid,view(nodevalues(Solution[1]; abs = true),1,:), levels = 9)
    scalarplot!(p[1,2],xgrid,view(nodevalues(Solution[2]),1,:), levels = 11, title = "p_h")
end

end