#= 

# 2D Natural Convection
([source code](SOURCE_URL))

This example solves the natural convection (or Boussinesque) problem on a triangular domain ``\Omega = \mathrm{conv}\lbrace (0,0),(1,0),(0,1) \rbrace``. Altogether, we are looking for a velocity
``\mathbf{u}``, a pressure ``\mathbf{p}`` and a stemperature ``T`` such that
```math
\begin{aligned}
- \Delta \mathbf{u} + (\mathbf{u} \cdot \nabla) \mathbf{u} + \nabla p & = Ra \mathbf{e}_2 T\\
\mathrm{div}(u) & = 0\\
- \Delta \mathbf{T} + \mathbf{u} \cdot \nabla \mathbf{T} & = 0
\end{aligned}
```
with some parameter ``Ra``. The velocity has zero Dirichlet boundary conditions, while the temperature is zero along the y-axis, trigonometric along the x-axis and do-nothing at the diagonal boundary of the triangular domain.

Instead of using a Newton scheme, we solve a simpler fixpoint iteration plus [Anderson acceleration](@ref).
Also, note that a divergence-free reconstruction operator is used for the velocity, which also helps with the convergence and accuracy of the lowest-order method for this test problem.

=#


module Example_2DNaturalConvection

using GradientRobustMultiPhysics

## everything is wrapped in a main function
function main(; verbosity = 0, Plotter = nothing, Ra = 1e6, viscosity = 1)

    ## set log level
    set_verbosity(verbosity)
    
    ## load mesh and refine
    xgrid = reference_domain(Triangle2D)
    xgrid = uniform_refine(xgrid,5)

    ## types for discretisation by Bernardi--Raugel pressure-robust (BDM1 reconstruction) + P1-FEM for temperature
    FETypes = [H1BR{2}, H1P0{1}, H1P1{1}]; 
    postprocess_operator = ReconstructionIdentity{HDIVBDM1{2}}

    ## load Stokes prototype and add a unknown for the temperature
    Problem = IncompressibleNavierStokesProblem(2; viscosity = viscosity, nonlinear = true, auto_newton = false)
    add_unknown!(Problem; unknown_name = "temperature", equation_name = "temperature equation")
    Problem.name = "natural convection problem"

    ## add boundary data for velocity (unknown 1) and temperature (unknown 3)
    add_boundarydata!(Problem, 1, [1,2,3], HomogeneousDirichletBoundary)
    function bnd_data_bottom!(result,x)
        result[1] = 2*(1-cos(2*pi*x[1]))
    end
    add_boundarydata!(Problem, 3, [1], BestapproxDirichletBoundary; data = DataFunction(bnd_data_bottom!, [1,2]; dependencies = "X", quadorder = 4))
    add_boundarydata!(Problem, 3, [3], HomogeneousDirichletBoundary)

    ## add Laplacian to temperature equation
    add_operator!(Problem,[3,3], LaplaceOperator(1.0; store = true))

    ## add coupling terms for velocity and temperature
    add_operator!(Problem,[3,3], ConvectionOperator(1, postprocess_operator, 2, 1; auto_newton = false))
    function gravity_kernel(result,input)
        result[1] = -Ra*input[2]
    end
    add_operator!(Problem,[1,3], AbstractBilinearForm([postprocess_operator, Identity], Action(Float64, gravity_kernel, [1,2]; dependencies = "")))

    ## show final problem description
    @show Problem
    
    ## construct FESpaces and Solution veector
    FES = [FESpace{FETypes[1]}(xgrid), FESpace{FETypes[2]}(xgrid), FESpace{FETypes[3]}(xgrid)]
    Solution = FEVector{Float64}(["v_h", "p_h", "T_h"],FES)

    ## solve (fixedpoint iteration by solving consecutively equations [3] and [1,2] + Anderson acceleration)
    solve!(Solution, Problem; subiterations = [[3],[1,2]], maxiterations = 100, target_residual = 1e-8, anderson_iterations = 5, anderson_metric = "l2", anderson_unknowns = [1,3], anderson_damping = 0.95, show_solver_config = true)
    
    ## plot
    GradientRobustMultiPhysics.plot(xgrid, [Solution[1], Solution[3]], [Identity, Identity]; Plotter = Plotter)
end

end
