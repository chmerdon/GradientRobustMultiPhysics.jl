#= 

# 223 : Natural Convection 2D
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


module Example223_NaturalConvection2D

using GradientRobustMultiPhysics

# boundary data for temperature on bottom
T_bottom = DataFunction((T,x) -> (T[1] = 2*(1-cos(2*pi*x[1]))), [1,2]; dependencies = "X", quadorder = 4)

## everything is wrapped in a main function
function main(; verbosity = 0, Plotter = nothing, Ra = 1e6, viscosity = 1, nrefinements = 6)

    ## set log level
    set_verbosity(verbosity)
    
    ## load mesh and refine
    xgrid = reference_domain(Triangle2D)
    xgrid = uniform_refine(xgrid, nrefinements)

    ## types for discretisation by Bernardi--Raugel pressure-robust (BDM1 reconstruction) + P1-FEM for temperature
    FETypes = [H1BR{2}, H1P0{1}, H1P1{1}]; 
    RIdentity = ReconstructionIdentity{HDIVBDM1{2}}

    ## load Stokes prototype and add a unknown for the temperature
    Problem = IncompressibleNavierStokesProblem(2; viscosity = viscosity, nonlinear = false, store = true)
    add_unknown!(Problem; unknown_name = "T", equation_name = "temperature equation")
    Problem.name = "natural convection problem"

    ## add convection term for velocity
    add_operator!(Problem, [1,1], ConvectionOperator(1, RIdentity, 2, 2; testfunction_operator = RIdentity, auto_newton = false))

    ## add boundary data for velocity (unknown 1) and temperature (unknown 3)
    add_boundarydata!(Problem, 1, [1,2,3], HomogeneousDirichletBoundary)
    add_boundarydata!(Problem, 3, [1], BestapproxDirichletBoundary; data = T_bottom)
    add_boundarydata!(Problem, 3, [3], HomogeneousDirichletBoundary)

    ## add Laplacian to temperature equation
    add_operator!(Problem,[3,3], LaplaceOperator(1.0; store = true, name = "∇(T)⋅∇(V)"))

    ## add coupling terms for velocity and temperature (convection + gravity)
    add_operator!(Problem,[3,3], ConvectionOperator(1, RIdentity, 2, 1; auto_newton = false, name = "(R(u)⋅∇(T)) V"))
    add_operator!(Problem,[1,3], AbstractBilinearForm([RIdentity, Identity], fdot_action(Float64,DataFunction([0,-1.0])); factor = Ra, name = "-Ra v⋅g T", store = true))

    ## show final problem description
    @show Problem
    
    ## construct FESpaces and Solution veector
    FES = [FESpace{FETypes[1]}(xgrid), FESpace{FETypes[2]}(xgrid), FESpace{FETypes[3]}(xgrid)]
    Solution = FEVector{Float64}(["v_h", "p_h", "T_h"],FES)

    ## solve (fixedpoint iteration by solving consecutively equations [3] and [1,2] + Anderson acceleration)
    solve!(Solution, Problem; subiterations = [[3],[1,2]], maxiterations = 100, target_residual = 1e-8, anderson_iterations = 5, anderson_metric = "l2", anderson_unknowns = [1,3], anderson_damping = 0.95, show_solver_config = true)
    
    # compute Nusselt number along bottom boundary
    NuIntegrator = ItemIntegrator(Float64,ON_BFACES,[Jump(Gradient)], fdot_action(Float64,DataFunction([0,-1.0])); regions = [1])
    println("\tNu = $(evaluate(NuIntegrator,[Solution[3]]))")

    ## plot
    GradientRobustMultiPhysics.plot(xgrid, [Solution[1], Solution[3]], [Identity, Identity]; Plotter = Plotter)
end

end
