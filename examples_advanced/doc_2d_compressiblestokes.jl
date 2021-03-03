#= 

# 2D Compressible Stokes
([source code](SOURCE_URL))

This example solves the compressible Stokes equations where one seeks a (vector-valued) velocity ``\mathbf{u}``, a density ``\varrho`` and a pressure ``p`` such that
```math
\begin{aligned}
- \mu \Delta \mathbf{u} + \lambda \nabla(\mathrm{div}(\mathbf{u})) + \nabla p & = \mathbf{f} + \varrho \mathbf{g}\\
\mathrm{div}(\varrho \mathbf{u}) & = 0\\
        p & = eos(\varrho)\\
        \int_\Omega \varrho \, dx & = M\\
        \varrho & \geq 0.
\end{aligned}
```
Here eos ``eos`` is some equation of state function that describes the dependence of the pressure on the density
(and further physical quantities like temperature in a more general setting).
Moreover, ``\mu`` and ``\lambda`` are Lame parameters and ``\mathbf{f}`` and ``\mathbf{g}`` are given right-hand side data.

In this example we solve a analytical toy problem with the prescribed solution
```math
\begin{aligned}
\mathbf{u}(\mathbf{x}) & =0\\
\varrho(\mathbf{x}) & = 1 - (x_2 - 0.5)/c\\
p &= eos(\varrho) := c \varrho^\gamma
\end{aligned}
```
such that ``\mathbf{f} = 0`` and ``\mathbf{g}`` nonzero to match the prescribed solution.
This example is designed to study the well-balanced property of a discretisation. Note that a gradient-robust discretisation (set reconstruct = true below)
has a much smaller L2 velocity error (i.e. approximatse the well-balanced state much better). For larger c the problem gets more incompressible which reduces
the error further as then the right-hand side is a perfect gradient also when evaluated with the (now closer to a constant) discrete density.
See reference below for more details.

!!! reference

    "A gradient-robust well-balanced scheme for the compressible isothermal Stokes problem",\
    M. Akbas, T. Gallouet, A. Gassmann, A. Linke and C. Merdon,\
    Computer Methods in Applied Mechanics and Engineering 367 (2020),\
    [>Journal-Link<](https://doi.org/10.1016/j.cma.2020.113069)
    [>Preprint-Link<](https://arxiv.org/abs/1911.01295)

=#


module Example_2DCompressibleStokes

using GradientRobustMultiPhysics
using ExtendableGrids
using Printf


## the equation of state
function equation_of_state!(c,gamma)
    function closure(pressure,density)
        for j = 1 : length(density)
            pressure[j] = c*density[j]^gamma
        end
    end
end

## the exact density (used for initial value of density if configured so)
function exact_density!(M,c)
    function closure(result,x::Array{<:Real,1})
        result[1] = M*(1.0 - (x[2] - 0.5)/c)
    end
end 

## the exact velocity (zero!)
function exact_velocity!(result)
    result[1] = 0.0
    result[2] = 0.0
end 

## gravity right-hand side (just gravity but with opposite sign!)
function rhs_gravity!(gamma,c)
    function closure(result,x::Array{<:Real,1})
        result[1] = 1.0 - (x[2] - 0.5)/c # = density
        result[2] = - result[1]^(gamma-2) * gamma
        result[1] = 0.0
    end
end   

## everything is wrapped in a main function
function main(; verbosity = 1, Plotter = nothing, reconstruct::Bool = true, c = 10, gamma = 1.4, M = 1, shear_modulus = 1e-3, lambda = -1e-3/3)

    ## load mesh and refine
    xgrid = simplexgrid("assets/2d_grid_mountainrange.sg")
    xgrid = uniform_refine(xgrid,1)

    ## solve without and with reconstruction
    Solution = setup_and_solve(xgrid; reconstruct = false, c = c, M = M, lambda = lambda, shear_modulus = shear_modulus, gamma = gamma, verbosity = verbosity)
    ## plots
    GradientRobustMultiPhysics.plot(Solution, [1,2,3], [Identity, Identity, Identity]; Plotter = Plotter, verbosity = verbosity - 1)

    Solution2 = setup_and_solve(xgrid; reconstruct = true, c = c, M = M, lambda = lambda, shear_modulus = shear_modulus, gamma = gamma, verbosity = verbosity)
    ## plots
    GradientRobustMultiPhysics.plot(Solution2, [1,2,3], [Identity, Identity, Identity]; Plotter = Plotter, verbosity = verbosity - 1)

    ## compare L2 error for velocity and density
    user_velocity = DataFunction(exact_velocity!, [2,2]; name = "u_exact", dependencies = "", quadorder = 0)
    user_density = DataFunction(exact_density!(M,c), [1,2]; name = "rho_exact", dependencies = "X", quadorder = 3)
    L2VelocityErrorEvaluator = L2ErrorIntegrator(Float64, user_velocity, Identity)
    L2DensityErrorEvaluator = L2ErrorIntegrator(Float64, user_density, Identity)
    L2error = sqrt(evaluate(L2VelocityErrorEvaluator,Solution[1]))
    L2error2 = sqrt(evaluate(L2VelocityErrorEvaluator,Solution2[1]))
    @printf("\n        reconstruct     false    |    true\n")
    @printf("================================================\n")
    @printf("L2error(Velocity) | %.5e  | %.5e \n",L2error,L2error2)
    L2error = sqrt(evaluate(L2DensityErrorEvaluator,Solution[2]))
    L2error2 = sqrt(evaluate(L2DensityErrorEvaluator,Solution2[2]))
    @printf("L2error(Density)  | %.5e  | %.5e \n",L2error,L2error2)
    
end

function setup_and_solve(xgrid; reconstruct = true, c = 1, gamma = 1, M = 1, shear_modulus = 1, lambda = 0, verbosity = 0)

    ## negotiate edata functions to the package
    user_velocity = DataFunction(exact_velocity!, [2,2]; name = "u_exact", dependencies = "", quadorder = 0)
    user_density = DataFunction(exact_density!(M,c), [1,2]; name = "rho_exact", dependencies = "X", quadorder = 3)
    user_gravity = DataFunction(rhs_gravity!(gamma,c), [2,2]; name = "g", dependencies = "X", quadorder = 3)

    ## solver parameters
    timestep = shear_modulus / (2*c)
    initial_density_bestapprox = true # otherwise we start with a constant density which also works but takes longer
    maxTimeSteps = 1000  # termination criterion 1
    stationarity_threshold = 1e-13/shear_modulus # stop when change is below this treshold

    ## set finite element type [velocity, density,  pressure]
    FETypes = [H1BR{2}, H1P0{1}, H1P0{1}] # Bernardi--Raugel
    
    ## set function operators depending on reconstruct
    if reconstruct
        VeloIdentity = ReconstructionIdentity{HDIVBDM1{2}} # identity operator for gradient-robust scheme
        VeloDivergence = ReconstructionDivergence{HDIVBDM1{2}} # divergence operator for gradient-robust scheme
    else # classical choices
        VeloIdentity = Identity
        VeloDivergence = Divergence
    end


    ## generate empty PDEDescription for three unknowns
    ## unknown 1 : velocity (vector-valued)
    ## unknown 2 : density
    ## unknown 3 : pressure
    Problem = PDEDescription("compressible Stokes problem")
    add_unknown!(Problem; unknown_name = "velocity", equation_name = "momentum equation")
    add_unknown!(Problem; unknown_name = "density", equation_name = "continuity equation")
    add_unknown!(Problem; unknown_name = "pressure", equation_name = "equation of state")
    add_boundarydata!(Problem, 1,  [1,2,3,4], HomogeneousDirichletBoundary)

    ## momentum equation
    add_operator!(Problem, [1,1], LaplaceOperator(2*shear_modulus,2,2; store = true))
    if lambda != 0
        add_operator!(Problem, [1,1], AbstractBilinearForm("lambda * div(u) * div(v) (lambda = $lambda)",VeloDivergence,VeloDivergence,MultiplyScalarAction(lambda,1); store = true))
    end
    add_operator!(Problem, [1,3], AbstractBilinearForm("div(v)*p",Divergence,Identity,MultiplyScalarAction(-1.0, 1); store = true))

    function gravity_action()
        temp = zeros(Float64,2)
        function closure(result,input,x, t)
            eval!(temp, user_gravity, x, t)
            result[1] = - temp[1] * input[1] - temp[2] * input[2]
        end
        gravity_action_kernel = ActionKernel(closure, [1,2]; name = "gravity action kernel", dependencies = "XT", quadorder = user_gravity.quadorder)
        return Action(Float64, gravity_action_kernel)
    end    
    add_operator!(Problem, [1,2], AbstractBilinearForm("g*v*rho",VeloIdentity,Identity,gravity_action(); store = true))

    ## continuity equation
    ## here a finite volume upwind convection operator on triangles is used
    add_operator!(Problem, [2,2], FVConvectionDiffusionOperator(1))

    ## equation of state
    ## here we do some best-approximation of the pressure that comes out of the equation of state
    eos_action_kernel = ActionKernel(equation_of_state!(c,gamma),[1,1]; dependencies = "", quadorder = 0)
    eos_action = Action(Float64, eos_action_kernel)
    add_operator!(Problem, [3,2], AbstractBilinearForm("p * eos(density)",Identity,Identity,eos_action; apply_action_to = 2))
    add_operator!(Problem, [3,3], AbstractBilinearForm("p*q",Identity,Identity,MultiplyScalarAction(-1,1); store = true))

    ## show Problem definition
    show(Problem)

    ## generate FESpaces and solution vector
    FESpaceV = FESpace{FETypes[1]}(xgrid)
    FESpacePD = FESpace{FETypes[2]}(xgrid)
    Solution = FEVector{Float64}("velocity",FESpaceV)
    append!(Solution,"density",FESpacePD)
    append!(Solution,"pressure",FESpacePD)

    ## initial values for density (bestapproximation or constant)
    if initial_density_bestapprox 
        L2DensityBestapproximationProblem = L2BestapproximationProblem(user_density; bestapprox_boundary_regions = [])
        InitialDensity = FEVector{Float64}("L2-Bestapproximation density",FESpacePD)
        solve!(InitialDensity, L2DensityBestapproximationProblem)
        Solution[2][:] = InitialDensity[1][:]
    else
        for j = 1 : FESpacePD.ndofs
            Solution[2][j] = M
        end
    end

    ## initial values for pressure obtained from equation of state
    equation_of_state!(c,gamma)(Solution[3],Solution[2])
    Minit = M * sum(Solution[2][:] .* xgrid[CellVolumes])

    ## generate time-dependent solver
    ## we have three equations [1] for velocity, [2] for density, [3] for pressure
    ## that are set to be iterated one after another via the subiterations argument
    ## only the density equation is made time-dependent via the timedependent_equations argument
    ## so we can reuse the other subiteration matrices in each timestep
    TCS = TimeControlSolver(Problem, Solution, BackwardEuler; subiterations = [[1],[2],[3]], maxlureuse = [-1,1,-1], timedependent_equations = [2], verbosity = verbosity)
    advance_until_stationarity!(TCS, timestep; maxTimeSteps = maxTimeSteps, stationarity_threshold = stationarity_threshold)

    ## compute error in mass constraint
    Md = sum(Solution[2][:] .* xgrid[CellVolumes])
    @printf("  mass_error = %.4e - %.4e = %.4e \n",Minit, Md, abs(Minit-Md))

    return Solution
end

end