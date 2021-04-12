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

## gravity right-hand side (just gravity but with opposite sign!)
function rhs_gravity!(gamma,c)
    function closure(result,x::Array{<:Real,1})
        result[1] = 1.0 - (x[2] - 0.5)/c # = density
        result[2] = - result[1]^(gamma-2) * gamma
        result[1] = 0.0
    end
end   

## everything is wrapped in a main function
function main(; verbosity = 0, Plotter = nothing, reconstruct::Bool = true, c = 10, gamma = 1.4, M = 1, shear_modulus = 1e-3, lambda = -1e-3/3)

    ## set log level
    set_verbosity(verbosity)

    ## load mesh and refine
    xgrid = uniform_refine(simplexgrid("assets/2d_grid_mountainrange.sg"),1)

    ## solve without and with reconstruction and plot
    Solution = setup_and_solve(xgrid; reconstruct = false, c = c, M = M, lambda = lambda, shear_modulus = shear_modulus, gamma = gamma)
    Solution2 = setup_and_solve(xgrid; reconstruct = true, c = c, M = M, lambda = lambda, shear_modulus = shear_modulus, gamma = gamma)

    # plot everything
    GradientRobustMultiPhysics.plot(xgrid, [Solution[1],Solution[2],Solution2[1],Solution2[2]], [Identity, Identity, Identity, Identity]; add_grid_plot = true, Plotter = Plotter)

    ## compare L2 error for velocity and density
    user_velocity = DataFunction([0,0]; name = "u")
    user_density = DataFunction(exact_density!(M,c), [1,2]; name = "ϱ", dependencies = "X", quadorder = 1)
    L2VelocityErrorEvaluator = L2ErrorIntegrator(Float64, user_velocity, Identity)
    L2DensityErrorEvaluator = L2ErrorIntegrator(Float64, user_density, Identity)
    L2error = sqrt(evaluate(L2VelocityErrorEvaluator,Solution[1]))
    L2error2 = sqrt(evaluate(L2VelocityErrorEvaluator,Solution2[1]))
    L2error3 = sqrt(evaluate(L2DensityErrorEvaluator,Solution[2]))
    L2error4 = sqrt(evaluate(L2DensityErrorEvaluator,Solution2[2]))
    @printf("\n        reconstruct     false    |    true\n")
    @printf("================================================\n")
    @printf("L2error(Velocity) | %.5e  | %.5e \n",L2error,L2error2)
    @printf("L2error(Density)  | %.5e  | %.5e \n",L2error3,L2error4)
    
end

function setup_and_solve(xgrid; reconstruct = true, c = 1, gamma = 1, M = 1, shear_modulus = 1, lambda = 0, verbosity = 0)

    ## negotiate edata functions to the package
    user_density = DataFunction(exact_density!(M,c), [1,2]; name = "ϱ", dependencies = "X", quadorder = 1)
    user_gravity = DataFunction(rhs_gravity!(gamma,c), [2,2]; name = "g", dependencies = "X", quadorder = 1)

    ## solver parameters
    timestep = 2 * shear_modulus / c 
    initial_density_bestapprox = true # otherwise we start with a constant density which also works but takes longer
    maxTimeSteps = 1000  # termination criterion 1
    stationarity_threshold = 1e-12/shear_modulus # stop when change is below this treshold

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
    ## unknown 1 : velocity v (vector-valued)
    ## unknown 2 : density ϱ
    ## unknown 3 : pressure p
    Problem = PDEDescription("compressible Stokes problem")
    add_unknown!(Problem; unknown_name = "v", equation_name = "momentum equation")
    add_unknown!(Problem; unknown_name = "ϱ", equation_name = "continuity equation")
    add_unknown!(Problem; unknown_name = "p", equation_name = "equation of state")
    add_boundarydata!(Problem, 1,  [1,2,3,4], HomogeneousDirichletBoundary)

    ## momentum equation
    add_operator!(Problem, [1,1], LaplaceOperator(2*shear_modulus; store = true))
    if lambda != 0
        add_operator!(Problem, [1,1], AbstractBilinearForm([VeloDivergence,VeloDivergence]; name = "λ (div(u),div(v))", factor = lambda, store = true))
    end
    add_operator!(Problem, [1,3], AbstractBilinearForm([Divergence,Identity]; name = "(div(v),p)", factor = -1, store = true))

    function gravity_action()
        temp = zeros(Float64,2)
        function closure(result,input,x, t)
            eval!(temp, user_gravity, x, t)
            result[1] = - temp[1] * input[1] - temp[2] * input[2]
        end
        gravity_action = Action(Float64, closure, [1,2]; name = "gravity action", dependencies = "XT", quadorder = user_gravity.quadorder)
    end    
    add_operator!(Problem, [1,2], AbstractBilinearForm([VeloIdentity,Identity],gravity_action(); name = "ϱ(g ⋅ v)", store = true))

    ## continuity equation (by FV upwind on triangles)
    add_operator!(Problem, [2,2], FVConvectionDiffusionOperator(1))

    ## equation of state (by best-approximation, P0 mass matrix is diagonal)
    eos_action_kernel = ActionKernel(equation_of_state!(c,gamma),[1,1]; dependencies = "", quadorder = 0)
    eos_action = Action(Float64, eos_action_kernel)
    add_operator!(Problem, [3,2], AbstractBilinearForm([Identity,Identity],eos_action; name = "(p,eos(ϱ))", apply_action_to = [2]))
    add_operator!(Problem, [3,3], AbstractBilinearForm([Identity,Identity]; name = "(p,q)", factor = -1, store = true))

    ## generate FESpaces and solution vector
    FES = [FESpace{FETypes[1]}(xgrid), FESpace{FETypes[2]}(xgrid), FESpace{FETypes[3]}(xgrid)]
    Solution = FEVector{Float64}(["v_h (reconst=$reconstruct)", "ϱ_h (reconst=$reconstruct)", "p_h (reconst=$reconstruct)"],FES)

    ## initial values for density (bestapproximation or constant)
    if initial_density_bestapprox 
        L2DensityBestapproximationProblem = L2BestapproximationProblem(user_density; bestapprox_boundary_regions = [])
        InitialDensity = FEVector{Float64}("ϱ_best",FES[2])
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
    TCS = TimeControlSolver(Problem, Solution, BackwardEuler;
                                        subiterations = [[1],[2],[3]],
                                        skip_update = [-1,1,-1],
                                        timedependent_equations = [2],
                                        maxiterations = 1,
                                        show_solver_config = true,
                                        check_nonlinear_residual = false)
    advance_until_stationarity!(TCS, timestep; maxTimeSteps = maxTimeSteps, stationarity_threshold = stationarity_threshold)

    ## compute error in mass constraint
    Md = sum(Solution[2][:] .* xgrid[CellVolumes])
    @printf("  mass_error = %.4e - %.4e = %.4e \n",Minit, Md, abs(Minit-Md))

    return Solution
end

end