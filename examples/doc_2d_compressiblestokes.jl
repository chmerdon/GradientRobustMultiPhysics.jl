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
This example is designed to study the well-balanced property of a discretisations. Note that gradient-robust discretisations (set reconstruct = true below)
have a much smaller L2 velocity error (i.e. approximate the well-balanced state much better). For larger c the problem gets more incompressible which reduces
the error further as then the right-hand side is a perfect gradient also when evaluated with the (now closer to a constant) discrete density.
Also, on a uniform mesh the gradient-robust method is perfect!

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
function exact_density!(c)
    function closure(result,x)
        result[1] = 1.0 - (x[2] - 0.5)/c
    end
end 

## the exact velocity (zero!)
function exact_velocity!(result,x)
    result[1] = 0.0
    result[2] = 0.0
end 

## gravity right-hand side (just gravity but with opposite sign!)
function rhs_gravity!(gamma,c)
    function closure(result,x)
        result[1] = 1.0 - (x[2] - 0.5)/c # = density
        result[2] = - result[1]^(gamma-2) * gamma
        result[1] = 0.0
    end
end   

## everything is wrapped in a main function
function main(; verbosity = 1, Plotter = nothing, reconstruct::Bool = true)

    ## generate mesh
    #xgrid = uniform_refine(grid_unitsquare(Triangle2D),4); # uncomment this line for a structured grid
    xgrid = uniform_refine(grid_unitsquare_mixedgeometries(),4); # unstructured mesh

    ## problem data
    c = 10 # coefficient in equation of state
    gamma = 1.4 # power in gamma law in equations of state
    M = 1  # mass constraint for density
    shear_modulus = 1
    lambda = - 1//3 * shear_modulus

    ## choose finite element type [velocity, density,  pressure]
    FETypes = [H1BR{2}, L2P0{1}, L2P0{1}] # Bernardi--Raugel
    #FETypes = [H1CR{2}, L2P0{1}, L2P0{1}] # Crouzeix--Raviart (possibly needs smaller timesteps)

    ## solver parameters
    timestep = 2 * shear_modulus / (2*c)
    initial_density_bestapprox = true # otherwise we start with a constant density which also works but takes longer
    maxIterations = 300  # termination criterion 1
    target_change = 1e-10 # stopf when change is below this treshold

    #####################################################################################    
    #####################################################################################

    ## load compressible Stokes problem prototype and assign boundary data
    CStokesProblem = CompressibleNavierStokesProblem(equation_of_state!(c,gamma), rhs_gravity!(gamma,c), 2; shear_modulus = shear_modulus, lambda = lambda, nonlinear = false)
    add_boundarydata!(CStokesProblem, 1,  [1,2,3,4], HomogeneousDirichletBoundary)

    ## error intergrators for velocity and density
    L2VelocityErrorEvaluator = L2ErrorIntegrator(exact_velocity!, Identity, 2, 2; bonus_quadorder = 0)
    L2DensityErrorEvaluator = L2ErrorIntegrator(exact_density!(c), Identity, 2, 1; bonus_quadorder = 1)

    ## modify testfunction in operators
    if reconstruct
        TestFunctionOperatorIdentity = ReconstructionIdentity{HDIVRT0{2}} # identity operator for gradient-robust scheme
        TestFunctionOperatorDivergence = ReconstructionDivergence{HDIVRT0{2}} # divergence operator for gradient-robust scheme
    else # classical choices
        TestFunctionOperatorIdentity = Identity
        TestFunctionOperatorDivergence = Divergence
    end
    CStokesProblem.LHSOperators[1,1][2].operator1 = TestFunctionOperatorDivergence
    CStokesProblem.LHSOperators[1,1][2].operator2 = TestFunctionOperatorDivergence
    CStokesProblem.LHSOperators[1,2][1].operator1 = TestFunctionOperatorIdentity

    ## store matrix of velo-pressure and velo-gravity operator
    ## so that only a matrix-vector multiplication is needed in every iteration
    CStokesProblem.LHSOperators[1,2][1].store_operator = true
    CStokesProblem.LHSOperators[1,3][1].store_operator = true

    ## generate FESpaces and solution vector
    FESpaceV = FESpace{FETypes[1]}(xgrid)
    FESpacePD = FESpace{FETypes[2]}(xgrid)
    Solution = FEVector{Float64}("velocity",FESpaceV)
    append!(Solution,"density",FESpacePD)
    append!(Solution,"pressure",FESpacePD)

    ## initial values for density (bestapproximation or constant)
    if initial_density_bestapprox 
        L2DensityBestapproximationProblem = L2BestapproximationProblem(exact_density!(c), 2, 1; bestapprox_boundary_regions = [], bonus_quadorder = 2)
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

    ## generate time-dependent solver
    ## we have three equations [1] for velocity, [2] for density, [3] for pressure
    ## that are set to be iterated one after another via the subiterations argument
    ## only the density equation is made time-dependent via the timedependent_equations argument
    ## so we can reuse the other subiteration matrices in each timestep
    TCS = TimeControlSolver(CStokesProblem, Solution, BackwardEuler; subiterations = [[1],[2],[3]], maxlureuse = [-1,1,-1], timedependent_equations = [1,2], verbosity = verbosity)

    ## loop in pseudo-time until stationarity detected
    ## we also output M to see that the mass constraint is preserved all the way
    for iteration = 1 : maxIterations
        statistics = advance!(TCS, timestep)
        M = sum(Solution[2][:] .* xgrid[CellVolumes])
        @printf("  iteration %4d",iteration)
        @printf("  time = %.4e",TCS.ctime)
        @printf("  M = %.4e \n",M)
        ## first row of statistics is the linear residual for each equation
        @printf("      linresidual = [%.4e,%.4e,%.4e]\n",statistics[1,1],statistics[2,1],statistics[3,1])
        ## second row of statistics is the change in each unknown
        @printf("      change = [%.4e,%.4e,%.4e]\n",statistics[1,2],statistics[2,2],statistics[3,2])
        if sum(statistics[:,2]) < target_change
            println("  terminated (change below tolerance)")
            break;
        end
    end

    ## compute L2 error for velocity and density
    L2error = sqrt(evaluate(L2VelocityErrorEvaluator,Solution[1]))
    println("\nL2error(Velocity) = $L2error")
    L2error = sqrt(evaluate(L2DensityErrorEvaluator,Solution[2]))
    println("L2error(Density) = $L2error")
    
    ## plots
    GradientRobustMultiPhysics.plot(Solution, [1,2,3], [Identity, Identity, Identity]; Plotter = Plotter, verbosity = verbosity, use_subplots = true)
    
end

end