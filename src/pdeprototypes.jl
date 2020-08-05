
"""
````
function IncompressibleNavierStokesProblem(
    dimension::Int = 2;
    viscosity = 1.0,
    nonlinear::Bool = true,
    nopressureconstraint::Bool = false,
    pmean = 0)
````

Creates a PDEDescription for the incompressible Navier-Stokes equations of the specified dimension and globally constant viscosity parameter.
"""
function IncompressibleNavierStokesProblem(
    dimension::Int = 2;
    viscosity = 1.0,
    nonlinear::Bool = true,
    no_pressure_constraint::Bool = false,
    pmean = 0)

    if nonlinear == true
        name = "incompressible Navier-Stokes-Problem"
    else
        name = "incompressible Stokes-Problem"
    end

    # generate empty PDEDescription for two unknowns
    # unknown 1 : velocity (vector-valued)
    # unknown 2 : pressure
    Problem = PDEDescription(name, 2, [dimension,1], dimension)

    # add Laplacian for velocity
    add_operator!(Problem, [1,1], LaplaceOperator(viscosity,dimension,dimension))

    # add Lagrange multiplier for divergence of velocity
    add_operator!(Problem, [1,2], LagrangeMultiplier(Divergence))
    
    if nonlinear
        add_operator!(Problem, [1,1], ConvectionOperator(1, Identity, dimension, dimension))
    end
    
    # zero pressure integral mean
    if no_pressure_constraint == false
        add_constraint!(Problem, FixedIntegralMean(2,pmean))
    end

    return Problem
end

"""
````
function CompressibleNavierStokesProblem(
    equation_of_state!::Function,
    gravity!::Function,
    dimension::Int = 2;
    viscosity = 1.0,
    lambda = 1.0,
    nonlinear::Bool = true)
````

Creates a PDEDescription for the compressible Navier-Stokes equations of the specified dimension, Lame parameters viscosity and lambda and the given equation of state function.
"""
function CompressibleNavierStokesProblem(
    equation_of_state!::Function,
    gravity!::Function,
    dimension::Int = 2;
    shear_modulus = 1.0,
    lambda = 1.0,
    nonlinear::Bool = true)


    function gravity_function() # result = G(v) = -gravity*input
        temp = zeros(Float64,dimension)
        function closure(result,input,x)
            gravity!(temp,x)
            result[1] = 0
            for j = 1 : dimension
                result[1] += temp[j]*input[j] 
            end
        end
    end    
    gravity_action = XFunctionAction(gravity_function(),1,dimension)

    if nonlinear == true
        name = "compressible Navier-Stokes-Problem"
    else
        name = "compressible Stokes-Problem"
    end

    # generate empty PDEDescription for three unknowns
    # unknown 1 : velocity (vector-valued)
    # unknown 2 : density
    # unknown 3 : pressure
    Problem = PDEDescription(name, 3, [dimension,1,1], dimension)

    # momentum equation
    add_operator!(Problem, [1,1], LaplaceOperator(shear_modulus,dimension,dimension))
    if lambda != 0
        add_operator!(Problem, [1,1], AbstractBilinearForm("lambda * grad(div(u)) (lambda = $lambda)",Divergence,Divergence,MultiplyScalarAction(lambda,1)))
    end
    if nonlinear
        add_operator!(Problem, [1,1], ConvectionOperator(1, Identity, dimension, dimension))
    end

    add_operator!(Problem, [1,2], AbstractBilinearForm("gravity*velocity*density",Identity,Identity,gravity_action))
    add_operator!(Problem, [1,3], AbstractBilinearForm(Divergence,Identity))

    # continuity equation
    add_operator!(Problem, [2,2], FVUpwindDivergenceOperator(1))

    # equation of state
    eos_action = FunctionAction(equation_of_state!,1,dimension)
    add_operator!(Problem, [3,2], ReactionOperator(eos_action; apply_action_to = 2))
    add_operator!(Problem, [3,3], ReactionOperator(MultiplyScalarAction(-1.0,1)))

    return Problem
end

"""
````
function LinearElasticityProblem(
    dimension::Int = 2;
    elasticity_modulus = 1.0,
    shearmodulus = 1.0,
    lambda = 1.0)
````

Creates a PDEDescription for the linear elasticity problem of the specified dimension.

If dimension == 1, only the elasticity_modulus is used as a parameter in the Hookian stiffness operator.
If dimension == 2, shear_modulus and lambda are used as Lame parameters in the Hookian stiffness operator.
    
Boundary and right-hand side data or other modifications have to be added afterwards.
"""
function LinearElasticityProblem(
    dimension::Int = 2;
    elasticity_modulus = 1.0,
    shearmodulus = 1.0,
    lambda = 1.0)

    # generate empty PDEDescription for one unknown
    # unknown 1 : displacement (vector-valued)
    Problem = PDEDescription("linear elasticity problem", 1, [dimension], dimension)
    if dimension == 2
        add_operator!(Problem, [1,1], HookStiffnessOperator2D(shearmodulus,lambda))
    elseif dimension == 1
        add_operator!(Problem, [1,1], HookStiffnessOperator1D(elasticity_modulus))
    end

    return Problem
end


"""
````
function PoissonProblem(
    dimension::Int = 2;
    ncomponents::Int = 1,
    diffusion = 1.0)
````

Creates a PDEDescription for a Poisson problem with specified number of components and globally constant diffusion parameter.
    
Boundary and right-hand side data or other modifications have to be added afterwards.
"""
function PoissonProblem(
    dimension::Int = 2;
    ncomponents::Int = 1,
    diffusion = 1.0)

    # generate empty PDEDescription for one unknown
    Problem = PDEDescription("Poisson problem", 1, [ncomponents], dimension)
    add_operator!(Problem, [1,1], LaplaceOperator(diffusion,dimension,ncomponents))

    return Problem
end


"""
````
function L2BestapproximationProblem(
    uexact::Function,
    dimension::Int = 2,
    ncomponents::Int = 1;
    bonus_quadorder::Int = 0,
    bestapprox_boundary_regions = [])
````

Creates an PDEDescription for an L2-Bestapproximation problem for the given exact function. Since this prototype already includes boundary and right-hand side data also a bonus quadrature order can be specified to steer the accuracy.
"""
function L2BestapproximationProblem(
    uexact::Function,
    dimension::Int = 2,
    ncomponents::Int = 1;
    bonus_quadorder::Int = 0,
    bestapprox_boundary_regions = [],
    time_dependent_data::Bool = false,
    time = 0)

    # generate empty PDEDescription for one unknown
    Problem = PDEDescription("L2-Bestapproximation problem", 1, [ncomponents], dimension)
    add_operator!(Problem, [1,1], ReactionOperator(DoNotChangeAction(ncomponents)))
    data(result,x) = (time_dependent_data) ? uexact(result,x,time) : uexact(result,x)
    add_rhsdata!(Problem, 1, RhsOperator(Identity, [0], data, dimension, ncomponents; bonus_quadorder = bonus_quadorder))
    if length(bestapprox_boundary_regions) > 0
        if dimension == 1 # in 1D Dirichlet boundary can be interpolated
            add_boundarydata!(Problem, 1, bestapprox_boundary_regions, InterpolateDirichletBoundary; data = data, bonus_quadorder = bonus_quadorder)
        else
            add_boundarydata!(Problem, 1, bestapprox_boundary_regions, BestapproxDirichletBoundary; data = data, bonus_quadorder = bonus_quadorder)
        end 
    end

    return Problem
end


"""
````
function H1BestapproximationProblem(
    exact_function_gradient::Function,
    exact_function_boundary::Function,
    dimension::Int = 2,
    ncomponents::Int = 1;
    bonus_quadorder::Int = 0,
    bonus_quadorder_boundary::Int = 0,
    bestapprox_boundary_regions = [])
````

Creates an PDEDescription for an H1-Bestapproximation problem for the given exact function (only used on the boundary) and its exact gradient (used in the right-hand side). Since this prototype already includes boundary and right-hand side data also a bonus quadrature order can be specified to steer the accuracy.
"""
function H1BestapproximationProblem(
    uexact_gradient::Function,
    uexact::Function,
    dimension::Int = 2,
    ncomponents::Int = 1;
    bonus_quadorder::Int = 0,
    bonus_quadorder_boundary::Int = 0,
    bestapprox_boundary_regions = [],
    time_dependent_data::Bool = false,
    time = 0)

    # generate empty PDEDescription for one unknown
    Problem = PDEDescription("H1-Bestapproximation problem", 1, [ncomponents], dimension)
    add_operator!(Problem, [1,1], LaplaceOperator(1.0,dimension,ncomponents))
    data(result,x) = (time_dependent_data) ? uexact(result,x,time) : uexact(result,x)
    data_gradient(result,x) = (time_dependent_data) ? uexact_gradient(result,x,time) : uexact_gradient(result,x)
    add_rhsdata!(Problem, 1, RhsOperator(Gradient, [0], data_gradient, dimension, ncomponents*dimension; bonus_quadorder = bonus_quadorder))
    if length(bestapprox_boundary_regions) > 0
        if dimension == 1 # in 1D Dirichlet boundary can be interpolated
            add_boundarydata!(Problem, 1, bestapprox_boundary_regions, InterpolateDirichletBoundary; data = data, bonus_quadorder = bonus_quadorder_boundary)
        else
            add_boundarydata!(Problem, 1, bestapprox_boundary_regions, BestapproxDirichletBoundary; data = data, bonus_quadorder = bonus_quadorder_boundary)
        end
    else
        add_constraint!(Problem, FixedIntegralMean(0.0))
    end

    return Problem
end

