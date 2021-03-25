
"""
````
function IncompressibleNavierStokesProblem(
    dimension::Int = 2;
    viscosity = 1.0,
    nonlinear::Bool = false,
    auto_newton::Bool = false,
    nopressureconstraint::Bool = false,
    pmean = 0)
````

Creates a PDEDescription for the incompressible (Navier-)Stokes equations of the specified dimension and globally constant viscosity parameter.
If nonlinear = true the nonlinear convection term is added to the PDEDescription. If also auto_newton = true, a Newton iteration is devised
by automatic differentiation of the convection term.
"""
function IncompressibleNavierStokesProblem(
    dimension::Int = 2;
    viscosity = 1.0,
    nonlinear::Bool = true,
    auto_newton::Bool = true, # Newton by AD
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
    Problem = PDEDescription(name)
    add_unknown!(Problem; equation_name = "momentum equation", unknown_name = "velocity")
    add_unknown!(Problem; equation_name = "incompressibility constraint", unknown_name = "pressure")

    # add Laplacian for velocity
    add_operator!(Problem, [1,1], LaplaceOperator(viscosity,dimension,dimension; store = true))

    # add Lagrange multiplier for divergence of velocity
    add_operator!(Problem, [1,2], LagrangeMultiplier(Divergence))
    
    if nonlinear
        add_operator!(Problem, [1,1], ConvectionOperator(1, Identity, dimension, dimension; auto_newton = auto_newton))
    end
    
    # zero pressure integral mean
    if no_pressure_constraint == false
        add_constraint!(Problem, FixedIntegralMean(2,pmean))
    end

    return Problem
end


"""
````
function LinearElasticityProblem(
    dimension::Int = 2;
    elasticity_modulus = 1.0,
    shear_modulus = 1.0,
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
    shear_modulus = 1.0,
    lambda = 1.0)

    # generate empty PDEDescription for one unknown
    # unknown 1 : displacement (vector-valued)
    Problem = PDEDescription("linear elasticity problem")
    add_unknown!(Problem; unknown_name = "displacement", equation_name = "displacement equation")
    if dimension == 3
        add_operator!(Problem, [1,1], HookStiffnessOperator3D(shear_modulus,lambda))
    elseif dimension == 2
        add_operator!(Problem, [1,1], HookStiffnessOperator2D(shear_modulus,lambda))
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
    diffusion = 1.0,
    AT::Type{<:AbstractAssemblyType} = ON_CELLS)

    # generate empty PDEDescription for one unknown
    Problem = PDEDescription("Poisson problem")
    add_unknown!(Problem; unknown_name = "u", equation_name = "Poisson equation")
    add_operator!(Problem, [1,1], LaplaceOperator(diffusion,dimension,ncomponents; AT = AT))

    return Problem
end


"""
````
function L2BestapproximationProblem(
    uexact::UserData{AbstractDataFunction},
    dimension::Int = 2,
    ncomponents::Int = 1;
    bonus_quadorder::Int = 0,
    bestapprox_boundary_regions = [])
````

Creates an PDEDescription for an L2-Bestapproximation problem for the given exact function. Since this prototype already includes boundary and right-hand side data also a bonus quadrature order can be specified to steer the accuracy.
"""
function L2BestapproximationProblem(
    uexact::UserData{AbstractDataFunction};
    name = "L2-Bestapproximation problem",
    unknown_name = "L2-bestapproximation",
    equation_name = "L2-bestapproximation equation",
    bestapprox_boundary_regions = [],
    AT::Type{<:AbstractAssemblyType} = ON_CELLS)

    ncomponents = uexact.dimensions[1]
    xdim = uexact.dimensions[2]
    # generate empty PDEDescription for one unknown
    Problem = PDEDescription(name)
    add_unknown!(Problem; unknown_name = unknown_name, equation_name = equation_name)
    add_operator!(Problem, [1,1], ReactionOperator(DoNotChangeAction(ncomponents); AT = AT))
    add_rhsdata!(Problem, 1, RhsOperator(Identity, [0], uexact; AT = AT))
    if length(bestapprox_boundary_regions) > 0
        if xdim == 1 # in 1D Dirichlet boundary can be interpolated
            add_boundarydata!(Problem, 1, bestapprox_boundary_regions, InterpolateDirichletBoundary; data = uexact)
        else
            add_boundarydata!(Problem, 1, bestapprox_boundary_regions, BestapproxDirichletBoundary; data = uexact)
        end 
    end

    return Problem
end


"""
````
function H1BestapproximationProblem(
    exact_function_gradient::UserData{AbstractDataFunction},
    exact_function_boundary::UserData{AbstractDataFunction};
    bonus_quadorder::Int = 0,
    bonus_quadorder_boundary::Int = 0,
    bestapprox_boundary_regions = [])
````

Creates an PDEDescription for an H1-Bestapproximation problem for the given exact function (only used on the boundary) and its exact gradient (used in the right-hand side). Since this prototype already includes boundary and right-hand side data also a bonus quadrature order can be specified to steer the accuracy.
"""
function H1BestapproximationProblem(
    uexact_gradient::UserData{AbstractDataFunction},
    uexact::UserData{AbstractDataFunction};
    name = "H1-Bestapproximation problem",
    unknown_name = "H1-bestapproximation",
    equation_name = "H1-bestapproximation equation",
    bestapprox_boundary_regions = [])

    ncomponents = uexact.dimensions[1]
    xdim = Int(uexact_gradient.dimensions[1] / ncomponents)

    # generate empty PDEDescription for one unknown
    Problem = PDEDescription(name)
    add_unknown!(Problem; unknown_name = unknown_name, equation_name = equation_name)
    add_operator!(Problem, [1,1], LaplaceOperator(1.0,xdim,ncomponents))
    add_rhsdata!(Problem, 1, RhsOperator(Gradient, [0], uexact_gradient))
    if length(bestapprox_boundary_regions) > 0
        if xdim == 1 # in 1D Dirichlet boundary can be interpolated
            add_boundarydata!(Problem, 1, bestapprox_boundary_regions, InterpolateDirichletBoundary; data = uexact)
        else
            add_boundarydata!(Problem, 1, bestapprox_boundary_regions, BestapproxDirichletBoundary; data = uexact)
        end
    else
        add_constraint!(Problem, FixedIntegralMean(0.0))
    end

    return Problem
end

