
"""
````
function IncompressibleNavierStokesProblem(
    dimension::Int = 2;
    viscosity = 1.0,
    nonlinear::Bool = true,
    no_pressure_constraint::Bool = false,
    pmean = 0)
````

Creates a PDEDescription for the incompressible Navier-Stokes equations of the specified dimension and globally constant viscosity parameter.
    
If nonlinear == true a nonlinear convection term is added to the equation.
If no_pressure_constraint == false a GlobalConstraint for the pressure is added that fixes the pressure mean to the given value for pmean.

Boundary and right-hand side data or other modifications have to be added afterwards.
"""
function IncompressibleNavierStokesProblem(
    dimension::Int = 2;
    viscosity = 1.0,
    nonlinear::Bool = true,
    no_pressure_constraint::Bool = false,
    pmean = 0)

    # LEFT-HAND-SIDE: STOKES OPERATOR
    MyLHS = Array{Array{AbstractPDEOperator,1},2}(undef,2,2)
    #MyLHS[1,1] = [LaplaceOperator(DoNotChangeAction(4))]
    MyLHS[1,1] = [LaplaceOperator(viscosity,dimension,dimension)]
    MyLHS[1,2] = [LagrangeMultiplier(Divergence)] # automatically fills transposed block
    MyLHS[2,1] = []
    MyLHS[2,2] = []

    if nonlinear
        push!(MyLHS[1,1], ConvectionOperator(1, dimension, dimension))
    end

    # RIGHT-HAND SIDE: empty, user can fill in data later
    MyRHS = Array{Array{AbstractPDEOperator,1},1}(undef,2)
    MyRHS[1] = []
    MyRHS[2] = []

    # BOUNDARY OPERATOR: empty, user can fill in data later
    MyBoundaryVelocity = BoundaryOperator(dimension,dimension)
    MyBoundaryPressure = BoundaryOperator(dimension,1) # empty, no pressure boundary conditions
    
    # GLOBAL CONSTRAINTS: zero pressure integral mean
    MyGlobalConstraints = Array{AbstractGlobalConstraint,1}(undef,1)
    if no_pressure_constraint == false
        MyGlobalConstraints[1] = FixedIntegralMean(2,pmean)
    end

    if nonlinear == true
        name = "incompressible Navier-Stokes-Problem"
    else
        name = "incompressible Stokes-Problem"
    end

    return PDEDescription(name,MyLHS,MyRHS,[MyBoundaryVelocity,MyBoundaryPressure],MyGlobalConstraints)
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

The three equations/unknowns of the PDE refer to velocity, density and pressure in that order.
    
If nonlinear == true a nonlinear convection term is added to the equation.
    
Boundary and right-hand side data or other modifications have to be added afterwards.
"""
function CompressibleNavierStokesProblem(
    equation_of_state!::Function,
    gravity!::Function,
    dimension::Int = 2;
    viscosity = 1.0,
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

    # LEFT-HAND-SIDE: STOKES OPERATOR
    MyLHS = Array{Array{AbstractPDEOperator,1},2}(undef,3,3)
    MyLHS[1,1] = [LaplaceOperator(viscosity,dimension,dimension)]
    if lambda != 0
        push!(MyLHS[1,1],AbstractBilinearForm("lambda * grad(div(u)) (lambda = $lambda)",Divergence,Divergence,MultiplyScalarAction(lambda,1)))
    end
    MyLHS[1,2] = [AbstractBilinearForm("gravity*velocity*density",Identity,Identity,gravity_action)]
    MyLHS[1,3] = [AbstractBilinearForm(Divergence,Identity)]
    MyLHS[2,1] = []
    MyLHS[2,2] = [FVUpwindDivergenceOperator(1)]
    MyLHS[2,3] = []
    MyLHS[3,1] = []
    eos_action = FunctionAction(equation_of_state!,1,dimension)
    MyLHS[3,2] = [ReactionOperator(eos_action; apply_action_to = 2)]
    MyLHS[3,3] = [ReactionOperator(MultiplyScalarAction(-1.0,1))]

    if nonlinear
        push!(MyLHS[1,1], ConvectionOperator(1, dimension, dimension))
    end

    # RIGHT-HAND SIDE: empty, user can fill in data later
    MyRHS = Array{Array{AbstractPDEOperator,1},1}(undef,3)
    MyRHS[1] = []
    MyRHS[2] = []
    MyRHS[3] = []

    # BOUNDARY OPERATOR: empty, user can fill in data later
    MyBoundaryVelocity = BoundaryOperator(dimension,dimension)
    MyBoundaryDensity = BoundaryOperator(dimension,1) # empty, no density boundary conditions
    MyBoundaryPressure = BoundaryOperator(dimension,1) # empty, no pressure boundary conditions

    if nonlinear == true
        name = "compressible Navier-Stokes-Problem"
    else
        name = "compressible Stokes-Problem"
    end

    return PDEDescription(name,MyLHS,MyRHS,[MyBoundaryVelocity,MyBoundaryDensity,MyBoundaryPressure])
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

    # LEFT-HAND-SIDE: LINEAR ELASTICITY TENSOR
    MyLHS = Array{Array{AbstractPDEOperator,1},2}(undef,1,1)
    #MyLHS[1,1] = [LaplaceOperator(DoNotChangeAction(4))]
    if dimension == 2
        MyLHS[1,1] = [HookStiffnessOperator2D(shearmodulus,lambda)]
    elseif dimension == 1
        MyLHS[1,1] = [HookStiffnessOperator1D(elasticity_modulus)]
    end

    # RIGHT-HAND SIDE: empty, user can fill in data later
    MyRHS = Array{Array{AbstractPDEOperator,1},1}(undef,1)
    MyRHS[1] = []

    # BOUNDARY OPERATOR: empty, user can fill in data later
    MyBoundary = BoundaryOperator(dimension,dimension)

    name = "linear elasticity problem"

    return PDEDescription(name,MyLHS,MyRHS,[MyBoundary])
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

    # LEFT-HAND-SIDE: LAPLACE OPERATOR
    MyLHS = Array{Array{AbstractPDEOperator,1},2}(undef,1,1)
    #MyLHS[1,1] = [LaplaceOperator(DoNotChangeAction(4))]
    MyLHS[1,1] = [LaplaceOperator(diffusion,dimension,ncomponents)]

    # RIGHT-HAND SIDE: empty, user can fill in data later
    MyRHS = Array{Array{AbstractPDEOperator,1},1}(undef,1)
    MyRHS[1] = []

    # BOUNDARY OPERATOR: empty, user can fill in data later
    MyBoundary = BoundaryOperator(dimension,dimension)

    name = "Poisson problem"

    return PDEDescription(name,MyLHS,MyRHS,[MyBoundary])
end


"""
````
function L2BestapproximationProblem(
    exact_function::Function,
    dimension::Int = 2,
    ncomponents::Int = 1;
    bonus_quadorder::Int = 0,
    bestapprox_boundary_regions = [])
````

Creates an PDEDescription for an L2-Bestapproximation problem for the given exact function. Since this prototype already includes boundary and right-hand side data also a bonus quadrature order can be specified to steer the accuracy.
"""
function L2BestapproximationProblem(
    exact_function::Function,
    dimension::Int = 2,
    ncomponents::Int = 1;
    bonus_quadorder::Int = 0,
    bestapprox_boundary_regions = [])

    # LEFT-HAND-SIDE: REACTION OPERATOR
    MyLHS = Array{Array{AbstractPDEOperator,1},2}(undef,1,1)
    MyLHS[1,1] = [ReactionOperator(DoNotChangeAction(ncomponents))]

    # RIGHT-HAND-SIDE: L2-Product with exact_function
    MyRHS = Array{Array{AbstractPDEOperator,1},1}(undef,1)
    MyRHS[1] = [RhsOperator(Identity, [exact_function], dimension, ncomponents; bonus_quadorder = bonus_quadorder)]

    # BOUNDARY OPERATOR: 
    MyBoundary = BoundaryOperator(dimension,1) # leave it like that = no boundary conditions
    if length(bestapprox_boundary_regions) > 0
        append!(MyBoundary, bestapprox_boundary_regions, BestapproxDirichletBoundary; data = exact_function, bonus_quadorder = bonus_quadorder)
    end

    return PDEDescription("L2-Bestapproximation problem",MyLHS,MyRHS,[MyBoundary])
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
    exact_function_gradient::Function,
    exact_function_boundary::Function,
    dimension::Int = 2,
    ncomponents::Int = 1;
    bonus_quadorder::Int = 0,
    bonus_quadorder_boundary::Int = 0,
    bestapprox_boundary_regions = [])

    # LEFT-HAND-SIDE: LAPLACE OPERATOR
    MyLHS = Array{Array{AbstractPDEOperator,1},2}(undef,1,1)
    MyLHS[1,1] = [LaplaceOperator(1.0,dimension,ncomponents)]

    # RIGHT-HAND-SIDE: H1-Product with exact_function_gradient
    MyRHS = Array{Array{AbstractPDEOperator,1},1}(undef,1)
    MyRHS[1] = [RhsOperator(Gradient, [exact_function_gradient], dimension, ncomponents*dimension; bonus_quadorder = bonus_quadorder)]

    # BOUNDARY OPERATOR: 
    MyBoundary = BoundaryOperator(dimension,1) # leave it like that = no boundary conditions
    if length(bestapprox_boundary_regions) > 0
        append!(MyBoundary, bestapprox_boundary_regions, BestapproxDirichletBoundary; data = exact_function_boundary, bonus_quadorder = bonus_quadorder_boundary)
        return PDEDescription("H1-Bestapproximation problem",MyLHS,MyRHS,[MyBoundary])
    else
        # choose solution to have zero integral mean
        MyGlobalConstraints = Array{Array{AbstractGlobalConstraint,1},1}(undef,1)
        MyGlobalConstraints[1] = Array{AbstractGlobalConstraint,1}(undef,0)
        MyGlobalConstraints[1] = [FixedIntegralMean(0.0)]
        return PDEDescription("H1-Bestapproximation problem",MyLHS,MyRHS,[MyBoundary],MyGlobalConstraints)
    end

end

