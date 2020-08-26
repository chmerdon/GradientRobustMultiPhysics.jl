push!(LOAD_PATH, "../src")
using GradientRobustMultiPhysics
using ExtendableGrids
using ForwardDiff
ENV["MPLBACKEND"]="qt5agg"
using PyPlot
using Printf


function getProblemData(nu::Real = 1.0, lambda::Real = 0.0; use_nonlinear_convection::Bool = false, use_gravity::Bool = true, symmetric_gradient::Bool = false, c::Real = 1.0, gamma::Real = 1.0, density_power::Real = 0.0, total_mass::Real = 1.0, nrBoundaryRegions::Int = 4)


    function exact_density!(result, x) # exact density
        result[1] = total_mass - (x[2]^density_power - 1.0/(density_power+1))/c
    end
    function exact_density(x) # exact density
        return total_mass - (x[2]^density_power - 1.0/(density_power+1))/c
    end


    function equation_of_state!(pressure,density)
        for j=1:length(density)
            pressure[j] = c*density[j]^gamma
        end    
    end    

    function exact_streamfunction(x)
        return (x[1]^2 * (x[1] - 1)^2 * x[2]^2 * (x[2] - 1)^2)
    end


    rho = zeros(Real,1)
    function exact_pressure(x) # exact pressure
        exact_density!(rho, x)
        equation_of_state!(rho,rho)
        return rho[1]
    end

    function exact_pressure!(result, x) # exact pressure
        exact_density!(result, x)
        equation_of_state!(result,result)
    end

    # exact velocity by ForwardDiff
    thetagrad = DiffResults.GradientResult([0.0,0.0]);
    function exact_velocity!(result, x)
        ForwardDiff.gradient!(thetagrad,exact_streamfunction,x);
        result[1] = -DiffResults.gradient(thetagrad)[2]/exact_density(x)
        result[2] = DiffResults.gradient(thetagrad)[1]/exact_density(x)
    end    
    
    # volume_data by ForwardDiff
    hessian = [0.0 0.0;0.0 0.0]
    p(x) = exact_pressure(x)
    grad = DiffResults.GradientResult([0.0,0.0]);
    hessian = DiffResults.HessianResult([0.0,0.0]);
    velo_rotated(a) = ForwardDiff.gradient(exact_streamfunction,a);
    velo1 = x -> -velo_rotated(x)[2]/exact_density(x)
    velo2 = x -> velo_rotated(x)[1]/exact_density(x)
    gravity = [0.0,0.0]#-1.0]
    function volume_data!(result,x)
        fill!(result,0.0)

        # add friction term
        ForwardDiff.hessian!(hessian,velo1,x)
        if symmetric_gradient == false
            result[1] -= 2*nu * (DiffResults.hessian(hessian)[1] + DiffResults.hessian(hessian)[4])
        else
            result[1] -= 2*nu * DiffResults.hessian(hessian)[1] + nu * DiffResults.hessian(hessian)[4]
            result[2] -= nu * DiffResults.hessian(hessian)[3]
        end
        result[1] -= lambda * DiffResults.hessian(hessian)[1]
        result[2] -= lambda * DiffResults.hessian(hessian)[3]
        
        ForwardDiff.hessian!(hessian,velo2,x)
        if symmetric_gradient == false
            result[2] -= 2*nu * (DiffResults.hessian(hessian)[1] + DiffResults.hessian(hessian)[4])
        else
            result[2] -= 2*nu * DiffResults.hessian(hessian)[4] + nu * DiffResults.hessian(hessian)[1]
            result[1] -= nu * DiffResults.hessian(hessian)[3]
        end
        result[1] -= lambda * DiffResults.hessian(hessian)[2]
        result[2] -= lambda * DiffResults.hessian(hessian)[4]

        #
        
        # add gradient of pressure
        ForwardDiff.gradient!(grad,p,x);
        result[1] += DiffResults.gradient(grad)[1]
        result[2] += DiffResults.gradient(grad)[2]

        # remove gravity effect
        if use_gravity
            exact_density!(rho,x)
            result[1] -= rho[1] * gravity[1]
            result[2] -= rho[1] * gravity[2]
        end

        # add rho*(u*grad)u term
        if use_nonlinear_convection
            exact_density!(rho,x)
            ForwardDiff.gradient!(grad,velo1,x);
            result[1] += velo1(x) * DiffResults.gradient(grad)[1]
            result[1] += velo2(x) * DiffResults.gradient(grad)[2]
            ForwardDiff.gradient!(grad,velo2,x);
            result[2] += velo1(x) * DiffResults.gradient(grad)[1]
            result[2] += velo2(x) * DiffResults.gradient(grad)[2]
        end
    end

    ## gravity right-hand side 
    function gravity!(gamma,c)
        function closure(result,x)
            result[1] = gravity[1]
            result[2] = gravity[2]
        end
    end   

    return equation_of_state!, volume_data!, exact_velocity!, exact_density!, exact_pressure!, gravity!
end

## everything is wrapped in a main function
function main()

    #####################################################################################
    #####################################################################################

    ## generate mesh
    #xgrid = grid_unitsquare(Triangle2D); # uncomment this line for a structured grid
    xgrid = grid_unitsquare_mixedgeometries(); # not so structured
    xgrid = split_grid_into(xgrid,Triangle2D)
    xgrid = uniform_refine(xgrid,3)

    ## problem data
    c = 1 # coefficient in equation of state
    gamma = 1 # power in gamma law in equations of state
    M = 1  # mass constraint for density
    shear_modulus = 1
    density_power = 1
    lambda = - 1//3 * shear_modulus

    ## choose finite element type [velocity, density,  pressure]
    FETypes = [H1BR{2}, L2P0{1}, L2P0{1}] # Bernardi--Raugel
    #FETypes = [H1CR{2}, L2P0{1}, L2P0{1}] # Crouzeix--Raviart (possibly needs smaller timesteps)

    ## solver parameters
    timestep = shear_modulus / (M*2*c)
    reconstruct = true # use pressure-robust discretisation of the finite element method?
    initial_bestapprox = true # otherwise we start with a constant density which also works but takes longer
    maxIterations = 1000  # termination criterion 1
    maxResidual = 1e-10 # termination criterion 2

    ## postprocess parameters
    plot_grid = false
    plot_pressure = false
    plot_density = true
    plot_velocity = true

    #####################################################################################    
    #####################################################################################

    ## load compressible Stokes problem prototype and assign boundary data

    equation_of_state!, volume_data!, exact_velocity!, exact_density!, exact_pressure!, rhs_gravity! = getProblemData(shear_modulus, lambda; use_gravity = true, symmetric_gradient = false, c = c, gamma = gamma, density_power = density_power, total_mass = M)

    CStokesProblem = CompressibleNavierStokesProblem(equation_of_state!, rhs_gravity!, 2; shear_modulus = shear_modulus, lambda = lambda, nonlinear = false)
    add_boundarydata!(CStokesProblem, 1,  [1,2,3,4], HomogeneousDirichletBoundary)
    add_rhsdata!(CStokesProblem, 1,  RhsOperator(Identity, [0], volume_data!, 2, 2; bonus_quadorder = 4))

    ## error intergrators for velocity and density
    L2VelocityErrorEvaluator = L2ErrorIntegrator(exact_velocity!, Identity, 2, 2; bonus_quadorder = 3)
    L2DensityErrorEvaluator = L2ErrorIntegrator(exact_density!, Identity, 2, 1; bonus_quadorder = 1)

    ## modify testfunction in operators
    if reconstruct
        TestFunctionOperatorIdentity = ReconstructionIdentity{HDIVRT0{2}} # identity operator for gradient-robust scheme
        TestFunctionOperatorDivergence = ReconstructionDivergence{HDIVRT0{2}} # divergence operator for gradient-robust scheme
    else # classical choices
        TestFunctionOperatorIdentity = Identity
        TestFunctionOperatorDivergence = Divergence
    end
    if abs(lambda) > 0
        CStokesProblem.LHSOperators[1,1][2].operator1 = TestFunctionOperatorDivergence
        CStokesProblem.LHSOperators[1,1][2].operator2 = TestFunctionOperatorDivergence
    end
    CStokesProblem.LHSOperators[1,2][1].operator1 = TestFunctionOperatorIdentity

    ## store matrix of velo-pressure and velo-gravity operator
    ## so that only a matrix-vector multiplication is needed in every iteration
    CStokesProblem.RHSOperators[1][1].store_operator = true
    CStokesProblem.LHSOperators[1,2][1].store_operator = true
    CStokesProblem.LHSOperators[1,3][1].store_operator = true

    ## generate FESpaces and solution vector
    FESpaceV = FESpace{FETypes[1]}(xgrid)
    FESpacePD = FESpace{FETypes[2]}(xgrid)
    Solution = FEVector{Float64}("velocity",FESpaceV)
    append!(Solution,"density",FESpacePD)
    append!(Solution,"pressure",FESpacePD)

    ## initial values for density (bestapproximation or constant)
    if initial_bestapprox 
        L2VelocityBestapproximationProblem = L2BestapproximationProblem(exact_velocity!, 2, 2; bestapprox_boundary_regions = [], bonus_quadorder = 3)
        InitialVelocity = FEVector{Float64}("L2-Bestapproximation velocity",FESpaceV)
        solve!(InitialVelocity, L2VelocityBestapproximationProblem)
        Solution[1][:] = InitialVelocity[1][:]

        L2DensityBestapproximationProblem = L2BestapproximationProblem(exact_density!, 2, 1; bestapprox_boundary_regions = [], bonus_quadorder = 1)
        InitialDensity = FEVector{Float64}("L2-Bestapproximation density",FESpacePD)
        solve!(InitialDensity, L2DensityBestapproximationProblem)
        Solution[2][:] = InitialDensity[1][:]
    else
        for j = 1 : FESpacePD.ndofs
            Solution[2][j] = M
        end
    end

    ## initial values for pressure obtained from equation of state
    equation_of_state!(Solution[3],Solution[2])

    ## generate time-dependent solver
    ## we have three equations [1] for velocity, [2] for density, [3] for pressure
    ## that are set to be iterated one after another via the subiterations argument
    ## only the density equation is made time-dependent via the timedependent_equations argument
    TCS = TimeControlSolver(CStokesProblem, Solution, BackwardEuler; verbosity = 1, subiterations = [[1],[2],[3]], timedependent_equations = [1,2])

    ## loop in pseudo-time until stationarity detected
    ## we also output M to see that the mass constraint is preserved all the way
    change = 0.0
    for iteration = 1 : maxIterations
        ## in the advance! step we can tell which matrices of which subiterations do not change
        ## and so allow for reuse of the lu factorization
        ## here only the matrix for the density update (2nd subiteration) changes in each step
        change = advance!(TCS, timestep; reuse_matrix = [true, false, true])
        M = sum(Solution[2][:] .* xgrid[CellVolumes])
        @printf("  iteration %4d",iteration)
        @printf("  time = %.4e",TCS.ctime)
        @printf("  change = [%.4e,%.4e,%.4e]",change[1],change[2],change[3])
        @printf("  M = %.4e \n",M)
        if sum(change) < maxResidual
            println("  terminated (below tolerance)")
            break;
        end
    end

    ## compute L2 error for velocity and density
    L2error = sqrt(evaluate(L2VelocityErrorEvaluator,Solution[1]))
    println("\nL2error(Velocity) = $L2error")
    L2error = sqrt(evaluate(L2DensityErrorEvaluator,Solution[2]))
    println("L2error(Density) = $L2error")
    
    ## plots
    ## split grid into triangles for plotter
    xgrid = split_grid_into(xgrid,Triangle2D)
    nnodes = size(xgrid[Coordinates],2)

    if plot_grid
        PyPlot.figure("grid")
        ExtendableGrids.plot(xgrid, Plotter = PyPlot)
    end

    if plot_pressure
        nodevals = zeros(Float64,1,nnodes)
        PyPlot.figure("pressure")
        nodevalues!(nodevals,Solution[3],FESpacePD)
        ExtendableGrids.plot(xgrid, nodevals[1,:]; Plotter = PyPlot)
    end

    if plot_density
        nodevals = zeros(Float64,1,nnodes)
        PyPlot.figure("density")
        nodevalues!(nodevals,Solution[2],FESpacePD)
        ExtendableGrids.plot(xgrid, nodevals[1,:]; Plotter = PyPlot)
    end

    if plot_velocity
        xCoordinates = xgrid[Coordinates]
        nodevals = zeros(Float64,2,nnodes)
        nodevalues!(nodevals,Solution[1],FESpaceV)
        PyPlot.figure("velocity")
        ExtendableGrids.plot(xgrid, sqrt.(nodevals[1,:].^2+nodevals[2,:].^2); Plotter = PyPlot, isolines = 3)
        quiver(xCoordinates[1,:],xCoordinates[2,:],nodevals[1,:],nodevals[2,:])
    end
end

main()
