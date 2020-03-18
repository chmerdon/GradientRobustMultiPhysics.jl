using DiffResults
using ForwardDiff

function getProblemData(nu::Real = 1.0, lambda::Real = 0.0; use_nonlinear_convection::Bool, use_gravity::Bool, symmetric_gradient::Bool = false, c::Real = 1.0, gamma::Real = 1.0, density_power::Real = 0.0, total_mass::Real = 1.0, nrBoundaryRegions::Int = 4)


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

    rho = zeros(Real,1)
    function exact_streamfunction(x)
        exact_density!(rho,x)
        return (x[1]^2 * (x[1] - 1)^2 * x[2]^2 * (x[2] - 1)^2)
    end


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
    gravity = [0.0,-1.0]
    function volume_data!(result,x)
        fill!(result,0.0)

        # add friction term
        ForwardDiff.hessian!(hessian,velo1,x)
        if symmetric_gradient == false
            result[1] -= 2*nu * (DiffResults.hessian(hessian)[1] + DiffResults.hessian(hessian)[4])
        else
            result[1] -= nu * (DiffResults.hessian(hessian)[1] + DiffResults.hessian(hessian)[4])
            result[1] -= nu * (DiffResults.hessian(hessian)[1])
            result[2] -= nu * (DiffResults.hessian(hessian)[3])
        end
        result[1] -= lambda * DiffResults.hessian(hessian)[1]
        result[2] -= lambda * DiffResults.hessian(hessian)[3]
        
        ForwardDiff.hessian!(hessian,velo2,x)
        if symmetric_gradient == false
            result[2] -= 2*nu * (DiffResults.hessian(hessian)[1] + DiffResults.hessian(hessian)[4])
        else
            result[2] -= nu * (DiffResults.hessian(hessian)[1] + DiffResults.hessian(hessian)[4])
            result[1] -= nu * (DiffResults.hessian(hessian)[2])
            result[2] -= nu * (DiffResults.hessian(hessian)[4])
        end
        result[1] -= lambda * DiffResults.hessian(hessian)[2]
        result[2] -= lambda * DiffResults.hessian(hessian)[4]

        #
        
        # add gradient of pressure
        ForwardDiff.gradient!(grad,p,x);
        result[1] += DiffResults.gradient(grad)[1]
        result[2] += DiffResults.gradient(grad)[2]

        # remove gravity effect (todo)
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


    # write data into problem description structure
    PD = FESolveCompressibleStokes.CompressibleStokesProblemDescription()

    # parameters
    PD.name = "P7 vortex compressible";
    PD.shear_modulus = (symmetric_gradient) ? nu : 2*nu;
    PD.lambda = lambda
    PD.total_mass = total_mass
    PD.use_symmetric_gradient = symmetric_gradient
    PD.use_nonlinear_convection = use_nonlinear_convection

    # boundar data
    PD.boundarydata4bregion = Vector{Function}(undef,nrBoundaryRegions)
    PD.boundarytype4bregion = ones(length(PD.boundarydata4bregion))
    PD.quadorder4bregion = zeros(length(PD.boundarydata4bregion))

    # volume data
    PD.volumedata4region = Vector{Function}(undef,1)
    PD.quadorder4region = [5]
    PD.volumedata4region[1] = volume_data!
    for j = 1 : length(PD.boundarytype4bregion)
        PD.boundarydata4bregion[j] = exact_velocity!
        PD.quadorder4bregion[j] = 0
    end    

    # gravity

    function gravity!(result,x)
        result[1] = 0
        result[2] = -1.0
    end    

    if use_gravity
        PD.gravity = gravity!
        PD.quadorder4gravity = 0
    else 
        PD.quadorder4gravity = -1
    end

    # equation of state
    PD.equation_of_state = equation_of_state!

    return PD, exact_velocity!, exact_density!, exact_pressure!
end