function getProblemData(nu::Real = 1.0, lambda::Real = 0.0; c::Real = 1.0, gamma::Real = 1.0, total_mass::Real = 1.0, nrBoundaryRegions::Int = 4)


    function exact_density!(result, x) # exact density
        fill!(result, 1.0)
    end


    function equation_of_state!(pressure,density)
        for j=1:length(density)
            pressure[j] = c*density[j]^gamma
        end    
    end    


    function exact_pressure!(result, x) # exact pressure
        exact_density!(result, x)
        equation_of_state!(result,result)
    end

    function volume_data!(result,x)
        result[1] = nu*(4*(2*x[2]-1)*(3*x[1]^4-6*x[1]^3+6*x[1]^2*x[2]^2 - 6*x[1]^2*x[2] + 3*x[1]^2 -6*x[1]*x[2]^2 + 6*x[1]*x[2] + x[2]^2 - x[2]))
        result[2] = -nu*(4*(2*x[1]-1)*(3*x[2]^4-6*x[2]^3+6*x[2]^2*x[1]^2 - 6*x[2]^2*x[1] + 3*x[2]^2 -6*x[2]*x[1]^2 + 6*x[2]*x[1] + x[1]^2 - x[1])) 
    end

    function exact_velocity!(result,x)
        result[1] = -2.0*(1.0-x[1])^2*x[1]^2*(1.0-x[2])^2*x[2]
        result[1] += 2.0*(1.0-x[1])^2*x[1]^2*(1.0-x[2])*x[2]^2
        result[2] = 2.0*(1.0-x[2])^2*x[2]^2*(1.0-x[1])^2*x[1]
        result[2] -= 2.0*(1.0-x[2])^2*x[2]^2*(1.0-x[1])*x[1]^2
    end

    # write data into problem description structure
    PD = FESolveCompressibleStokes.CompressibleStokesProblemDescription()

    # parameters
    PD.name = "P7 vortex compressible";
    PD.shear_modulus = nu;
    PD.lambda = lambda
    PD.total_mass = total_mass

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
    PD.quadorder4gravity = -1

    # equation of state
    PD.equation_of_state = equation_of_state!

    return PD, exact_velocity!, exact_density!, exact_pressure!
end