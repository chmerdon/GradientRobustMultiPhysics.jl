function getProblemData(nu::Real = 1.0, nrBoundaryRegions::Int = 4, rhs4poisson::Bool = false)

    function exact_pressure!(result, x) # exact pressure
        result[1] = x[1]^3 + x[2]^3 - 1//2 # integral mean for unit quare
    end


    function volume_data!(result,x)
        result[1] = nu*(4*(2*x[2]-1)*(3*x[1]^4-6*x[1]^3+6*x[1]^2*x[2]^2 - 6*x[1]^2*x[2] + 3*x[1]^2 -6*x[1]*x[2]^2 + 6*x[1]*x[2] + x[2]^2 - x[2]))
        result[2] = -nu*(4*(2*x[1]-1)*(3*x[2]^4-6*x[2]^3+6*x[2]^2*x[1]^2 - 6*x[2]^2*x[1] + 3*x[2]^2 -6*x[2]*x[1]^2 + 6*x[2]*x[1] + x[1]^2 - x[1]))
        if rhs4poisson == false
            result[1] += 3*x[1]^2
            result[2] += 3*x[2]^2
        end       
    end

    function exact_velocity!(result,x)
        result[1] = -2.0*(1.0-x[1])^2*x[1]^2*(1.0-x[2])^2*x[2]
        result[1] += 2.0*(1.0-x[1])^2*x[1]^2*(1.0-x[2])*x[2]^2
        result[2] = 2.0*(1.0-x[2])^2*x[2]^2*(1.0-x[1])^2*x[1]
        result[2] -= 2.0*(1.0-x[2])^2*x[2]^2*(1.0-x[1])*x[1]^2
    end

    # write data into problem description structure
    PD = FESolveStokes.StokesProblemDescription()
    PD.name = "P7 vortex";
    PD.viscosity = nu;
    PD.volumedata4region = Vector{Function}(undef,1)
    PD.boundarydata4bregion = Vector{Function}(undef,nrBoundaryRegions)
    PD.boundarytype4bregion = ones(length(PD.boundarydata4bregion))
    PD.quadorder4bregion = zeros(length(PD.boundarydata4bregion))
    PD.quadorder4region = [5]
    PD.volumedata4region[1] = volume_data!
    for j = 1 : length(PD.boundarytype4bregion)
        PD.boundarydata4bregion[j] = exact_velocity!
        PD.quadorder4bregion[j] = 0
    end    

    return PD, exact_velocity!, exact_pressure!
end