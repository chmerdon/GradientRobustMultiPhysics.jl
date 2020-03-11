function getProblemData(nu::Real = 1.0, nrBoundaryRegions::Int = 4)

    function zero_data!(result,x)
        result[1] = 0.0;
        result[2] = 0.0;
    end
    
    function exact_velocity!(result,x)
        r = sqrt(x[1]^2+x[2]^2)
        if (r <= 0.2)
            result[1] = -5*x[2]
            result[2] = 5*x[1]
        elseif (r <= 0.4)
            result[1] = -2*x[2]/r + 5*x[2]
            result[2] = 2*x[1]/r - 5*x[1]
        else
            result[1] = 0.0;
            result[2] = 0.0;
        end
    end
    
    # write data into problem description structure
    PD = FESolveStokes.StokesProblemDescription()
    PD.name = "Gresho vortex"
    PD.viscosity = nu;
    PD.time_dependent_data = false;
    PD.volumedata4region = Vector{Function}(undef,1)
    PD.quadorder4region = [0]
    PD.volumedata4region[1] = zero_data!
    
    PD.boundarydata4bregion = Vector{Function}(undef,nrBoundaryRegions)
    PD.boundarytype4bregion = ones(length(PD.boundarydata4bregion))
    PD.quadorder4bregion = zeros(length(PD.boundarydata4bregion))
    for j = 1 : length(PD.boundarytype4bregion)
        PD.boundarydata4bregion[j] = zero_data!
    end    

    return PD, exact_velocity!
end