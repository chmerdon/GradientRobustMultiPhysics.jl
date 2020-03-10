    function exact_pressure!(nu = 1.0, integral_mean = 0.0)
        function closure(result,x)
            result[1] = nu*(-2*x[1] - integral_mean)
        end    
    end

    function volume_data!(result,x)
        result[1] = 0.0;
        result[2] = 0.0;
    end

    function exact_velocity!(result,x)
        result[1] = x[2]*(2.0-x[2]);
        result[2] = 0.0;
    end

function getProblemData(nrBoundaryRegions::Int = 4)

    # write data into problem description structure
    PD = FESolveStokes.StokesProblemDescription()
    PD.name = "Hagen-Poisseuille"
    PD.viscosity = 1.0;

    # volume data
    PD.volumedata4region = Vector{Function}(undef,1)
    PD.volumedata4region[1] = volume_data! 
    PD.quadorder4region = [0]

    # boundary data
    PD.boundarydata4bregion = Vector{Function}(undef,nrBoundaryRegions)
    PD.boundarytype4bregion = ones(length(PD.boundarydata4bregion))
    PD.quadorder4bregion = ones(length(PD.boundarydata4bregion))*2
    for j = 1 : length(PD.boundarytype4bregion)
        PD.boundarydata4bregion[j] = exact_velocity!
    end    
    return PD
end