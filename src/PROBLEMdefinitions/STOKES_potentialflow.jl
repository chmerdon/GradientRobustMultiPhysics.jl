function getProblemData(order::Int, nonlinear::Bool, nrBoundaryRegions::Int = 4)
    @assert order == 2

    function volume_data!(result,x)
        result[1] = 0.0 # pressure balances everything
    end

    function exact_velocity!(result,x)
        if order == 2
            result[1] = 3*x[1]^2 - 3*x[2]^2
            result[2] = -6*x[1]*x[2]
        end    
    end


    function exact_pressure!(result, x) # exact exact_pressure
        result[1] = 0.0
        if nonlinear
            result[1] += -9 // 2 * (x[1]^2 + x[2]^2)^2 + 14 // 5
        end    
    end

    # write data into problem description structure
    PD = FESolveStokes.StokesProblemDescription()
    PD.name = "P7 potential flow of order $order";
    PD.viscosity = 1.0;
    PD.volumedata4region = Vector{Function}(undef,1)
    PD.boundarydata4bregion = Vector{Function}(undef,nrBoundaryRegions)
    PD.boundarytype4bregion = ones(length(PD.boundarydata4bregion))
    PD.quadorder4bregion = zeros(length(PD.boundarydata4bregion))
    PD.quadorder4region = [0]
    PD.volumedata4region[1] = volume_data!
    for j = 1 : length(PD.boundarytype4bregion)
        PD.boundarydata4bregion[j] = exact_velocity!
        PD.quadorder4bregion[j] = order
    end    

    return PD, exact_velocity!, exact_pressure!
end