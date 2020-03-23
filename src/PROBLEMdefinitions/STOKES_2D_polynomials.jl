function getProblemData(order::Int, nu::Real = 1.0, nonlinear = false, nrBoundaryRegions::Int = 4, rhs4poisson::Bool = false)

    function exact_pressure!(result,x) # integral means for unit square
        if order == 3
            result[1] = x[1]^2 + x[2]^2 - 2//3;    
        elseif order == 2
            result[1] = x[1] - 1//2;    
        else
            result[1] = 0;    
        end
    end
    
    
    function volume_data!(result,x)
        result[1] = 0.0;
        result[2] = 0.0;
        if order == 2
            result[1] = 6.0*nu
            result[2] = -6.0*nu
            if rhs4poisson == false
                result[1] += 1.0
                if nonlinear
                    result[1] -= 18.0*x[1]^2*x[2]     
                    result[2] -= 18.0*x[1]*x[2]^2
                end    
            end    
        elseif order == 3
            result[1] = 24.0*nu*x[2]
            result[2] = -24.0*nu*x[1]
            if rhs4poisson == false
                result[1] += 2*x[1]
                result[2] += 2*x[2]
                if nonlinear
                    result[1] -= 48.0*x[1]^3*x[2]^2     
                    result[2] -= 48.0*x[1]^2*x[2]^3
                end
            end
        end    
    end
    
    function exact_velocity!(result,x)
        result[1] = 0.0;
        result[2] = 0.0;
        if order == 0
            result[1] = -1.0
            result[2] = 1.0
        elseif order == 1
            result[1] = x[2]
            result[2] = 0.0
        elseif order == 2
            result[1] = -3*x[2]^2
            result[2] = 3*x[1]^2
        elseif order == 3
            result[1] = -4.0*x[2]^3
            result[2] = 4.0*x[1]^3
        end    
    end
    

    # write data into problem description structure
    PD = FESolveStokes.StokesProblemDescription()
    if nonlinear
        PD.name = "Navier-Stokes polynomials (order = " * string(order) * ")";
    else    
        PD.name = "Stokes polynomials (order = " * string(order) * ")";
    end
    PD.viscosity = nu;
    PD.volumedata4region = Vector{Function}(undef,1)
    PD.boundarydata4bregion = Vector{Function}(undef,nrBoundaryRegions)
    PD.boundarytype4bregion = ones(length(PD.boundarydata4bregion))
    PD.quadorder4bregion = zeros(length(PD.boundarydata4bregion))
    PD.quadorder4region =[(order < 2) ? 0 : order-2]
    PD.volumedata4region[1] = volume_data!
    for j = 1 : length(PD.boundarytype4bregion)
        PD.boundarydata4bregion[j] = exact_velocity!
        PD.quadorder4bregion[j] = order
    end    

    return PD, exact_velocity!, exact_pressure!
end