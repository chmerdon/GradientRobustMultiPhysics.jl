function getProblemData(diffusion = 1.0, nrBoundaryRegions::Int = 4)

    function exact_solution!(result,x)
        result[1] = x[1]*x[2]*(x[1] - 1.0)*(x[2] - 1.0)
    end   
    
    function zero_data!(result,x)
        result[1] = 0.0
    end     

    function exact_gradient!(result,x)
        result[1] = (2*x[1] - 1.0)*x[2]*(x[2] - 1.0)
        result[2] = (2*x[2] - 1.0)*x[1]*(x[1] - 1.0)
    end    

    function volume_data!(result, x)  
        result[1] = -diffusion*(2*x[1]*(x[1]-1.0) + 2*x[2]*(x[2]-1.0))
    end

    PD = FESolvePoisson.PoissonProblemDescription()
    PD.name = "2D Poisson Test problem with zero bnd data on  unit square (P4 bubble)"
    PD.diffusion = diffusion;
    PD.quadorder4diffusion = 0;
    # volume data
    PD.volumedata4region = Vector{Function}(undef,1)
    PD.volumedata4region[1] = volume_data! 
    PD.quadorder4region = ones(Int64,1)*(2)
    # boundary data
    PD.boundarydata4bregion = Vector{Function}(undef,nrBoundaryRegions)
    PD.boundarytype4bregion = ones(nrBoundaryRegions)
    PD.quadorder4bregion = zeros(Int64,nrBoundaryRegions)
    for j = 1 : nrBoundaryRegions
        PD.boundarydata4bregion[j] = zero_data!
    end    

    return PD, exact_solution!, exact_gradient!
end