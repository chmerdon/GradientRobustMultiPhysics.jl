function getProblemData(polynomial_coefficients::Array{Float64,2}, nrBoundaryRegions::Int = 4)

    # auto-computing coefficients of - 2nd derivative
    l = size(polynomial_coefficients,2)
    polynomial_coefficients_rhs = zeros(Float64,2,l-2)
    for j = 1 : l - 2
        polynomial_coefficients_rhs[1,j] = -j*(j+1)*polynomial_coefficients[1,j+2]
        polynomial_coefficients_rhs[2,j] = -j*(j+1)*polynomial_coefficients[2,j+2]
    end    
    u_order = l - 1;
    f_order = l - 1;

    function exact_solution!(result,x)
        result[1] = 0.0
        for j=1:l
            result[1] += polynomial_coefficients[1,j]*x[1]^(j-1)
            result[1] += polynomial_coefficients[2,j]*x[2]^(j-1)
        end
    end    

    function volume_data!(result, x)  
        result[1] = 0.0
        for j=1:l-2
            result[1] += polynomial_coefficients_rhs[1,j]*x[1]^(j-1)
            result[1] += polynomial_coefficients_rhs[2,j]*x[2]^(j-1)
        end
    end

    PD = FESolvePoisson.PoissonProblemDescription()
    PD.name = "2D Poission Test problem"
    PD.diffusion = 1.0;
    # volume data
    PD.volumedata4region = Vector{Function}(undef,1)
    PD.volumedata4region[1] = volume_data! 
    PD.quadorder4region = [0]
    # boundary data
    PD.boundarydata4bregion = Vector{Function}(undef,nrBoundaryRegions)
    PD.boundarytype4bregion = ones(nrBoundaryRegions)
    PD.quadorder4bregion = ones(length(PD.boundarydata4bregion))*2
    for j = 1 : nrBoundaryRegions
        PD.boundarydata4bregion[j] = exact_solution!
    end    

    return PD, exact_solution!
end