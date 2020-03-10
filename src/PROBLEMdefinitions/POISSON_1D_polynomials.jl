function getProblemData(polynomial_coefficients::Vector{Int64})

    # auto-computing coefficients of - 2nd derivative
    polynomial_coefficients_rhs = zeros(Float64,length(polynomial_coefficients)-2)
    for j = 1 : length(polynomial_coefficients) - 2
        polynomial_coefficients_rhs[j] = -j*(j+1)*polynomial_coefficients[j+2]
    end    
    u_order = length(polynomial_coefficients_rhs) - 1;
    f_order = length(polynomial_coefficients_rhs) - 1;

    function exact_solution!(result,x)
        result[1] = 0.0
        for j=1:length(polynomial_coefficients)
            result[1] += polynomial_coefficients[j]*x[1]^(j-1)
        end
    end    

    function volume_data!(result, x)  
        result[1] = 0.0
        for j=1:length(polynomial_coefficients_rhs)
            result[1] += polynomial_coefficients_rhs[j]*x[1]^(j-1)
        end
    end

    PD = FESolvePoisson.PoissonProblemDescription()
    PD.name = "1D Poission Test problem"
    PD.diffusion = 1.0;
    # volume data
    PD.volumedata4region = Vector{Function}(undef,1)
    PD.volumedata4region[1] = volume_data! 
    PD.quadorder4region = [0]
    # boundary data
    PD.boundarydata4bregion = Vector{Function}(undef,1)
    PD.boundarytype4bregion = [1]
    PD.quadorder4bregion = [u_order]
    PD.boundarydata4bregion[1] = exact_solution!

    return PD, exact_solution!
end