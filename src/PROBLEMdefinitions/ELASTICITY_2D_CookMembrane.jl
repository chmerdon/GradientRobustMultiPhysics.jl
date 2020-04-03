function getProblemData(shear_modulus = 1.0, lambda = 1.0)

    function zero_data!(result,x)
        result[1] = 0.0
    end     

    function neumann_data_right!(result, x)  
        result[1] = 0.0
        result[2] = 1000.0
    end

    PD = FESolveLinElasticity.ElasticityProblemDescription()
    PD.name = "Cook membrane linear elasticity"
    PD.shear_modulus = shear_modulus;
    PD.lambda = lambda
    # volume data
    PD.volumedata4region = Vector{Function}(undef,1)
    PD.volumedata4region[1] = zero_data! 
    PD.quadorder4region = zeros(Int64,1)
    # boundary data
    PD.boundarydata4bregion = Vector{Function}(undef,4)
    PD.boundarytype4bregion = ones(Int64,4)
    PD.quadorder4bregion = zeros(Int64,4)

    # left boundary (Dirichlet zero)
    PD.boundarydata4bregion[1] = zero_data!
    
    # right boundary (inhomhogeneous Neumann)
    PD.boundarydata4bregion[3] = neumann_data_right!
    PD.boundarytype4bregion[3] = 2
    
    # top/bottom boundary (homogeneous Neumann)
    PD.boundarydata4bregion[2] = zero_data!
    PD.boundarytype4bregion[2] = 2
    PD.boundarydata4bregion[4] = zero_data!
    PD.boundarytype4bregion[4] = 2
    
    return PD
end