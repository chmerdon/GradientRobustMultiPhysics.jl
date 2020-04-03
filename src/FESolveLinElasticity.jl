module FESolveLinElasticity

export solveLinElasticityProblem!

using SparseArrays
using ExtendableSparse
using LinearAlgebra
using BenchmarkTools
using FiniteElements
using FESolveCommon
using FESolveStokes
using Grid



mutable struct ElasticityProblemDescription
    name::String
    time_dependent_data:: Bool
    shear_modulus::Real # Lame constant
    lambda::Real # Lame constant
    volumedata4region:: Vector{Function}
    quadorder4region:: Vector{Int64}
    boundarydata4bregion:: Vector{Function}
    boundarytype4bregion:: Vector{Int64}
    quadorder4bregion:: Vector{Int64}
    ElasticityProblemDescription() = new("undefined elasticity problem", false,1.0,1.0)
end

function show(PD::ElasticityProblemDescription)

	println("ElasticityProblem description");
	println("         name : " * PD.name);
	println("    time-dep. : " * string(PD.time_dependent_data));
    print("rhs is defined: ");
    if length(PD.volumedata4region) > 0
        println("true")
    else
        println("false")    
    end    
    print("boundary_types: ");
    Base.show(PD.boundarytype4bregion)
    println("");

end


# computes solution of linear elasticity problem
function solveLinElasticityProblem!(val4dofs::Array, PD::ElasticityProblemDescription, FE::AbstractH1FiniteElement, dirichlet_penalty = 1e60)

    println("\nSOLVING LINEAR ELASTICITY PROBLEM")

    println(" |   FE = " * FE.name)
    ndofs = FiniteElements.get_ndofs(FE)
    println(" |ndofs = ", ndofs)
    println(" |");
    println(" |PROGRESS")

    # assemble system 
    @time begin
        # compute Poisson operator
        A = ExtendableSparseMatrix{Float64,Int64}(ndofs,ndofs)
        FESolveCommon.assemble_operator!(A,FESolveCommon.CELL_CEPSUdotEPSV,FE,PD.shear_modulus, PD.lambda);
        println("finished")
    end

    @time begin
        # compute right-hand side vector
        b = zeros(Float64,ndofs);
        for region = 1 : length(PD.volumedata4region)
            print("    |assembling rhs in region $region...")
            quadorder = PD.quadorder4bregion[region] + FiniteElements.get_polynomial_order(FE)
            FESolveCommon.assemble_operator!(b, FESolveCommon.CELL_FdotV, FE, PD.volumedata4region[region], quadorder)
            println("finished")
        end
    end

    @time begin
        # compute and apply boundary data
        println("    |incorporating boundary data...")
        print("       Dirichlet : ")
        Dbboundary_ids = findall(x->x == 1, PD.boundarytype4bregion)
        Base.show(Dbboundary_ids); println("");
        celldim::Int = size(FE.grid.nodes4cells,2) - 1;
        bdofs = FESolveCommon.computeDirichletBoundaryData!(val4dofs,FE,Dbboundary_ids, PD.boundarydata4bregion,celldim > 1,PD.quadorder4bregion);
        for i = 1 : length(bdofs)
           A[bdofs[i],bdofs[i]] = dirichlet_penalty;
           b[bdofs[i]] = val4dofs[bdofs[i]]*dirichlet_penalty;
        end
        print("         Neumann : ")
        Nbboundary_ids = findall(x->x == 2, PD.boundarytype4bregion)
        Base.show(Nbboundary_ids); println("");
        for j = 1 : length(Nbboundary_ids)
            if PD.quadorder4bregion[Nbboundary_ids[j]] > -1
                FESolveCommon.assemble_operator!(b,FESolveCommon.BFACE_FdotV,FE,Nbboundary_ids[j], PD.boundarydata4bregion[Nbboundary_ids[j]], PD.quadorder4bregion[Nbboundary_ids[j]]);
            end   
        end 
        println("     ...finished")
    end

    for i = 1 : length(bdofs)
       A[bdofs[i],bdofs[i]] = dirichlet_penalty;
       b[bdofs[i]] = val4dofs[bdofs[i]]*dirichlet_penalty;
    end
    
    @time begin
        print("    |solving...")
        val4dofs[:] = A\b;
        println("finished")
    end

    # compute residual (exclude bdofs)
    residual = A*val4dofs - b
    residual[bdofs] .= 0
    residual = norm(residual);
    println("    |residual=", residual)
    
    return residual
end


end
