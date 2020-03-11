module FESolvePoisson

export solvePoissonProblem!

using SparseArrays
using ExtendableSparse
using LinearAlgebra
using BenchmarkTools
using FiniteElements
using FESolveCommon
using Grid



mutable struct PoissonProblemDescription
    name::String
    time_dependent_data:: Bool
    diffusion:: Float64
    volumedata4region:: Vector{Function}
    quadorder4region:: Vector{Int64}
    boundarydata4bregion:: Vector{Function}
    boundarytype4bregion:: Vector{Int64}
    quadorder4bregion:: Vector{Int64}
    PoissonProblemDescription() = new("undefined Poisson problem", false, 1.0)
end

function show(PD::PoissonProblemDescription)

	println("PoissonProblem description");
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


# computes solution of Poisson problem
function solvePoissonProblem!(val4dofs::Array, PD::PoissonProblemDescription, FE::AbstractH1FiniteElement, dirichlet_penalty = 1e60)

    println("\nSOLVING POISSON PROBLEM")

    println(" |   FE = " * FE.name)
    ndofs = FiniteElements.get_ndofs(FE)
    println(" |ndofs = ", ndofs)
    println(" |");
    println(" |PROGRESS")

    # assemble system 
    @time begin
        # compute Poisson operator
        print("    |assembling matrix...")
        A = ExtendableSparseMatrix{Float64,Int64}(ndofs,ndofs)
        FESolveCommon.assemble_operator!(A,FESolveCommon.CELL_DUdotDV,FE,PD.diffusion);
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
        Dnboundary_ids = findall(x->x == 2, PD.boundarytype4bregion)
        if length(Dnboundary_ids) > 0
            print("       Do-nothing : ")
            Base.show(Dnboundary_ids); println("");
        end    
        print("       Dirichlet : ")
        Dbboundary_ids = findall(x->x == 1, PD.boundarytype4bregion)
        Base.show(Dbboundary_ids); println("");
        celldim::Int = size(FE.grid.nodes4cells,2) - 1;
        if (celldim == 1)
            bdofs = FESolveCommon.computeDirichletBoundaryData!(val4dofs,FE,Dbboundary_ids, PD.boundarydata4bregion,false,PD.quadorder4bregion);
        else
            bdofs = FESolveCommon.computeDirichletBoundaryData!(val4dofs,FE,Dbboundary_ids, PD.boundarydata4bregion,true,PD.quadorder4bregion);
        end    
        for i = 1 : length(bdofs)
           A[bdofs[i],bdofs[i]] = dirichlet_penalty;
           b[bdofs[i]] = val4dofs[bdofs[i]]*dirichlet_penalty;
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
