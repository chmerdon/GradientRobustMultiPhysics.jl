module FESolvePoisson

export solvePoissonProblem!

using SparseArrays
using ExtendableSparse
using LinearAlgebra
using BenchmarkTools
using FiniteElements
using FESolveCommon
using Grid


# computes solution of Poisson problem
function solvePoissonProblem!(val4dofs::Array, nu::Real, volume_data!::Function, boundary_data!, FE::AbstractH1FiniteElement, quadrature_order::Int, dirichlet_penalty = 1e60)

    println("\nSOLVING POISSON PROBLEM")

    println(" |   FE = " * FE.name)
    println(" |ndofs = ", FiniteElements.get_ndofs(FE))
    println(" |");
    println(" |PROGRESS")

    # assemble system 
    A, b = FESolveCommon.assembleSystem(nu,"H1","L2",volume_data!,FE,quadrature_order);

    # apply boundary data
    celldim::Int = size(FE.grid.nodes4cells,2) - 1;
    if (celldim == 1)
        bdofs = FESolveCommon.computeDirichletBoundaryData!(val4dofs,FE,boundary_data!,false);
    else
        bdofs = FESolveCommon.computeDirichletBoundaryData!(val4dofs,FE,boundary_data!,true);
    end
    
    for i = 1 : length(bdofs)
       A[bdofs[i],bdofs[i]] = dirichlet_penalty;
       b[bdofs[i]] = val4dofs[bdofs[i]]*dirichlet_penalty;
    end
    
    @time begin
        print("    |solving...")
        try
            val4dofs[:] = A\b;
        catch    
            println("Unsupported Number type for sparse lu detected: trying again with dense matrix");
            try
                val4dofs[:] = Array{typeof(FE.grid.coords4nodes[1]),2}(A)\b;
            catch OverflowError
                println("OverflowError (Rationals?): trying again as Float64 sparse matrix");
                val4dofs[:] = Array{Float64,2}(A)\b;
            end
        end
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
