module FESolvePoisson

export solvePoissonProblem!,solveMixedPoissonProblem!

using SparseArrays
using ExtendableSparse
using LinearAlgebra
using BenchmarkTools
using FiniteElements
using FESolveCommon
using FESolveStokes
using Grid



mutable struct PoissonProblemDescription
    name::String
    time_dependent_data:: Bool
    diffusion # Real/Matrix or matrix-valued function
    quadorder4diffusion::Int64
    convection # vector-valued function
    quadorder4convection::Int64
    volumedata4region:: Vector{Function}
    quadorder4region:: Vector{Int64}
    boundarydata4bregion:: Vector{Function}
    boundarytype4bregion:: Vector{Int64}
    quadorder4bregion:: Vector{Int64}
    PoissonProblemDescription() = new("undefined Poisson problem", false,1.0,0,Nothing,-1)
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
        A = ExtendableSparseMatrix{Float64,Int64}(ndofs,ndofs)
        if typeof(PD.diffusion) <: Real
            print("    |assembling matrix (constant scalar diffusion)...")
            FESolveCommon.assemble_operator!(A,FESolveCommon.CELL_DUdotDV,FE,PD.diffusion);
        else
            print("    |assembling matrix (variable, matrix-valued diffusion)...")
            FESolveCommon.assemble_operator!(A,FESolveCommon.CELL_MdotDUdotDV,FE,PD.diffusion, PD.quadorder4diffusion);
        end        
        if PD.quadorder4convection > -1
            print("    |assembling convection matrix...")
            FESolveCommon.assemble_operator!(A,FESolveCommon.CELL_FdotDUdotV,FE,PD.convection,PD.quadorder4convection);
        end
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
        bdofs = FESolveCommon.computeDirichletBoundaryData!(val4dofs,FE,Dbboundary_ids, PD.boundarydata4bregion,celldim > 1,PD.quadorder4bregion);
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



# computes solution of Poisson problem in (dual) mixed formulation
#   sigma = diffusion * nabla u
#   div(sigma) = -f
function solveMixedPoissonProblem!(val4dofs::Array, PD::PoissonProblemDescription, FE_stress::AbstractFiniteElement, FE_divergence::AbstractFiniteElement, dirichlet_penalty = 1e60)

    println("\nSOLVING MIXED POISSON PROBLEM")

    println(" |   FE_stress = " * FE_stress.name)
    println(" |      FE_div = " * FE_divergence.name)
    ndofs_stress = FiniteElements.get_ndofs(FE_stress)
    ndofs_div = FiniteElements.get_ndofs(FE_divergence)
    ndofs = ndofs_stress + ndofs_div
    println(" |       ndofs = ", ndofs)
    println(" |");
    println(" |PROGRESS")

    # assemble system 
    @time begin
        # compute operator
        print("    |assembling matrix...")
        A = ExtendableSparseMatrix{Float64,Int64}(ndofs,ndofs)
        FESolveStokes.assemble_operator!(A,FESolveStokes.CELL_DIVFREE_UdotV,FE_stress,FE_divergence,1.0/PD.diffusion);
        println("finished")
    end

    # compute right-hand side vector
    @time begin
        b = zeros(Float64,ndofs)
        b2 = zeros(Float64,ndofs_div);
        print("    |assembling rhs...")
        FESolveCommon.assemble_operator!(b2, FESolveCommon.CELL_FdotV, FE_divergence, PD.volumedata4region[1], PD.quadorder4bregion[1])
        b[ndofs_stress+1:end] += b2

        # boundary integral for right-hand side
        for j = 1 : length(PD.boundarydata4bregion)
            if PD.boundarydata4bregion[j] != Nothing
                FESolveCommon.assemble_operator!(b, FESolveCommon.BFACE_FdotVn, FE_stress, j, PD.boundarydata4bregion[j], PD.quadorder4bregion[j])
            end    
        end  
        println("finished")
    end
    
    @time begin
        print("    |solving...")
        val4dofs[:] = A\b;
        println("finished")
    end

    # compute residual (exclude bdofs)
    residual = A*val4dofs - b
    #residual[bdofs] .= 0
    residual = norm(residual);
    println("    |residual=", residual)
    
    return residual
end

end
