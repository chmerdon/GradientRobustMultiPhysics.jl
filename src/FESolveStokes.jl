module FESolveStokes

export solveStokesProblem!

using ExtendableSparse
using LinearAlgebra
using BenchmarkTools
using FiniteElements
using FESolveCommon
using Grid
using Quadrature
using ForwardDiff



mutable struct StokesProblemDescription
    name::String
    time_dependent:: Bool
    viscosity:: Float64
    initial_time:: Float64
    final_time:: Float64
    initial_velocity:: Function
    volumedata4region:: Vector{Function}
    quadorder4region:: Vector{Int64}
    boundarydata4bregion:: Vector{Function}
    boundarytype4bregion:: Vector{Int64}
    quadorder4bregion:: Vector{Int64}
    StokesProblemDescription() = new("undefined problem", false, 1.0)
end

function show(PD::StokesProblemDescription)

	println("Problem description");
	println("         name : " * PD.name);
	println("    time-dep. : " * string(PD.time_dependent));
    if (PD.time_dependent)
        println("time-interval : ["* PD.initial_time * "," * PD.final_time * "]");
    end    
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




# STOKES operator
include("FEoperators/CELL_STOKES.jl");



function solveStokesProblem!(val4dofs::Array,PD::StokesProblemDescription, FE_velocity::FiniteElements.AbstractFiniteElement,FE_pressure::FiniteElements.AbstractFiniteElement, use_reconstruction::Bool = false, dirichlet_penalty::Float64 = 1e60)
        
    ndofs_velocity = FiniteElements.get_ndofs(FE_velocity);
    ndofs_pressure = FiniteElements.get_ndofs(FE_pressure);
    ndofs = ndofs_velocity + ndofs_pressure;
    
    println("\nSOLVING STOKES PROBLEM")
    println(" |FEvelocity = " * FE_velocity.name)
    println(" |FEpressure = " * FE_pressure.name)
    println(" |totalndofs = ", ndofs)
    println(" |");
    println(" |PROGRESS")
    
    # assemble system 
    @time begin
        print("    |assembling matrix...")
        A = ExtendableSparseMatrix{Float64,Int64}(ndofs+1,ndofs+1) # +1 due to Lagrange multiplier for integral mean
        assemble_Stokes_Operator4FE!(A,PD.viscosity,FE_velocity,FE_pressure);
        println("finished")
    end
        
    @time begin
        # compute right-hand side vector
        b = zeros(Float64,ndofs);
        for region = 1 : length(PD.volumedata4region)
            print("    |assembling rhs in region $region...")
            if use_reconstruction
                @assert FiniteElements.Hdivreconstruction_available(FE_velocity)
                FE_Reconstruction = FiniteElements.get_Hdivreconstruction_space(FE_velocity);
                ndofsHdiv = FiniteElements.get_ndofs(FE_Reconstruction)
                b2 = zeros(Float64,ndofsHdiv);
                quadorder = PD.quadorder4bregion[region] + FiniteElements.get_polynomial_order(FE_Reconstruction)
                FESolveCommon.assemble_rhsL2!(b2, PD.volumedata4region[region], FE_Reconstruction, quadorder)
                println("finished")
                print("    |Hdivreconstruction...")
                T = ExtendableSparseMatrix{Float64,Int64}(ndofs_velocity,ndofsHdiv)
                FiniteElements.get_Hdivreconstruction_trafo!(T,FE_velocity);
                b[1:ndofs_velocity] = T*b2;
            else
                quadorder = PD.quadorder4bregion[region] + FiniteElements.get_polynomial_order(FE_velocity)
                FESolveCommon.assemble_rhsL2!(b, PD.volumedata4region[region], FE_velocity, quadorder)
            end    
            println("finished")
        end
    end

    @time begin
        # compute and apply boundary data
        print("    |boundary data...")
        bdofs = FESolveCommon.computeDirichletBoundaryData!(val4dofs,FE_velocity,PD.boundarydata4bregion[1],true);
        for i = 1 : length(bdofs)
           A[bdofs[i],bdofs[i]] = dirichlet_penalty;
           b[bdofs[i]] = val4dofs[bdofs[i]]*dirichlet_penalty;
        end
        println("finished")
    end
    
     # add value for Lagrange multiplier for integral mean
    if length(val4dofs) == ndofs
        append!(val4dofs,0.0);
        append!(b,0.0); # add value for Lagrange multiplier for integral mean
    end
    

    @time begin
        print("    |solving...")
        try
            val4dofs[:] = A\b;
        catch    
            println("Unsupported Number type for sparse lu detected: trying again with dense matrix");
            try
                val4dofs[:] = Array{typeof(FE_velocity.grid.coords4nodes[1]),2}(A)\b;
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
