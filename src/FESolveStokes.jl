module FESolveStokes

export solveStokesProblem!

using ExtendableSparse
using SparseArrays
using LinearAlgebra
using BenchmarkTools
using FiniteElements
using FESolveCommon
using DiffResults
using ForwardDiff
using Grid
using Quadrature


# STOKES operator
include("FEoperators/H1L2_cell_Stokes.jl");

function solveStokesProblem!(val4dofs::Array,nu::Real,volume_data!::Function,boundary_data!,grid::Grid.Mesh,FE_velocity::FiniteElements.AbstractFiniteElement,FE_pressure::FiniteElements.AbstractFiniteElement,quadrature_order::Int, use_reconstruction::Bool = false, dirichlet_penalty::Float64 = 1e60)
        
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
        assemble_Stokes_Operator4FE!(A,nu,FE_velocity,FE_pressure);
        println("finished")
    end
        
    @time begin
        # compute right-hand side vector
        print("    |assembling rhs...")
        b = zeros(Float64,ndofs);
        if use_reconstruction
            @assert FiniteElements.Hdivreconstruction_available(FE_velocity)
            FE_Reconstruction = FiniteElements.get_Hdivreconstruction_space(FE_velocity);
            ndofsHdiv = FiniteElements.get_ndofs(FE_Reconstruction)
            b2 = zeros(Float64,ndofsHdiv);
            FESolveCommon.assemble_rhsL2!(b2, volume_data!, FE_Reconstruction, quadrature_order)
            println("finished")
            print("    |Hdivreconstruction...")
            T = ExtendableSparseMatrix{Float64,Int64}(ndofs_velocity,ndofsHdiv)
            FiniteElements.get_Hdivreconstruction_trafo!(T,FE_velocity);
            b[1:ndofs_velocity] = T*b2;
        else
            FESolveCommon.assemble_rhsL2!(b, volume_data!, FE_velocity, quadrature_order)
        end    
        println("finished")
    end
    
     # add value for Lagrange multiplier for integral mean
    if length(val4dofs) == ndofs
        append!(val4dofs,0.0);
        append!(b,0.0); # add value for Lagrange multiplier for integral mean
    end
    
    # apply boundary data
    bdofs = FESolveCommon.computeDirichletBoundaryData!(val4dofs,FE_velocity,boundary_data!,true);
    for i = 1 : length(bdofs)
       A[bdofs[i],bdofs[i]] = dirichlet_penalty;
       b[bdofs[i]] = val4dofs[bdofs[i]]*dirichlet_penalty;
    end
    @assert maximum(bdofs) <= ndofs_velocity

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
