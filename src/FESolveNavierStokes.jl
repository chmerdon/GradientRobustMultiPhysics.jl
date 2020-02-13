module FESolveNavierStokes

export solveNavierStokesProblem!

using ExtendableSparse
using LinearAlgebra
using BenchmarkTools
using FiniteElements
using FESolveCommon
using FESolveStokes
using Grid
using Quadrature
using ForwardDiff


# NAVIER-STOKES operators
include("FEoperators/H1_cell_Stokes_ADUxDV.jl");
include("FEoperators/HDIVxH1xHDIV_cell_Stokes_ADUxDV.jl");

function solveNavierStokesProblem!(val4dofs::Array,nu::Real,volume_data!::Function,boundary_data!,grid::Grid.Mesh,FE_velocity::FiniteElements.AbstractFiniteElement,FE_pressure::FiniteElements.AbstractFiniteElement,quadrature_order::Int, use_reconstruction::Bool = false, maxiterations = 20, dirichlet_penalty::Float64 = 1e60)
        
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
        print("    |assembling linear part of matrix...")
        Alin = ExtendableSparseMatrix{Float64,Int64}(ndofs+1,ndofs+1) # +1 due to Lagrange multiplier for integral mean
        FESolveStokes.assemble_Stokes_Operator4FE!(Alin,nu,FE_velocity,FE_pressure);
        println("finished")
    end
    
     # add value for Lagrange multiplier for integral mean
     if length(val4dofs) == ndofs
        append!(val4dofs,0.0);
    end

    @time begin
        # compute right-hand side vector
        print("    |assembling rhs...")
        b = zeros(Float64,ndofs+1);
        if use_reconstruction
            @assert FiniteElements.Hdivreconstruction_available(FE_velocity)
            FE_Reconstruction = FiniteElements.get_Hdivreconstruction_space(FE_velocity);
            ndofsHdiv = FiniteElements.get_ndofs(FE_Reconstruction)
            b2 = zeros(Float64,ndofsHdiv);
            FESolveCommon.assemble_rhsL2!(b2, volume_data!, FE_Reconstruction, quadrature_order)
            println("finished")
            print("    |init Hdivreconstruction...")
            T = ExtendableSparseMatrix{Float64,Int64}(ndofs+1,ndofsHdiv)
            FiniteElements.get_Hdivreconstruction_trafo!(T,FE_velocity);
            b = T*b2;
        else
            FESolveCommon.assemble_rhsL2!(b, volume_data!, FE_velocity, quadrature_order)
        end    
        println("finished")
    end

    @time begin
        # compute boundary data
        print("    |boundary data...")
        bdofs = FESolveCommon.computeDirichletBoundaryData!(val4dofs,FE_velocity,boundary_data!,true);
        println("finished")
    end
    
    
    iteration::Int64 = 0
    res = zeros(Float64,ndofs+1)
    residual = 0.0
    A = ExtendableSparseMatrix{Float64,Int64}(ndofs+1,ndofs+1) # +1 due to Lagrange multiplier for integral mean
    for j=1:maxiterations
        iteration += 1
        println("    |entering iteration $iteration")

        # reload linear part
        A = deepcopy(Alin)
        
        if (iteration > 1)
            # compute nonlinear term
            print("      |assembling nonlinear term...")
            @time begin
                if use_reconstruction
                    val4dofs_hdiv = T'*val4dofs
                    Atemp = ExtendableSparseMatrix{Float64,Int64}(ndofsHdiv,ndofs+1)
                    assemble_ugradu_matrix4FE!(Atemp,val4dofs_hdiv,FE_Reconstruction, FE_velocity);    
                    A += T*Atemp
                else   
                    assemble_ugradu_matrix4FE!(A,val4dofs,FE_velocity);
                end    
                println("finished")
            end

            # compute nonlinear residual (exclude bdofs)
            res = A*val4dofs - b
            res[bdofs] .= 0
            residual = norm(res);
            println("      |NSE residual=", residual)
            if (residual < 1e-12)
                println("    converged")
                break;
            end    
        end    


        # apply boundary data
        for i = 1 : length(bdofs)
            A[bdofs[i],bdofs[i]] = dirichlet_penalty;
            b[bdofs[i]] = val4dofs[bdofs[i]]*dirichlet_penalty;
         end

        @time begin
            print("      |solving...")
            val4dofs[:] = A\b;
            println("finished")
        end

        # compute linear residual (exclude bdofs)
        res = A*val4dofs - b
        res[bdofs] .= 0
        residual = norm(res);
        println("      |solver residual=", residual)
    end
    
    
    return residual
end


end
