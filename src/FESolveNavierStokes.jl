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
include("FEoperators/CELL_NAVIERSTOKES_AdotDUdotDV.jl");
include("FEoperators/CELL_NAVIERSTOKES_AdotDAdotDV.jl");

function solveNavierStokesProblem!(val4dofs::Array,PD::FESolveStokes.StokesProblemDescription, FE_velocity::FiniteElements.AbstractFiniteElement,FE_pressure::FiniteElements.AbstractFiniteElement, reconst_variant::Int = 0, maxiterations::Int = 20, dirichlet_penalty::Float64 = 1e60, symmetry_penalty::Float64 = 1e10)
        
    ndofs_velocity = FiniteElements.get_ndofs(FE_velocity);
    ndofs_pressure = FiniteElements.get_ndofs(FE_pressure);
    ndofs = ndofs_velocity + ndofs_pressure;
    
    println("\nSOLVING NAVIER-STOKES PROBLEM")
    println(" |FEvelocity = " * FE_velocity.name)
    println(" |FEpressure = " * FE_pressure.name)
    println(" |totalndofs = ", ndofs)
    println(" |");
    println(" |PROGRESS")
    
    # assemble system 
    @time begin
        print("    |assembling linear part of matrix...")
        Alin = ExtendableSparseMatrix{Float64,Int64}(ndofs,ndofs) # +1 due to Lagrange multiplier for integral mean
        FESolveStokes.assemble_operator!(Alin, FESolveStokes.CELL_STOKES, FE_velocity, FE_pressure, PD.viscosity);

        # compute integrals of pressure dofs
        pm = zeros(Float64,ndofs_pressure,1)
        FESolveCommon.assemble_operator!(pm,FESolveCommon.DOMAIN_1dotV,FE_pressure);
        
        println("finished")
    end
    
    @time begin
        # compute right-hand side vector
        b = zeros(Float64,ndofs);
        T = Nothing
        ndofsHdiv = 0
        FE_Reconstruction = Nothing
        for region = 1 : length(PD.volumedata4region)
            print("    |assembling rhs in region $region...")
            if reconst_variant > 0
                @assert FiniteElements.Hdivreconstruction_available(FE_velocity)
                FE_Reconstruction = FiniteElements.get_Hdivreconstruction_space(FE_velocity, reconst_variant);
                ndofsHdiv = FiniteElements.get_ndofs(FE_Reconstruction)
                b2 = zeros(Float64,ndofsHdiv);
                quadorder = PD.quadorder4bregion[region] + FiniteElements.get_polynomial_order(FE_Reconstruction)
                FESolveCommon.assemble_operator!(b2, FESolveCommon.CELL_FdotV, FE_Reconstruction, PD.volumedata4region[region], quadorder)
                println("finished")
                print("    |Hdivreconstruction...")
                T = ExtendableSparseMatrix{Float64,Int64}(ndofs,ndofsHdiv)
                FiniteElements.get_Hdivreconstruction_trafo!(T,FE_velocity,FE_Reconstruction);
                b = T*b2;
            else
                quadorder = PD.quadorder4bregion[region] + FiniteElements.get_polynomial_order(FE_velocity)
                FESolveCommon.assemble_operator!(b, FESolveCommon.CELL_FdotV, FE_velocity, PD.volumedata4region[region], quadorder)
            end    
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
        bdofs = FESolveCommon.computeDirichletBoundaryData!(val4dofs,FE_velocity,Dbboundary_ids, PD.boundarydata4bregion,true,PD.quadorder4bregion);
        Dsymboundary_ids = findall(x->x == 3, PD.boundarytype4bregion)
        if length(Dnboundary_ids) > 0
            print("       Symmetry : ")
            Base.show(Dsymboundary_ids); println("");
            FESolveCommon.assemble_operator!(Alin, FESolveCommon.BFACE_UndotVn, FE_velocity, Dsymboundary_ids, symmetry_penalty)
        end    
        println("     ...finished")
    end
    
    # fix one pressure value
    if length(Dnboundary_ids) == 0
        append!(bdofs,ndofs)
    end    
    
    iteration::Int64 = 0
    res = zeros(Float64,ndofs)
    residual = 0.0
    A = ExtendableSparseMatrix{Float64,Int64}(ndofs,ndofs)
    for j=1:maxiterations
        iteration += 1
        println("    |entering iteration $iteration")

        # reload linear part
        A = deepcopy(Alin)
        
        if (iteration > 1)

            #test = zeros(Float64,ndofs_velocity)
            #assemble_operator!(test,CELL_NAVIERSTOKES_AdotDAdotDV,FE_velocity,FE_velocity,FE_velocity,val4dofs,val4dofs);
            #Base.show(test'*val4dofs[1:ndofs_velocity])

            # compute nonlinear term
            @time begin
                if reconst_variant > 0
                    use_reconstruction4A = false
                    Atemp = ExtendableSparseMatrix{Float64,Int64}(ndofsHdiv,ndofs)
                    if use_reconstruction4A
                        print("      |assembling nonlinear term (A*D)U*V (reconstruction in A & V)...")
                        val4dofs_hdiv = val4dofs'*T
                        assemble_operator!(Atemp,CELL_NAVIERSTOKES_AdotDUdotDV,FE_Reconstruction,FE_velocity,FE_Reconstruction,val4dofs_hdiv);    
                    else    
                        print("      |assembling nonlinear term (A*D)U*V (reconstruction only in V)...")
                        assemble_operator!(Atemp,CELL_NAVIERSTOKES_AdotDUdotDV,FE_velocity,FE_velocity,FE_Reconstruction,val4dofs);    
                    end    
                    ExtendableSparse.flush!(Atemp)
                    ExtendableSparse.flush!(A)
                    A.cscmatrix += T.cscmatrix*Atemp.cscmatrix
                else   
                    print("      |assembling nonlinear term (A*D)U*V...")
                    assemble_operator!(A,CELL_NAVIERSTOKES_AdotDUdotDV,FE_velocity,FE_velocity,FE_velocity,val4dofs);
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
    
    # move pressure mean to zero
    domain_area = sum(FE_pressure.grid.volume4cells[:])
    mean = dot(pm[:],val4dofs[ndofs_velocity+1:end])/domain_area
    val4dofs[ndofs_velocity+1:end] .-= mean;
    
    return residual
end



function PerformIMEXTimeStep(TSS::FESolveStokes.TransientStokesSolver, dt::Real = 1 // 10)
    @assert TSS.ProblemData.time_dependent_data == false # true case has to be implemented

    # update matrix and right-hand side vector

    println("    |")
    println("    |time + dt = " * string(TSS.current_time) * " + " * string(dt))
    if TSS.current_dt != dt
        println("    |updating matrix...")
        TSS.current_dt = dt
        TSS.SystemMatrix = deepcopy(TSS.MassMatrix)
        ExtendableSparse.flush!(TSS.SystemMatrix)
        TSS.SystemMatrix.cscmatrix.nzval .*= (1.0/dt)
        ExtendableSparse.flush!(TSS.StokesMatrix)
        TSS.SystemMatrix.cscmatrix += TSS.StokesMatrix.cscmatrix
        ExtendableSparse.flush!(TSS.SystemMatrix)
    else
        println("    |skipping updating matrix (dt did not change)...")   
    end     

    println("    |updating rhs...")

    # reload linear part
    TSS.rhsvector = TSS.MassMatrix * TSS.current_solution
    TSS.rhsvector .*= (1.0/dt)
    TSS.rhsvector += TSS.datavector

    # nonlinear part
    if (TSS.FE_reconst != TSS.FE_velocity)
        temp = zeros(Float64,size(TSS.ReconstructionMatrix,2))
        assemble_operator!(temp,CELL_NAVIERSTOKES_AdotDAdotDV,TSS.FE_velocity,TSS.FE_velocity,TSS.FE_reconst,TSS.current_solution,TSS.current_solution)
        TSS.rhsvector += TSS.ReconstructionMatrix*temp
    else
        assemble_operator!(TSS.rhsvector,CELL_NAVIERSTOKES_AdotDAdotDV,TSS.FE_velocity,TSS.FE_velocity,TSS.FE_velocity,TSS.current_solution,TSS.current_solution)
    end    
    
    #println("    |apply boundary data...")
    for i = 1 : length(TSS.bdofs)
        TSS.SystemMatrix[TSS.bdofs[i],TSS.bdofs[i]] = TSS.dirichlet_penalty;
        TSS.rhsvector[TSS.bdofs[i]] = TSS.current_solution[TSS.bdofs[i]]*TSS.dirichlet_penalty;
    end

    # solve for next time step
    # todo: reuse LU decomposition of matrix
    print("    |solving...")
    TSS.last_solution[:] = TSS.current_solution[:]
    TSS.current_solution = TSS.SystemMatrix\TSS.rhsvector

    # compute residual (exclude bdofs)
    TSS.rhsvector = TSS.SystemMatrix*TSS.current_solution - TSS.rhsvector
    TSS.rhsvector[TSS.bdofs] .= 0
    residual = norm(TSS.rhsvector);
    println(" (residual=" * string(residual) *")")

    TSS.current_time += dt

    return residual;
end


end
