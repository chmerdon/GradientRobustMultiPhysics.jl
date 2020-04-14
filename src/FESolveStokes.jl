module FESolveStokes

export solveStokesProblem!, solveStokesProblem_iterative!

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
    time_dependent_data:: Bool
    viscosity:: Float64
    volumedata4region:: Vector{Function}
    quadorder4region:: Vector{Int64}
    boundarydata4bregion:: Vector{Function}
    boundarytype4bregion:: Vector{Int64}
    quadorder4bregion:: Vector{Int64}
    StokesProblemDescription() = new("undefined Stokes problem", false, 1.0)
end

function show(PD::StokesProblemDescription)

	println("StokesProblem description");
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




# STOKES operator
include("FEoperators/CELL_STOKES.jl");
include("FEoperators/CELL_DIVFREE_UdotV.jl");

function computeDivFreeBestApproximation!(val4dofs::Array, volume_data!::Function,boundary_data!, FE_velocity::FiniteElements.AbstractFiniteElement, FE_pressure::FiniteElements.AbstractFiniteElement, quadorder::Int = 1, dirichlet_penalty::Float64 = 1e60, pressure_penalty::Float64 = 1e60)
        
    ndofs_velocity = FiniteElements.get_ndofs(FE_velocity);
    ndofs_pressure = FiniteElements.get_ndofs(FE_pressure);
    ndofs = ndofs_velocity + ndofs_pressure;
    
    println("\nSOLVING DIVFREE-BESTAPPROXIMATION PROBLEM")
    println(" |FEvelocity = " * FE_velocity.name)
    println(" |FEpressure = " * FE_pressure.name)
    println(" |totalndofs = ", ndofs)
    println(" |");
    println(" |PROGRESS")
    
    # assemble system 
    @time begin
        # compute Stokes operator
        print("    |assembling matrix...")
        A = ExtendableSparseMatrix{Float64,Int64}(ndofs,ndofs) # +1 due to Lagrange multiplier for integral mean
        assemble_operator!(A,CELL_DIVFREE_UdotV,FE_velocity,FE_pressure);
    end
        
    @time begin
        # compute right-hand side vector
        b = zeros(Float64,ndofs);
        print("    |assembling rhs...")
        quadorder = quadorder + FiniteElements.get_polynomial_order(FE_velocity)
        FESolveCommon.assemble_operator!(b, FESolveCommon.CELL_FdotV, FE_velocity, volume_data!, quadorder)
        println("finished")
    end

    @time begin
        # compute and apply boundary data
        if boundary_data! != Nothing
            println("    |incorporating boundary data (full Dirichlet)...")
            bdofs = FESolveCommon.computeDirichletBoundaryData!(val4dofs,FE_velocity,boundary_data!,quadorder);
            println("     ...finished")
        end
    end
    
    # fix one pressure value
    #A[ndofs,ndofs] = pressure_penalty;
    #b[ndofs] = 1;
       
    # solve
    val4dofs[1:ndofs] = A\b;
    
    # compute residual (exclude bdofs)
    residual = A*val4dofs[1:ndofs] - b
    residual[bdofs] .= 0
    residual[ndofs] = 0
    residual = norm(residual);
    println("    |residual=", residual)
    
    return residual
end

function solveStokesProblem!(val4dofs::Array,PD::StokesProblemDescription, FE_velocity::FiniteElements.AbstractFiniteElement,FE_pressure::FiniteElements.AbstractFiniteElement; reconst_variant::Int = 0, dirichlet_penalty::Float64 = 1e60, symmetry_penalty::Float64 = 1e10)
        
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
        # compute Stokes operator
        print("    |assembling matrix...")
        A = ExtendableSparseMatrix{Float64,Int64}(ndofs,ndofs) # +1 due to Lagrange multiplier for integral mean
        assemble_operator!(A,CELL_STOKES,FE_velocity,FE_pressure,PD.viscosity);

        # compute integrals of pressure dofs
        pm = zeros(Float64,ndofs_pressure,1)
        FESolveCommon.assemble_operator!(pm,FESolveCommon.DOMAIN_1dotV,FE_pressure);
        println("finished")
    end
        
    @time begin
        # compute right-hand side vector
        b = zeros(Float64,ndofs);
        for region = 1 : length(PD.volumedata4region)
            print("    |assembling rhs in region $region...")
            if reconst_variant > 0
                ET = FE_velocity.grid.elemtypes[1];
                FE_Reconstruction = FiniteElements.get_Hdivreconstruction_space(FE_velocity, ET, reconst_variant);
                ndofsHdiv = FiniteElements.get_ndofs(FE_Reconstruction)
                b2 = zeros(Float64,ndofsHdiv);
                quadorder = PD.quadorder4bregion[region] + FiniteElements.get_polynomial_order(FE_Reconstruction)
                FESolveCommon.assemble_operator!(b2, FESolveCommon.CELL_FdotV, FE_Reconstruction, PD.volumedata4region[region], quadorder)
                println("finished")
                print("    |Hdivreconstruction...")
                T = ExtendableSparseMatrix{Float64,Int64}(ndofs_velocity,ndofsHdiv)
                FiniteElements.get_Hdivreconstruction_trafo!(T,FE_velocity,FE_Reconstruction);
                b[1:ndofs_velocity] = T*b2;
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
        for i = 1 : length(bdofs)
           A[bdofs[i],bdofs[i]] = dirichlet_penalty;
           b[bdofs[i]] = val4dofs[bdofs[i]]*dirichlet_penalty;
        end
        Dsymboundary_ids = findall(x->x == 3, PD.boundarytype4bregion)
        if length(Dnboundary_ids) > 0
            print("       Symmetry : ")
            Base.show(Dsymboundary_ids); println("");
            FESolveCommon.assemble_operator!(A, FESolveCommon.BFACE_UndotVn, FE_velocity, Dsymboundary_ids, symmetry_penalty)
        end    
        println("     ...finished")
    end
    
    # fix one pressure value
    if length(Dnboundary_ids) == 0
        append!(bdofs,ndofs)
    end    

    # solve
    val4dofs[:] = A\b;
    
    # move pressure mean to zero
    if length(Dnboundary_ids) == 0
        domain_area = sum(FE_pressure.grid.volume4cells[:])
        mean = dot(pm[:],val4dofs[ndofs_velocity+1:end])/domain_area
        val4dofs[ndofs_velocity+1:end] .-= mean;
    end
    
    # compute residual (exclude bdofs)
    residual = A*val4dofs - b
    residual[bdofs] .= 0
    residual[ndofs] = 0
    residual = norm(residual);
    println("    |residual=", residual)
    
    return residual
end

function solveStokesProblem_iterative!(val4dofs::Array,PD::StokesProblemDescription, FE_velocity::FiniteElements.AbstractFiniteElement,FE_pressure::FiniteElements.AbstractFiniteElement, reconst_variant::Int = 0, divergence_penalty::Float64 = 1e6 , dirichlet_penalty::Float64 = 1e30)
        
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
        # compute Stokes operator
        print("    |assembling matrix...")
        A0 = ExtendableSparseMatrix{Float64,Int64}(ndofs_velocity,ndofs_velocity)
        B = ExtendableSparseMatrix{Float64,Int64}(ndofs_velocity,ndofs_pressure)
        C = ExtendableSparseMatrix{Float64,Int64}(ndofs_velocity,ndofs_velocity)
        M = ExtendableSparseMatrix{Float64,Int64}(ndofs_pressure,ndofs_pressure)
        assemble_operator!(A0,B,CELL_STOKES,FE_velocity,FE_pressure,PD.viscosity);
        FESolveCommon.assemble_operator!(C,FESolveCommon.CELL_DIVUdotDIVV,FE_velocity);
        FESolveCommon.assemble_operator!(M,FESolveCommon.CELL_UdotV,FE_pressure);

        # compute integrals of pressure dofs
        pm = zeros(Float64,ndofs_pressure,1)
        FESolveCommon.assemble_operator!(pm,FESolveCommon.DOMAIN_1dotV,FE_pressure);
        println("finished")
    end
        
    @time begin
        # compute right-hand side vector
        b0 = zeros(Float64,ndofs_velocity);
        for region = 1 : length(PD.volumedata4region)
            print("    |assembling rhs in region $region...")
            if reconst_variant > 0
                ET = FE_velocity.grid.elemtypes[1];
                FE_Reconstruction = FiniteElements.get_Hdivreconstruction_space(FE_velocity, ET, reconst_variant);
                ndofsHdiv = FiniteElements.get_ndofs(FE_Reconstruction)
                b2 = zeros(Float64,ndofsHdiv);
                quadorder = PD.quadorder4bregion[region] + FiniteElements.get_polynomial_order(FE_Reconstruction)
                FESolveCommon.assemble_operator!(b2, FESolveCommon.CELL_FdotV, FE_Reconstruction, PD.volumedata4region[region], quadorder)
                println("finished")
                print("    |Hdivreconstruction...")
                T = ExtendableSparseMatrix{Float64,Int64}(ndofs_velocity,ndofsHdiv)
                FiniteElements.get_Hdivreconstruction_trafo!(T,FE_velocity,FE_Reconstruction);
                b0[1:ndofs_velocity] = T*b2;
            else
                quadorder = PD.quadorder4bregion[region] + FiniteElements.get_polynomial_order(FE_velocity)
                FESolveCommon.assemble_operator!(b0, FESolveCommon.CELL_FdotV, FE_velocity, PD.volumedata4region[region], quadorder)
            end    
            println("finished")
        end
    end

    @time begin
        # compute boundary data
        print("    |boundary data...")
        bdofs = FESolveCommon.computeDirichletBoundaryData!(val4dofs,FE_velocity,PD.boundarydata4bregion[1],true,PD.quadorder4bregion[1]);
        println("finished")
    end
    
    residual = 1.0
    b = zeros(Float64,ndofs_velocity);
    A = ExtendableSparseMatrix{Float64,Int64}(ndofs_velocity,ndofs_velocity)
    residual_momentum = zeros(Float64,ndofs_velocity);
    residual_divergence = zeros(Float64,ndofs_velocity);
    pressure = zeros(Float64,ndofs_pressure);
    maxIterations = 10
    velocity = zeros(Float64,ndofs_velocity);

    while ((residual > 1e-12) && (maxIterations > 0))
        maxIterations -= 1
        divergence_penalty /= 4

        # solve vector Laplacian for right-hand side b - nabla p_h
        A = A0 + divergence_penalty*C
        b = b0 - B*pressure

        # apply boundary data       
        for i = 1 : length(bdofs)
            A[bdofs[i],bdofs[i]] = dirichlet_penalty;
            b[bdofs[i]] = val4dofs[bdofs[i]]*dirichlet_penalty;
        end
        velocity = A\b
        residual_divergence = B'*velocity
        println("    |residual_divergence=", norm(residual_divergence))


        # update pressure by p_h = p_h - \pi_Q(div u_h)
        pressure += divergence_penalty*(M\(B'*velocity))

        # compute residual (exclude bdofs)
        residual_momentum = A*velocity - b0 + B*pressure
        residual_momentum[bdofs] .= 0
        println("    |residual_momentum=", norm(residual_momentum))
    end    
    residual = sqrt(norm(residual_momentum)^2 + norm(residual_divergence)^2);
    val4dofs[1:ndofs_velocity] = velocity
    val4dofs[ndofs_velocity+1:end] = pressure

    # move pressure mean to zero
    domain_area = sum(FE_pressure.grid.volume4cells[:])
    mean = dot(pm[:],val4dofs[ndofs_velocity+1:end])/domain_area
    val4dofs[ndofs_velocity+1:end] .-= mean;
    
    return residual
end




mutable struct TransientStokesSolver
    ProblemData::StokesProblemDescription
    FE_velocity::AbstractFiniteElement
    FE_pressure::AbstractFiniteElement
    FE_reconst::AbstractFiniteElement
    StokesMatrix::ExtendableSparseMatrix
    MassMatrix::ExtendableSparseMatrix
    ReconstructionMatrix::ExtendableSparseMatrix
    bdofs::Vector{Int64}
    dirichlet_penalty::Float64
    datavector::Vector{Float64}
    last_solution::Vector{Float64}
    current_solution::Vector{Float64}
    current_time::Real
    current_iteration::Int64
    current_dt::Float64
    SystemMatrix::ExtendableSparseMatrix
    rhsvector::Vector{Float64}
end


function setupTransientStokesSolver(PD::StokesProblemDescription, FE_velocity::FiniteElements.AbstractFiniteElement,FE_pressure::FiniteElements.AbstractFiniteElement, initial_solution::Array, reconst_variant::Int = 0, dirichlet_penalty::Float64 = 1e60, symmetry_penalty::Float64 = 1e10)
        
    ndofs_velocity = FiniteElements.get_ndofs(FE_velocity);
    ndofs_pressure = FiniteElements.get_ndofs(FE_pressure);
    ndofs = ndofs_velocity + ndofs_pressure;
    val4dofs = zeros(Float64,ndofs);
    val4dofs[:] = initial_solution[:];
    
    println("\nSETUP TRANSIENT STOKES PROBLEM")
    println(" |FEvelocity = " * FE_velocity.name)
    println(" |FEpressure = " * FE_pressure.name)
    println(" |totalndofs = ", ndofs)
    println(" |");
    println(" |PROGRESS")
    
    # assemble system 
    @time begin
        # compute Stokes operator
        print("    |assembling Stokes matrix...")
        A = ExtendableSparseMatrix{Float64,Int64}(ndofs,ndofs)
        assemble_operator!(A,CELL_STOKES,FE_velocity,FE_pressure,PD.viscosity);
        println("finished")

        # compute mass matrixif reconst_variant > 0
        print("    |assembling mass matrix...")
        M = ExtendableSparseMatrix{Float64,Int64}(ndofs,ndofs)
        if reconst_variant > 0
            ET = FE_velocity.grid.elemtypes[1];
            FE_Reconstruction = FiniteElements.get_Hdivreconstruction_space(FE_velocity, ET, reconst_variant);
            ndofsHdiv = FiniteElements.get_ndofs(FE_Reconstruction)
            Mhdiv = ExtendableSparseMatrix{Float64,Int64}(ndofsHdiv,ndofsHdiv)
            FESolveCommon.assemble_operator!(Mhdiv,FESolveCommon.CELL_UdotV,FE_Reconstruction);
            println("finished")
            print("    |Hdivreconstruction...")
            T = ExtendableSparseMatrix{Float64,Int64}(ndofs,ndofsHdiv)
            FiniteElements.get_Hdivreconstruction_trafo!(T,FE_velocity,FE_Reconstruction);
            ExtendableSparse.flush!(Mhdiv)
            ExtendableSparse.flush!(T)
            Mtemp = T.cscmatrix*Mhdiv.cscmatrix
            M.cscmatrix = Mtemp*T.cscmatrix'
        else
            FE_Reconstruction = FE_velocity
            T = ExtendableSparseMatrix{Float64,Int64}(0,0)
            FESolveCommon.assemble_operator!(M,FESolveCommon.CELL_UdotV,FE_velocity);
            println("finished")
        end    

        # compute integrals of pressure dofs
        pm = zeros(Float64,ndofs_pressure,1)
        FESolveCommon.assemble_operator!(pm,FESolveCommon.DOMAIN_1dotV,FE_pressure);
        println("finished")
    end
        
    @time begin
        # compute right-hand side vector
        b = zeros(Float64,ndofs);
        for region = 1 : length(PD.volumedata4region)
            print("    |assembling rhs in region $region...")
            if reconst_variant > 0
                ET = FE_velocity.grid.elemtypes[1];
                FE_Reconstruction = FiniteElements.get_Hdivreconstruction_space(FE_velocity, ET, reconst_variant);
                ndofsHdiv = FiniteElements.get_ndofs(FE_Reconstruction)
                b2 = zeros(Float64,ndofsHdiv);
                quadorder = PD.quadorder4bregion[region] + FiniteElements.get_polynomial_order(FE_Reconstruction)
                FESolveCommon.assemble_operator!(b2, FESolveCommon.CELL_FdotV, FE_Reconstruction, PD.volumedata4region[region], quadorder)
                println("finished")
                print("    |Hdivreconstruction...")
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
        for i = 1 : length(bdofs)
           A[bdofs[i],bdofs[i]] = dirichlet_penalty;
           b[bdofs[i]] = val4dofs[bdofs[i]]*dirichlet_penalty;
        end
        Dsymboundary_ids = findall(x->x == 3, PD.boundarytype4bregion)
        if length(Dnboundary_ids) > 0
            print("       Symmetry : ")
            Base.show(Dsymboundary_ids); println("");
            FESolveCommon.assemble_operator!(A, FESolveCommon.BFACE_UndotVn, FE_velocity, Dsymboundary_ids, symmetry_penalty)
        end    
        println("     ...finished")
    end
    
    # fix one pressure value
    if length(Dnboundary_ids) == 0
        append!(bdofs,ndofs);
    end    

    rhsvector = zeros(Float64,ndofs)
    SystemMatrix = ExtendableSparseMatrix{Float64,Int64}(ndofs,ndofs)
    return TransientStokesSolver(PD,FE_velocity,FE_pressure,FE_Reconstruction,A,M,T,bdofs,dirichlet_penalty,b,initial_solution,val4dofs,0.0,0,-999,SystemMatrix,rhsvector)
end

function PerformTimeStep(TSS::TransientStokesSolver, dt::Real = 1 // 10)
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
    TSS.rhsvector = TSS.MassMatrix * TSS.current_solution
    TSS.rhsvector .*= (1.0/dt)
    TSS.rhsvector += TSS.datavector
    
    # println("    |apply boundary data...")
    for i = 1 : length(TSS.bdofs)
        TSS.SystemMatrix[TSS.bdofs[i],TSS.bdofs[i]] = TSS.dirichlet_penalty;
        TSS.rhsvector[TSS.bdofs[i]] = TSS.current_solution[TSS.bdofs[i]]*TSS.dirichlet_penalty;
    end

    # solve for next time step
    # todo: reuse LU decomposition of matrix
    print("    |solving...")
    TSS.last_solution = deepcopy(TSS.current_solution)
    TSS.current_solution = TSS.SystemMatrix\TSS.rhsvector

    #change = norm(TSS.last_solution - TSS.current_solution);
    #println("    |change=", change)

    # compute residual (exclude bdofs)
    TSS.rhsvector = TSS.SystemMatrix*TSS.current_solution - TSS.rhsvector
    TSS.rhsvector[TSS.bdofs] .= 0
    residual = norm(TSS.rhsvector);
    println(" (residual=" * string(residual) *")")

    TSS.current_time += dt

    return residual;
end




end
