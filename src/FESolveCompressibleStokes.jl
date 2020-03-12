module FESolveCompressibleStokes

export solveCompressibleStokesProblem!

using ExtendableSparse
using LinearAlgebra
using BenchmarkTools
using FiniteElements
using FESolveCommon
using FESolveStokes
using FESolveNavierStokes
using Grid
using Quadrature

mutable struct CompressibleStokesProblemDescription
    name::String
    time_dependent_data:: Bool
    viscosity:: Float64
    total_mass:: Float64
    equation_of_state:: Function
    volumedata4region:: Vector{Function}
    quadorder4region:: Vector{Int64}
    boundarydata4bregion:: Vector{Function}
    boundarytype4bregion:: Vector{Int64}
    quadorder4bregion:: Vector{Int64}
    CompressibleStokesProblemDescription() = new("undefined compressible Stokes problem", false, 1.0, 1.0)
end

function show(PD::CompressibleStokesProblemDescription)

	println("CompressibleStokesProblem description");
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

mutable struct CompressibleStokesSolver
    ProblemData::CompressibleStokesProblemDescription
    FE_velocity::AbstractFiniteElement
    FE_densitypressure::AbstractFiniteElement
    StokesMatrix::ExtendableSparseMatrix
    MassMatrixV::ExtendableSparseMatrix
    MassMatrixD::ExtendableSparseMatrix
    bdofs::Vector{Int64}
    dirichlet_penalty::Float64
    datavector::Vector{Float64}
    last_velocity::Vector{Float64}
    last_density::Vector{Float64}
    current_solution::Vector{Float64} # contains velocity and density
    current_time::Real
    current_iteration::Int64
    current_dt::Float64
    SystemMatrixV::ExtendableSparseMatrix
    SystemMatrixD::ExtendableSparseMatrix
    rhsvectorV::Vector{Float64}
    rhsvectorD::Vector{Float64}
end



# COMPRESSIBLE STOKES operators
# we need operators FdotAdotV, 

function setupCompressibleStokesSolver(PD::CompressibleStokesProblemDescription, FE_velocity::FiniteElements.AbstractFiniteElement,FE_densitypressure::FiniteElements.AbstractFiniteElement, initial_velocity::Array, initial_density::Array, reconst_variant::Int = 0, dirichlet_penalty::Float64 = 1e60, symmetry_penalty::Float64 = 1e10)
        
    ndofs_velocity = FiniteElements.get_ndofs(FE_velocity);
    ndofs_densitypressure = FiniteElements.get_ndofs(FE_densitypressure);
    ndofs = ndofs_velocity + 2*ndofs_densitypressure;
    val4dofs = zeros(Float64,ndofs);
    val4dofs[1:ndofs_velocity] = initial_velocity;
    val4dofs[ndofs_velocity+1:ndofs_velocity+ndofs_densitypressure] = initial_density;
    
    println("\nSETUP COMPRESSIBLE STOKES PROBLEM")
    println(" |FEvelocity = " * FE_velocity.name)
    println(" |FEdensity = " * FE_densitypressure.name)
    println(" |FEpressure = " * FE_densitypressure.name)
    println(" |totalndofs = ", ndofs)
    println(" |");
    println(" |PROGRESS")
    
    # assemble system 
    @time begin
        # compute Stokes operator
        print("    |assembling Laplacian...")
        A = ExtendableSparseMatrix{Float64,Int64}(ndofs_velocity,ndofs_velocity)
        FESolveCommon.assemble_operator!(A,FESolveCommon.CELL_DUdotDV,FE_velocity,PD.viscosity);
        println("finished")

        # compute mass matrix
        print("    |assembling mass matrix for pressure/density...")
        Mp = ExtendableSparseMatrix{Float64,Int64}(ndofs_densitypressure,ndofs_densitypressure)
        FESolveCommon.assemble_operator!(Mp,FESolveCommon.CELL_UdotV,FE_densitypressure);
        println("finished")
        print("    |assembling mass matrix for velocity...")
        Mv = ExtendableSparseMatrix{Float64,Int64}(ndofs_velocity,ndofs_velocity)
        if reconst_variant > 0
            @assert FiniteElements.Hdivreconstruction_available(FE_velocity)
            FE_Reconstruction = FiniteElements.get_Hdivreconstruction_space(FE_velocity, reconst_variant);
            ndofsHdiv = FiniteElements.get_ndofs(FE_Reconstruction)
            Mhdiv = ExtendableSparseMatrix{Float64,Int64}(ndofsHdiv,ndofsHdiv)
            FESolveCommon.assemble_operator!(Mhdiv,FESolveCommon.CELL_UdotV,FE_Reconstruction);
            println("finished")
            print("    |Hdivreconstruction...")
            T = ExtendableSparseMatrix{Float64,Int64}(ndofs_velocity,ndofsHdiv)
            FiniteElements.get_Hdivreconstruction_trafo!(T,FE_velocity,FE_Reconstruction);
            ExtendableSparse.flush!(Mhdiv)
            ExtendableSparse.flush!(T)
            Mtemp = T.cscmatrix*Mhdiv.cscmatrix
            Mv.cscmatrix = Mtemp*T.cscmatrix'
        else
            FESolveCommon.assemble_operator!(Mv,FESolveCommon.CELL_UdotV,FE_velocity);
            println("finished")
        end    
    end
        
    @time begin
        # compute right-hand side vector
        b = zeros(Float64,ndofs_velocity);
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

    rhsvectorV = zeros(Float64,ndofs_velocity)
    SystemMatrixV = ExtendableSparseMatrix{Float64,Int64}(ndofs_velocity,ndofs_velocity)
    rhsvectorD = zeros(Float64,ndofs_densitypressure)
    SystemMatrixD = ExtendableSparseMatrix{Float64,Int64}(ndofs_densitypressure,ndofs_densitypressure)
    return CompressibleStokesSolver(PD,FE_velocity,FE_densitypressure,A,Mv,Mp,bdofs,dirichlet_penalty,b,initial_velocity,initial_density,val4dofs,0.0,0,-999,SystemMatrixV,SystemMatrixD,rhsvectorV,rhsvectorD)
end

function PerformTimeStep(TSS::CompressibleStokesSolver, dt::Real = 1 // 10)
    @assert TSS.ProblemData.time_dependent_data == false # true case has to be implemented

    # update matrix and right-hand side vector

    println("    |time = ", TSS.current_time)
    println("    |dt = ", TSS.current_dt)
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
    
    println("    |apply boundary data...")
    for i = 1 : length(TSS.bdofs)
        TSS.SystemMatrix[TSS.bdofs[i],TSS.bdofs[i]] = TSS.dirichlet_penalty;
        TSS.rhsvector[TSS.bdofs[i]] = TSS.current_solution[TSS.bdofs[i]]*TSS.dirichlet_penalty;
    end

    # solve for next time step
    # todo: reuse LU decomposition of matrix
    println("    |solving...")
    TSS.last_solution = deepcopy(TSS.current_solution)
    TSS.current_solution = TSS.SystemMatrix\TSS.rhsvector

    change = norm(TSS.last_solution - TSS.current_solution);
    println("    |change=", change)

    # compute residual (exclude bdofs)
    TSS.rhsvector = TSS.SystemMatrix*TSS.current_solution - TSS.rhsvector
    TSS.rhsvector[TSS.bdofs] .= 0
    residual = norm(TSS.rhsvector);
    println("    |residual=", residual)

    TSS.current_time += dt

    return residual;
end




end
