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


# COMPRESSIBLE STOKES operators
include("FEoperators/CELL_FdotRHOdotV.jl"); # gravity rhs

mutable struct CompressibleStokesProblemDescription
    name::String
    time_dependent_data:: Bool
    shear_modulus:: Float64
    use_symmetric_gradient:: Bool
    lambda:: Float64
    total_mass:: Float64
    equation_of_state:: Function
    volumedata4region:: Vector{Function}
    quadorder4region:: Vector{Int64}
    gravity:: Function
    quadorder4gravity:: Int64 # -1 means not defined
    boundarydata4bregion:: Vector{Function}
    boundarytype4bregion:: Vector{Int64}
    quadorder4bregion:: Vector{Int64}
    CompressibleStokesProblemDescription() = new("undefined compressible Stokes problem", false, 1.0, false,-2.0/3.0,1.0)
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
    GradGradMatrix::ExtendableSparseMatrix
    DivPressureMatrix::ExtendableSparseMatrix
    DivDivMatrix::ExtendableSparseMatrix
    GravityMatrix::ExtendableSparseMatrix
    MassMatrixV::ExtendableSparseMatrix
    MassMatrixD::ExtendableSparseMatrix
    bdofs::Vector{Int64}
    dirichlet_penalty::Float64
    datavector::Vector{Float64}
    NormalFluxMatrix::ExtendableSparseMatrix
    last_velocity::Vector{Float64}
    last_density::Vector{Float64}
    last_pressure::Vector{Float64}
    current_solution::Vector{Float64} # contains velocity and density
    current_time::Real
    current_iteration::Int64
    current_dt::Float64
    SystemMatrixV::ExtendableSparseMatrix
    SystemMatrixD::ExtendableSparseMatrix
    fluxes::Vector{Float64}
    rhsvectorV::Vector{Float64}
    rhsvectorD::Vector{Float64}
end



# COMPRESSIBLE STOKES operators
# we need operators FdotAdotV, 

function setupCompressibleStokesSolver(PD::CompressibleStokesProblemDescription, FE_velocity::FiniteElements.AbstractFiniteElement,FE_densitypressure::FiniteElements.AbstractFiniteElement, initial_velocity::Array, initial_density::Array, reconst_variant::Int = 0, dirichlet_penalty::Float64 = 1e60, symmetry_penalty::Float64 = 1e10)
    @assert typeof(FE_densitypressure) == FiniteElements.L2P0FiniteElement{Float64,1}

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

    Grid.ensure_cells4faces!(FE_velocity.grid)
    Grid.ensure_signs4cells!(FE_velocity.grid)
    nfaces = size(FE_velocity.grid.nodes4faces,1)
    
    # assemble system 
    @time begin
        # compute Stokes operator
        print("    |assembling Stokes operator...") # needs to be replaced by symmetric gradients later
        A = ExtendableSparseMatrix{Float64,Int64}(ndofs_velocity,ndofs_velocity)
        B = ExtendableSparseMatrix{Float64,Int64}(ndofs_velocity,ndofs_densitypressure)
        FESolveStokes.assemble_operator!(A,B,FESolveStokes.CELL_STOKES,FE_velocity,FE_densitypressure, PD.shear_modulus, PD.use_symmetric_gradient);
        println("finished")

        print("    |precomputing normal fluxes of velocity basis...")
        NFM = ExtendableSparseMatrix{Float64,Int64}(nfaces,ndofs_velocity)
        FESolveCommon.assemble_operator!(NFM,FESolveCommon.FACE_1dotVn,FE_velocity);
        println("finished")

        print("    |assembling DivDiv matrix...")
        D = ExtendableSparseMatrix{Float64,Int64}(ndofs_velocity,ndofs_velocity)
        FESolveCommon.assemble_operator!(D,FESolveCommon.CELL_DIVUdotDIVV,FE_velocity,PD.lambda);
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
            print("    |Hdivreconstruction mass matrix...")
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
                print("    |Hdivreconstruction rhs...")
                b = T*b2;
            else
                quadorder = PD.quadorder4bregion[region] + FiniteElements.get_polynomial_order(FE_velocity)
                FESolveCommon.assemble_operator!(b, FESolveCommon.CELL_FdotV, FE_velocity, PD.volumedata4region[region], quadorder)
            end    
            println("finished")
        end
    end

    # add gravity
    G = ExtendableSparseMatrix{Float64,Int64}(ndofs_velocity,ndofs_densitypressure)
    if PD.quadorder4gravity > -1
        G = ExtendableSparseMatrix{Float64,Int64}(ndofs_velocity,ndofs_densitypressure)
        @time begin
            # compute gravity matrix
            print("    |assembling gravity matrix...")
            if reconst_variant > 0
                    G2 = ExtendableSparseMatrix{Float64,Int64}(ndofsHdiv,ndofs_densitypressure)
                    assemble_operator!(G2, CELL_FdotRHOdotV, FE_Reconstruction, FE_densitypressure, PD.gravity, PD.quadorder4gravity)
                    println("finished")
                    print("    |Hdivreconstruction gravity matrix...")
                    ExtendableSparse.flush!(G2)
                    G.cscmatrix = T.cscmatrix*G2.cscmatrix;
            else
                assemble_operator!(G, CELL_FdotRHOdotV, FE_velocity, FE_densitypressure, PD.gravity, PD.quadorder4gravity)
            end    
            println("finished")
        end
    else
        print("    |skipping gravity matrix (not defined)...")
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
    fluxes = zeros(Float64,nfaces)
    return CompressibleStokesSolver(PD,FE_velocity,FE_densitypressure,A,B,D,G,Mv,Mp,bdofs,dirichlet_penalty,b,NFM,initial_velocity,initial_density,initial_density,val4dofs,0.0,0,-999,SystemMatrixV,SystemMatrixD,fluxes,rhsvectorV,rhsvectorD)
end

function PerformTimeStep(CSS::CompressibleStokesSolver, dt::Real = 1 // 10)
    @assert CSS.ProblemData.time_dependent_data == false # true case has to be implemented

    # update matrix and right-hand side vector

    println("    |time = ", CSS.current_time)
    println("    |dt = ", dt)
    if CSS.current_dt != dt
        print("    |updating velocity matrix...")
        CSS.current_dt = dt
        CSS.SystemMatrixV = deepcopy(CSS.MassMatrixV)
        ExtendableSparse.flush!(CSS.SystemMatrixV)
        CSS.SystemMatrixV.cscmatrix.nzval .*= (1.0/dt)
        ExtendableSparse.flush!(CSS.GradGradMatrix)
        CSS.SystemMatrixV.cscmatrix += CSS.GradGradMatrix.cscmatrix
        ExtendableSparse.flush!(CSS.DivDivMatrix)
        CSS.SystemMatrixV.cscmatrix += CSS.DivDivMatrix.cscmatrix
        ExtendableSparse.flush!(CSS.SystemMatrixV)
        println("finished")
    else
        println("    |skipping updating matrix (dt did not change)...")   
    end     
    
    ndofs_velocity = FiniteElements.get_ndofs(CSS.FE_velocity)
    ndofs_densitypressure = FiniteElements.get_ndofs(CSS.FE_densitypressure)
    CSS.last_velocity = CSS.current_solution[1:ndofs_velocity]


    print("    |updating density matrix (upwinding)...")
    # time derivative
    CSS.SystemMatrixD = deepcopy(CSS.MassMatrixD)
    ExtendableSparse.flush!(CSS.SystemMatrixD)
    CSS.SystemMatrixD.cscmatrix.nzval .*= (1.0/dt)

    # upwinding div(rho*u) = 0
    CSS.fluxes = CSS.NormalFluxMatrix * CSS.last_velocity
    # test divergence
    # Base.show(sum(fluxes[CSS.FE_velocity.grid.faces4cells].*CSS.FE_velocity.grid.signs4cells,dims = 2))
    flux = 0.0
    for cell=1:length(CSS.last_density)
        for j=1:3
            face = CSS.FE_velocity.grid.faces4cells[cell,j]
            other_cell = setdiff(CSS.FE_velocity.grid.cells4faces[face,:],cell)[1]
            coeff = (other_cell == 0) ? 1.0 : 0.5
            flux = CSS.fluxes[face]*CSS.FE_velocity.grid.signs4cells[cell,j]
            if flux > 0
                CSS.SystemMatrixD[cell,cell] += coeff*flux
                if other_cell > 0
                    CSS.SystemMatrixD[other_cell,cell] -= coeff*flux
                end    
            else   
                if other_cell > 0
                    CSS.SystemMatrixD[other_cell,other_cell] -= coeff*flux
                    CSS.SystemMatrixD[cell,other_cell] += coeff*flux
                end 

            end
        end
    end
    println("finished")


    print("    |updating rhs for density...")
    CSS.last_density = CSS.current_solution[ndofs_velocity+1:ndofs_velocity+ndofs_densitypressure]
    CSS.rhsvectorD = CSS.MassMatrixD * CSS.last_density
    CSS.rhsvectorD .*= (1.0/dt)
    println("finished")


    print("    |solving for new density...")
    CSS.current_solution[ndofs_velocity+1:ndofs_velocity+ndofs_densitypressure] = CSS.SystemMatrixD\CSS.rhsvectorD
    println("finished")
    changeD = norm(CSS.last_density - CSS.current_solution[ndofs_velocity+1:ndofs_velocity+ndofs_densitypressure]);
    println("    |change_density=", changeD)
    mass =  sum(CSS.current_solution[ndofs_velocity+1:ndofs_velocity+ndofs_densitypressure] .* CSS.FE_densitypressure.grid.volume4cells, dims = 1)
    println("    |total_mass=", mass)


    print("    |updating pressure by equation of state...")
    CSS.ProblemData.equation_of_state(CSS.last_pressure,CSS.current_solution[ndofs_velocity+1:ndofs_velocity+ndofs_densitypressure])
    println("finished")
    
    print("    |updating rhs for velocity...")
    fill!(CSS.rhsvectorV,0.0)

    # add time derivative of velocity
    CSS.rhsvectorV = CSS.MassMatrixV * CSS.last_velocity
    CSS.rhsvectorV .*= (1.0/dt)

    # add f * V
    CSS.rhsvectorV += CSS.datavector

    # add pressure gradient (of eqs'ed density) to rhs
    CSS.rhsvectorV -= CSS.DivPressureMatrix * CSS.last_pressure

    # add gravity
    if CSS.ProblemData.quadorder4gravity > -1
        CSS.rhsvectorV += CSS.GravityMatrix * CSS.current_solution[ndofs_velocity+1:ndofs_velocity+ndofs_densitypressure]
    end

    println("finished")
    
    print("    |apply boundary data...")
    for i = 1 : length(CSS.bdofs)
        CSS.SystemMatrixV[CSS.bdofs[i],CSS.bdofs[i]] = CSS.dirichlet_penalty;
        CSS.rhsvectorV[CSS.bdofs[i]] = CSS.last_velocity[CSS.bdofs[i]]*CSS.dirichlet_penalty;
    end
    println("finished")

    # solve for next time step
    # todo: reuse LU decomposition of matrix
    print("    |solving for new velocity...")
    CSS.current_solution[1:ndofs_velocity] = CSS.SystemMatrixV\CSS.rhsvectorV
    println("finished")

    changeV = norm(CSS.last_velocity - CSS.current_solution[1:ndofs_velocity]);
    println("    |change_velocity=", changeV)

    CSS.current_time += dt


    return changeV+changeD;
end




end
