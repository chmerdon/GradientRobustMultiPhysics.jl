module FESolveCompressibleStokes

export solveCompressibleStokesProblem!

using ExtendableSparse
using SparseArrays
using LinearAlgebra
using BenchmarkTools
using FiniteElements
using FESolveCommon
using FESolveStokes
using FESolveNavierStokes
using Grid
using Quadrature
using VTKView


# COMPRESSIBLE STOKES operators
include("FEoperators/CELL_RHOdotUdotV.jl"); # gravity rhs
include("FEoperators/CELL_FdotRHOdotV.jl"); # gravity rhs
include("FEoperators/CELL_NAVIERSTOKES_RHOdotAdotDAdotDV.jl"); # nonlinear IMEX term 4 rhs

mutable struct CompressibleStokesProblemDescription
    name::String
    time_dependent_data:: Bool
    shear_modulus:: Float64
    lambda:: Float64
    total_mass:: Float64
    use_symmetric_gradient:: Bool
    use_nonlinear_convection:: Bool
    equation_of_state:: Function
    volumedata4region:: Vector{Function}
    quadorder4region:: Vector{Int64}
    gravity:: Function
    quadorder4gravity:: Int64 # -1 means not defined
    boundarydata4bregion:: Vector{Function}
    boundarytype4bregion:: Vector{Int64}
    quadorder4bregion:: Vector{Int64}
    CompressibleStokesProblemDescription() = new("undefined compressible Stokes problem", false, 1.0, -2.0/3.0,1.0, false, false)
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
    FE_reconst::AbstractFiniteElement
    GradGradMatrix::ExtendableSparseMatrix
    DivPressureMatrix::ExtendableSparseMatrix
    DivDivMatrix::ExtendableSparseMatrix
    GravityMatrix::ExtendableSparseMatrix
    ReconstMatrix::ExtendableSparseMatrix
    ReconstMassMatrix::ExtendableSparseMatrix
    MassMatrixD::ExtendableSparseMatrix
    bdofs::Vector{Int64}
    dirichlet_penalty::Float64
    datavector::Vector{Float64}
    NormalFluxMatrix::ExtendableSparseMatrix
    last_velocity::Vector{Float64}
    last_density::Vector{Float64}
    current_velocity::Vector{Float64}
    current_density::Vector{Float64}
    current_pressure::Vector{Float64}
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
    velocity = zeros(Float64,ndofs_velocity);
    density = zeros(Float64,ndofs_densitypressure);
    pressure = zeros(Float64,ndofs_densitypressure);
    velocity[:] = initial_velocity;
    density[:] = initial_density;
    PD.equation_of_state(pressure,density);
    
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
        print("    |assembling stiffness matrix and divxpressure matrix...") # needs to be replaced by symmetric gradients later
        A = ExtendableSparseMatrix{Float64,Int64}(ndofs_velocity,ndofs_velocity)
        B = ExtendableSparseMatrix{Float64,Int64}(ndofs_velocity,ndofs_densitypressure)
        if PD.use_symmetric_gradient
            FESolveCommon.assemble_operator!(A,FESolveCommon.CELL_EPSUdotEPSV,FE_velocity, 2*PD.shear_modulus);
        else
            FESolveCommon.assemble_operator!(A,FESolveCommon.CELL_DUdotDV,FE_velocity, 2*PD.shear_modulus);
        end    
        FESolveCommon.assemble_operator!(B,FESolveCommon.CELL_UdotDIVV,FE_densitypressure,FE_velocity);
        println("finished")

        print("    |precomputing normal fluxes of velocity basis...")
        NFM = ExtendableSparseMatrix{Float64,Int64}(nfaces,ndofs_velocity)
        FESolveCommon.assemble_operator!(NFM,FESolveCommon.FACE_1dotVn,FE_velocity);
        println("finished")

        # compute mass matrix
        print("    |assembling mass matrix for pressure/density...")
        Mp = ExtendableSparseMatrix{Float64,Int64}(ndofs_densitypressure,ndofs_densitypressure)
        FESolveCommon.assemble_operator!(Mp,FESolveCommon.CELL_UdotV,FE_densitypressure);
        println("finished")
        if reconst_variant > 0
            @assert FiniteElements.Hdivreconstruction_available(FE_velocity)
            FE_Reconstruction = FiniteElements.get_Hdivreconstruction_space(FE_velocity, reconst_variant);
            ndofsHdiv = FiniteElements.get_ndofs(FE_Reconstruction)
            T = ExtendableSparseMatrix{Float64,Int64}(ndofs_velocity,ndofsHdiv)
            FiniteElements.get_Hdivreconstruction_trafo!(T,FE_velocity,FE_Reconstruction);
            MvR = ExtendableSparseMatrix{Float64,Int64}(ndofs_velocity,ndofsHdiv)
        else
            FE_Reconstruction = FE_velocity;
            T = ExtendableSparseMatrix{Float64,Int64}(0,0)
            MvR = ExtendableSparseMatrix{Float64,Int64}(0,0)
        end    
        println("finished")
        
        if reconst_variant > 0
            # gradient-robust version uses discrete divergence
            # DivhDivh matrix can be computed by B' * Mp^{-1} * B
            # (where Mp luckily is a diagonal matrixfor P0 densities)
            print("    |assembling DivhDivh matrix...")
            D = ExtendableSparseMatrix{Float64,Int64}(ndofs_velocity,ndofs_velocity)
            ExtendableSparse.flush!(B)
            ExtendableSparse.flush!(Mp)
            D.cscmatrix = B.cscmatrix*spdiagm(0 => 1.0./diag(Mp))*B.cscmatrix'
            println("finished")
        else
            print("    |assembling DivDiv matrix...")
            D = ExtendableSparseMatrix{Float64,Int64}(ndofs_velocity,ndofs_velocity)
            FESolveCommon.assemble_operator!(D,FESolveCommon.CELL_DIVUdotDIVV,FE_velocity,PD.lambda);
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
        bdofs = FESolveCommon.computeDirichletBoundaryData!(velocity,FE_velocity,Dbboundary_ids, PD.boundarydata4bregion,true,PD.quadorder4bregion);
        for i = 1 : length(bdofs)
           A[bdofs[i],bdofs[i]] = dirichlet_penalty;
           b[bdofs[i]] = velocity[bdofs[i]]*dirichlet_penalty;
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
    return CompressibleStokesSolver(PD,FE_velocity,FE_densitypressure,FE_Reconstruction,A,B,D,G,T,MvR,Mp,bdofs,dirichlet_penalty,b,NFM,initial_velocity,initial_density,velocity,density,pressure,0.0,0,-999,SystemMatrixV,SystemMatrixD,fluxes,rhsvectorV,rhsvectorD)
end

function PerformTimeStep(CSS::CompressibleStokesSolver, dt::Real = 1 // 10)
    @assert CSS.ProblemData.time_dependent_data == false # true case has to be implemented

    # update matrix and right-hand side vector

    println("    |")
    println("    |time + dt = " * string(CSS.current_time) * " + " * string(dt))
    CSS.current_dt = dt

    ndofs_velocity = FiniteElements.get_ndofs(CSS.FE_velocity)
    ndofs_densitypressure = FiniteElements.get_ndofs(CSS.FE_densitypressure)
    CSS.last_velocity = CSS.current_velocity


    #@time begin
        #print("    |updating density matrix (upwinding)...")
        # time derivative
        CSS.SystemMatrixD = deepcopy(CSS.MassMatrixD)
        ExtendableSparse.flush!(CSS.SystemMatrixD)
        CSS.SystemMatrixD.cscmatrix.nzval .*= (1.0/dt)

        # upwinding div(rho*u) = 0
        CSS.fluxes = CSS.NormalFluxMatrix * CSS.last_velocity
        # test divergence
        #Base.show(sum(CSS.fluxes[CSS.FE_velocity.grid.faces4cells].*CSS.FE_velocity.grid.signs4cells,dims = 2))
        flux = 0.0
        face = 0
        other_cell = 0
        for cell=1:length(CSS.last_density), j=1:3
            face = CSS.FE_velocity.grid.faces4cells[cell,j]
            flux = CSS.fluxes[face]*CSS.FE_velocity.grid.signs4cells[cell,j]
            if CSS.FE_velocity.grid.cells4faces[face,1] == cell
                other_cell = CSS.FE_velocity.grid.cells4faces[face,2]
            else
                other_cell = CSS.FE_velocity.grid.cells4faces[face,1]    
            end
            if (other_cell > 0) 
                flux *= 0.5 
            end       
            if flux > 0
                CSS.SystemMatrixD[cell,cell] += flux
                if other_cell > 0
                    CSS.SystemMatrixD[other_cell,cell] -= flux
                end    
            else   
                if other_cell > 0
                    CSS.SystemMatrixD[other_cell,other_cell] -= flux
                    CSS.SystemMatrixD[cell,other_cell] += flux
                end 
            end
        end
        #println("finished")

        #print("    |updating rhs for density...")
        CSS.last_density = CSS.current_density
        CSS.rhsvectorD = CSS.MassMatrixD * CSS.last_density
        CSS.rhsvectorD .*= (1.0/dt)
        #println("finished")


        print("    |solving for new density...")
        CSS.current_density = CSS.SystemMatrixD\CSS.rhsvectorD
        changeD = norm(CSS.last_density - CSS.current_density);
        println("finished (change=" * string(changeD) * ")")

        #print("    |updating pressure by equation of state...")
        CSS.ProblemData.equation_of_state(CSS.current_pressure,CSS.current_density)
        #println("finished")
        
        #print("    |updating rhs for velocity...")


        #print("    |updating velocity mass matrix...")
        if CSS.FE_reconst != CSS.FE_velocity
            fill!(CSS.ReconstMassMatrix.cscmatrix.nzval,0.0)
            assemble_operator!(CSS.ReconstMassMatrix,CELL_RHOdotUdotV,CSS.FE_velocity,CSS.FE_reconst,CSS.FE_densitypressure,CSS.current_density);
            ExtendableSparse.flush!(CSS.ReconstMassMatrix)
            ExtendableSparse.flush!(CSS.ReconstMatrix)
            CSS.SystemMatrixV.cscmatrix = CSS.ReconstMassMatrix.cscmatrix*CSS.ReconstMatrix.cscmatrix'
        else
            fill!(CSS.SystemMatrixV.cscmatrix.nzval,0.0)
            assemble_operator!(CSS.SystemMatrixV,CELL_RHOdotUdotV,CSS.FE_velocity,CSS.FE_velocity,CSS.FE_densitypressure,CSS.current_density);
        end    
        #println("finished")

        ExtendableSparse.flush!(CSS.SystemMatrixV)
        CSS.SystemMatrixV.cscmatrix.nzval .*= (1.0/dt)
        # up to here the SystemMatrix equals the MassMatrixV
        # add time derivative of velocity to rhs
        CSS.rhsvectorV = CSS.SystemMatrixV * CSS.last_velocity
        CSS.rhsvectorV += CSS.datavector

        # complete SystemMatrixV
        ExtendableSparse.flush!(CSS.GradGradMatrix)
        CSS.SystemMatrixV.cscmatrix += CSS.GradGradMatrix.cscmatrix
        ExtendableSparse.flush!(CSS.DivDivMatrix)
        CSS.SystemMatrixV.cscmatrix += CSS.DivDivMatrix.cscmatrix
        ExtendableSparse.flush!(CSS.SystemMatrixV)

        # add pressure gradient (of eqs'ed density) to rhs
        CSS.rhsvectorV += CSS.DivPressureMatrix * CSS.current_pressure

        # add nonlinear use_nonlinear_convection
        if CSS.ProblemData.use_nonlinear_convection
            assemble_operator!(CSS.rhsvectorV,CELL_NAVIERSTOKES_RHOdotAdotDAdotDV,CSS.FE_velocity,CSS.FE_velocity,CSS.FE_densitypressure,CSS.current_velocity,CSS.current_velocity,CSS.current_density)
        end     
    

        # add gravity
        if CSS.ProblemData.quadorder4gravity > -1
            CSS.rhsvectorV += CSS.GravityMatrix * CSS.current_density
        end
        #println("finished")
        
        #print("    |apply boundary data...")
        for i = 1 : length(CSS.bdofs)
            CSS.SystemMatrixV[CSS.bdofs[i],CSS.bdofs[i]] = CSS.dirichlet_penalty;
            CSS.rhsvectorV[CSS.bdofs[i]] = CSS.last_velocity[CSS.bdofs[i]]*CSS.dirichlet_penalty;
        end
        #println("finished")

        # solve for next time step
        # todo: reuse LU decomposition of matrix
        print("    |solving for new velocity...")
        CSS.current_velocity = CSS.SystemMatrixV\CSS.rhsvectorV
        changeV = norm(CSS.last_velocity - CSS.current_velocity);
        println("finished (change=" * string(changeV) * ")")
        #mass =  sum(CSS.current_density .* CSS.FE_densitypressure.grid.volume4cells, dims = 1)
        #println("    |total mass=", mass)

        CSS.current_time += dt
    #end

    return changeV+changeD;
end




end
