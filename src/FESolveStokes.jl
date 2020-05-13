module FESolveStokes

export show,StokesProblemDescription
export AbstractBoundaryType, HomogeneousDirichletBoundary, InterpolateDirichletBoundary, BestapproxDirichletBoundary, NeumannBoundary
export solve!

using LinearAlgebra
using BenchmarkTools
using FiniteElements
using FEAssembly
using PDETools
using FEXGrid
using ExtendableGrids
using ExtendableSparse


mutable struct StokesProblemDescription
    name::String
    time_dependent_data:: Bool
    viscosity
    quadorder4diffusion::Int64
    convection # vector-valued function a for (a * grad)
    quadorder4convection::Int64
    volumedata4region:: Vector{Function}
    quadorder4region:: Vector{Int64}
    boundarydata4bregion:: Vector{Function}
    boundarytype4bregion:: Array{DataType,1}
    quadorder4bregion:: Vector{Int64}
    StokesProblemDescription() = new("undefined Stokes problem", false,1.0,0,Nothing,-1)
end

function show(PD::StokesProblemDescription)
    println("\nStokesProblemDescription");
    println("=========================")
	println("         name : " * PD.name);
    println("    time-dep. : " * string(PD.time_dependent_data));
    
    println("BoundaryData")
    for j = 1 : length(PD.boundarydata4bregion)
        println("  [$j] $(PD.boundarytype4bregion[j])");
    end
end


function boundarydata!(
    Target::FEVectorBlock,
    boundarytype4bregion::Array{DataType,1},
    boundarydata4bregion::Array{Function,1},
    quadorder4bregion::Array{Int,1};
    verbosity::Int = 0,
    dirichlet_penalty::Float64 = 1e60)

    FE = Target.FEType
    xdim = size(FE.xgrid[Coordinates],1) 
    xBFaces = FE.xgrid[BFaces]
    nbfaces = length(xBFaces)
    xBFaceRegions = FE.xgrid[BFaceRegions]
    ncomponents = get_ncomponents(typeof(FE))

    if verbosity > 0
        println("\n  Assembling boundary data...")
    end    

    ######################
    # Dirichlet boundary #
    ######################
    fixed_bdofs = []

    # INTERPOLATION DIRICHLET BOUNDARY
    InterDirichletBoundaryRegions = findall(x->x == InterpolateDirichletBoundary, boundarytype4bregion)
    if length(InterDirichletBoundaryRegions) > 0

        # find Dirichlet dofs
        xFaceDofs = FE.FaceDofs
        bregiondofs = []
        for r = 1 : length(InterDirichletBoundaryRegions)
            for bface = 1 : nbfaces
                append!(bregiondofs,xFaceDofs[:,xBFaces[bface]])
            end
            bregiondofs = Base.unique(bregiondofs)
            interpolate!(Target, boundarydata4bregion[r]; dofs = bregiondofs)
            append!(fixed_bdofs,bregiondofs)
            bregiondofs = []
        end    


        if verbosity > 0
            println("   Int-DBnd = $InterDirichletBoundaryRegions (ndofs = $(length(fixed_bdofs)))")
        end    

        
    end

    # HOMOGENEOUS DIRICHLET BOUNDARY
    HomDirichletBoundaryRegions = findall(x->x == HomogeneousDirichletBoundary, boundarytype4bregion)
    if length(HomDirichletBoundaryRegions) > 0

        # find Dirichlet dofs
        xFaceDofs = FE.FaceDofs
        hom_dofs = []
        for bface = 1 : nbfaces
            for r = 1 : length(HomDirichletBoundaryRegions)
                if xBFaceRegions[bface] == HomDirichletBoundaryRegions[r]
                    append!(hom_dofs,xFaceDofs[:,xBFaces[bface]])
                    break
                end    
            end    
        end
        hom_dofs = Base.unique(hom_dofs)
        append!(fixed_bdofs,hom_dofs)
        fixed_bdofs = Base.unique(fixed_bdofs)
        for j in hom_dofs
            Target[j] = 0
        end    

        if verbosity > 0
            println("   Hom-DBnd = $HomDirichletBoundaryRegions (ndofs = $(length(hom_dofs)))")
        end    
        
    end

    # BEST-APPROXIMATION DIRICHLET BOUNDARY
    BADirichletBoundaryRegions = findall(x->x == BestapproxDirichletBoundary, boundarytype4bregion)
    if length(BADirichletBoundaryRegions) > 0

        # find Dirichlet dofs
        xFaceDofs = FE.FaceDofs
        BAdofs = []
        for bface = 1 : nbfaces
            for r = 1 : length(BADirichletBoundaryRegions)
                if xBFaceRegions[bface] == BADirichletBoundaryRegions[r]
                    append!(BAdofs,xFaceDofs[:,xBFaces[bface]])
                    break
                end    
            end    
        end
        BAdofs = Base.unique(BAdofs)

        if verbosity > 0
            println("    BA-DBnd = $BADirichletBoundaryRegions (ndofs = $(length(BAdofs)))")
                
        end    
        # rhs action for region-wise boundarydata best approximation
        function bnd_rhs_function()
            temp = zeros(Float64,ncomponents)
            function closure(result, input, x, region)
                boundarydata4bregion[region](temp,x)
                result[1] = 0.0
                for j = 1 : ncomponents
                    result[1] += temp[j]*input[j] 
                end 
            end   
        end   
        bonus_quadorder = maximum(quadorder4bregion)
        action = RegionWiseXFunctionAction(bnd_rhs_function(),1,xdim; bonus_quadorder = bonus_quadorder)
        RHS_bnd = LinearForm(AbstractAssemblyTypeBFACE, FE, Identity, action; regions = BADirichletBoundaryRegions)
        b = zeros(Float64,FE.ndofs,1)
        assemble!(b, RHS_bnd)

        # compute mass matrix
        action = MultiplyScalarAction(1.0,ncomponents)
        A = FEMatrix{Float64}("MassMatrixBnd", FE)
        L2ProductBnd = SymmetricBilinearForm(AbstractAssemblyTypeBFACE, FE, Identity, action; regions = Array{Int64,1}([BADirichletBoundaryRegions; HomDirichletBoundaryRegions; InterDirichletBoundaryRegions]))  
        assemble!(A[1],L2ProductBnd; verbosity = verbosity - 1)

        # fix already set dofs by other boundary conditions
        for j in fixed_bdofs
            A[1][j,j] = dirichlet_penalty
            b[j] = Target[j]*dirichlet_penalty
        end
        append!(fixed_bdofs,BAdofs)
        fixed_bdofs = Base.unique(fixed_bdofs)

        # solve best approximation problem on boundary and write into Target
        Target[fixed_bdofs] = A.entries[fixed_bdofs,fixed_bdofs]\b[fixed_bdofs,1]
    end
    
    return fixed_bdofs
end    




# right hand side for Stokes problem
function RightHandSide(FE::AbstractFiniteElement,PD::StokesProblemDescription; verbosity::Int = 0)
    xdim = size(FE.xgrid[Coordinates],1) 
    function rhs_function()
        temp = zeros(Float64,xdim)
        function closure(result, input, x, region)
            PD.volumedata4region[region](temp,x)
            result[1] = 0.0
            for j = 1 : length(input)
                result[1] += temp[j]*input[j] 
            end    
        end
    end    
    bonus_quadorder = maximum(PD.quadorder4region)
    action = RegionWiseXFunctionAction(rhs_function(),1,xdim; bonus_quadorder = bonus_quadorder)
    RHS = LinearForm(AbstractAssemblyTypeCELL, FE, Identity, action)
    b = zeros(Float64,FE.ndofs,1)
    if verbosity > 0
        println("\n  Assembling right-hand side...")
        @time assemble!(b, RHS)
    else
        assemble!(b, RHS)
    end        
    return b
end    


# system matrix for Poisson problem
function SystemMatrix(FE_velocity::AbstractFiniteElement, FE_pressure::AbstractFiniteElement, PD::StokesProblemDescription; verbosity::Int = 0)
    #############
    # LAPLACIAN #
    #############
    xdim = size(FE_velocity.xgrid[Coordinates],1) 
    ndofs = FE_velocity.ndofs
    if typeof(PD.viscosity) <: Real 
        if PD.viscosity == 1
            action = DoNotChangeAction(xdim*xdim)
        else
            action = MultiplyScalarAction(PD.viscosity,xdim*xdim)
        end    
    else
        # TODO
    end
    A = FEMatrix{Float64}("StokesOperator", [FE_velocity, FE_pressure])
    H1Product = SymmetricBilinearForm(AbstractAssemblyTypeCELL, FE_velocity, Gradient, action)    
    DivPressure = BilinearForm(AbstractAssemblyTypeCELL, FE_velocity, FE_pressure, Divergence, Identity, MultiplyScalarAction(-1.0,1))    
    if verbosity > 0
        println("\n  Assembling Stokes operator...")
        println("      Laplacian block")
        println("        viscosity = $(typeof(action))")
        @time assemble!(A[1], H1Product; verbosity = verbosity - 1)
        println("      DivPressure block")
        @time assemble!(A[3], DivPressure; verbosity = verbosity - 1, transpose_copy = A[2])
    else 
        assemble!(A[1], H1Product; verbosity = verbosity - 1)
        assemble!(A[3], DivPressure; verbosity = verbosity - 1, transpose_copy = A[2])
    end
    return A
end

# solver for Poisson problems
function solve!(Velocity::FEVectorBlock{<:Real}, Pressure::FEVectorBlock{<:Real}, PD::StokesProblemDescription; dirichlet_penalty::Float64 = 1e60, verbosity::Int = 0)

        FE_velocity = Velocity.FEType
        FE_pressure = Pressure.FEType
        ndofs_velocity = FE_velocity.ndofs
        ndofs_pressure = FE_pressure.ndofs

        if verbosity > 0
            println("\nSOLVING STOKES PROBLEM")
            println("=======================")
            println("       name = $(PD.name)")
            println("   target v = $(Velocity.name)")
            println("   target p = $(Pressure.name)")
            println("       FE v = $(FE_velocity.name) (ndofs = $ndofs_velocity)")
            println("       FE p = $(FE_pressure.name) (ndofs = $ndofs_pressure)")
        end

        # boundarydata
        fixed_bdofs = FEAssembly.boundarydata!(Velocity, PD.boundarytype4bregion, PD.boundarydata4bregion, PD.quadorder4bregion; verbosity = verbosity, dirichlet_penalty = dirichlet_penalty)
    
        # assembly system
        A = SystemMatrix(FE_velocity, FE_pressure, PD; verbosity = verbosity - 1)
        b = RightHandSide(FE_velocity, PD; verbosity = verbosity - 1)
        b = [b;zeros(Float64,ndofs_pressure)]
        push!(fixed_bdofs,ndofs_velocity+1)
        
        # penalize fixed dofs
        for j = 1 : length(fixed_bdofs)
            b[fixed_bdofs[j],1] = dirichlet_penalty * Velocity[fixed_bdofs[j]]
            A.entries[fixed_bdofs[j],fixed_bdofs[j]] = dirichlet_penalty
        end

        # solve
        if verbosity > 0
            println("\n  Solving...")
            @time Solution = A.entries\b[:,1]
        else
            Solution = A.entries\b[:,1]
        end    
        Velocity[:] = Solution[1:ndofs_velocity]
        Pressure[:] = Solution[ndofs_velocity+1:ndofs_velocity+ndofs_pressure]

        # move integral mean
        pmeanIntegrator = ItemIntegrator(AbstractAssemblyTypeCELL, Identity, DoNotChangeAction(1))
        pressure_mean =  evaluate(pmeanIntegrator,Pressure; verbosity = verbosity - 1)
        total_area = sum(FE_velocity.xgrid[CellVolumes], dims=1)[1]
        pressure_mean /= total_area
        for j=1:ndofs_pressure
            Pressure[j]  -= pressure_mean
        end    
        
end


end
