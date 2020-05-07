module FESolvePoisson

export show,PoissonProblemDescription
export AbstractBoundaryType, HomogeneousDirichletBoundary, InterpolateDirichletBoundary, BestapproxDirichletBoundary, NeumannBoundary
export solve!

using LinearAlgebra
using BenchmarkTools
using FiniteElements
using FEAssembly
using FEXGrid
using ExtendableGrids
using ExtendableSparse

abstract type AbstractBoundaryType end
abstract type DirichletBoundary <: AbstractBoundaryType end
abstract type BestapproxDirichletBoundary <: DirichletBoundary end
abstract type InterpolateDirichletBoundary <: DirichletBoundary end
abstract type HomogeneousDirichletBoundary <: DirichletBoundary end
abstract type NeumannBoundary <: AbstractBoundaryType end
abstract type DoNothingBoundary <: NeumannBoundary end


mutable struct PoissonProblemDescription
    name::String
    time_dependent_data:: Bool
    diffusion
    quadorder4diffusion::Int64
    convection # vector-valued function
    quadorder4convection::Int64
    volumedata4region:: Vector{Function}
    quadorder4region:: Vector{Int64}
    boundarydata4bregion:: Vector{Function}
    boundarytype4bregion:: Array{DataType,1}
    quadorder4bregion:: Vector{Int64}
    PoissonProblemDescription() = new("undefined Poisson problem", false,1.0,0,Nothing,-1)
end

function show(PD::PoissonProblemDescription)
    println("\nPoissonProblemDescription");
    println("=========================")
	println("         name : " * PD.name);
    println("    time-dep. : " * string(PD.time_dependent_data));
    
    println("BoundaryData")
    for j = 1 : length(PD.boundarydata4bregion)
        println("  [$j] $(PD.boundarytype4bregion[j])");
    end
end


# right hand side for Poisson problem
function RightHandSide(FE::AbstractFiniteElement,PD::PoissonProblemDescription; verbosity::Int = 0)
    xdim = size(FE.xgrid[Coordinates],1) 
    function rhs_function(result,input,x,region)
        PD.volumedata4region[region](result,x)
        result[1] = result[1]*input[1] 
    end    
    action = RegionWiseXFunctionAction(rhs_function, FE.xgrid[CellRegions],1,xdim)
    bonus_quadorder = maximum(PD.quadorder4region)
    RHS = LinearForm(AbstractAssemblyTypeCELL, FE, Identity, action; bonus_quadorder = bonus_quadorder)
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
function SystemMatrix(FE::AbstractFiniteElement,PD::PoissonProblemDescription; verbosity::Int = 0)
    #############
    # LAPLACIAN #
    #############
    xdim = size(FE.xgrid[Coordinates],1) 
    ndofs = FE.ndofs
    if typeof(PD.diffusion) <: Real 
        action = MultiplyScalarAction(PD.diffusion,xdim)
    else
        # TODO
    end
    A = FEMatrix{Float64}("StiffnessMatrix", FE)
    H1Product = SymmetricBilinearForm(AbstractAssemblyTypeCELL, FE, Gradient, action; bonus_quadorder = 0)    
    if verbosity > 0
        println("\n  Assembling system matrix...")
        println("      diffusion = $(typeof(action))")
        @time assemble!(A[1], H1Product; verbosity = verbosity - 1)
    else 
        assemble!(A[1], H1Product; verbosity = verbosity - 1)
    end    
    ##############
    # CONVECTION #
    ##############
    if PD.convection != Nothing
        function convection_function(result,input,x) # dot(convection!, input=Gradient)
            convection_vector = deepcopy(input)
            PD.convection(convection_vector,x)
            result[1] = 0.0
            for j = 1 : length(input)
                result[1] += convection_vector[j]*input[j]
            end
        end    
        convection_action = XFunctionAction(convection_function, 1, xdim)
        ConvectionForm = BilinearForm(AbstractAssemblyTypeCELL, FE, FE, Gradient, Identity, convection_action; bonus_quadorder = PD.quadorder4convection)    
        if verbosity > 0
            println("      convection = $(typeof(action))")
            @time assemble!(A[1], ConvectionForm; verbosity = verbosity - 1)
        else 
            assemble!(A[1], ConvectionForm; verbosity = verbosity - 1)
        end   
    end


    return A
end

# boundary data treatment
function boundarydata!(Solution::FEVectorBlock{<:Real}, PD::PoissonProblemDescription; verbosity::Int = 0, dirichlet_penalty::Float64 = 1e60)
    FE = Solution.FEType
    xdim = size(FE.xgrid[Coordinates],1) 
    xBFaces = FE.xgrid[BFaces]
    nbfaces = length(xBFaces)
    xBFaceRegions = FE.xgrid[BFaceRegions]

    if verbosity > 0
        println("\n  Assembling boundary data...")
    end    
    # Dirichlet boundary
    fixed_bdofs = []
    InterDirichletBoundaryRegions = findall(x->x == InterpolateDirichletBoundary, PD.boundarytype4bregion)
    if length(InterDirichletBoundaryRegions) > 0

        # find Dirichlet dofs
        xFaceDofs = FE.FaceDofs
        bregiondofs = []
        for r = 1 : length(InterDirichletBoundaryRegions)
            for bface = 1 : nbfaces
                append!(bregiondofs,xFaceDofs[:,xBFaces[bface]])
            end
            bregiondofs = unique(bregiondofs)
            interpolate!(Solution, PD.boundarydata4bregion[r]; dofs = bregiondofs)
            append!(fixed_bdofs,bregiondofs)
            bregiondofs = []
        end    


        if verbosity > 0
            println("   Int-DBnd = $InterDirichletBoundaryRegions (ndofs = $(length(fixed_bdofs)))")
        end    

        
    end
    HomDirichletBoundaryRegions = findall(x->x == HomogeneousDirichletBoundary, PD.boundarytype4bregion)
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
        hom_dofs = unique(hom_dofs)
        append!(fixed_bdofs,hom_dofs)
        fixed_bdofs = unique(fixed_bdofs)
        for j in hom_dofs
            Solution[j] = 0
        end    

        if verbosity > 0
            println("   Hom-DBnd = $HomDirichletBoundaryRegions (ndofs = $(length(hom_dofs)))")
        end    
        
    end
    BADirichletBoundaryRegions = findall(x->x == BestapproxDirichletBoundary, PD.boundarytype4bregion)
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
        BAdofs = unique(BAdofs)

        if verbosity > 0
            println("    BA-DBnd = $BADirichletBoundaryRegions (ndofs = $(length(BAdofs)))")
                
        end    
        # rhs action for region-wise boundarydata best approximation
        function bnd_rhs_function(result,input,x,region)
            PD.boundarydata4bregion[region](result,x)
            result[1] = result[1]*input[1] 
        end    
        action = RegionWiseXFunctionAction(bnd_rhs_function,FE.xgrid[BFaceRegions],1,xdim)
        bonus_quadorder = maximum(PD.quadorder4bregion)
        RHS_bnd = LinearForm(AbstractAssemblyTypeBFACE, FE, Identity, action; regions = BADirichletBoundaryRegions, bonus_quadorder = bonus_quadorder)
        b = zeros(Float64,FE.ndofs,1)
        assemble!(b, RHS_bnd)

        # compute mass matrix
        action = MultiplyScalarAction(1.0,1)
        A = FEMatrix{Float64}("MassMatrixBnd", FE)
        L2ProductBnd = SymmetricBilinearForm(AbstractAssemblyTypeBFACE, FE, Identity, action; regions = Array{Int64,1}([BADirichletBoundaryRegions; HomDirichletBoundaryRegions; InterDirichletBoundaryRegions]), bonus_quadorder = 0)    
        assemble!(A[1],L2ProductBnd; verbosity = verbosity - 1)

        # fix already set dofs by other boundary conditions
        for j in fixed_bdofs
            A[1][j,j] = dirichlet_penalty
            b[j] = Solution[j]*dirichlet_penalty
        end
        append!(fixed_bdofs,BAdofs)
        fixed_bdofs = unique(fixed_bdofs)

        # solve best approximation problem on boundary and write into Solution
        Solution[fixed_bdofs] = A.entries[fixed_bdofs,fixed_bdofs]\b[fixed_bdofs,1]
    end    


    return fixed_bdofs
end


# solver for Poisson problems
function solve!(Solution::FEVectorBlock{<:Real}, PD::PoissonProblemDescription; dirichlet_penalty::Float64 = 1e60, verbosity::Int = 0)

        FE = Solution.FEType

        if verbosity > 0
            println("\nSOLVING POISSON PROBLEM")
            println("=======================")
            println("       name = $(PD.name)")
            println("     target = $(Solution.name)")
            println("         FE = $(FE.name) (ndofs = $(FE.ndofs))")
        end

        # boundarydata
        fixed_bdofs = boundarydata!(Solution, PD; verbosity = verbosity - 1, dirichlet_penalty = dirichlet_penalty)
    
        # assembly system
        A = SystemMatrix(FE, PD; verbosity = verbosity - 1)
        b = RightHandSide(FE, PD; verbosity = verbosity - 1)
        
        # penalize fixed dofs
        for j = 1 : length(fixed_bdofs)
            b[fixed_bdofs[j],1] = dirichlet_penalty * Solution[fixed_bdofs[j]]
            A[1][fixed_bdofs[j],fixed_bdofs[j]] = dirichlet_penalty
        end

        # solve
        # here would like to write
        #Solution[:] = A[1]\b[:,1]
        # bu that does not work at the moment, so we use
        Solution[:] = A.entries\b[:,1]
end


end
