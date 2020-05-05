module FESolvePoisson

export show,PoissonProblemDescription
export AbstractBoundaryType, HomogeneousDirichletBoundary, BestapproxDirichletBoundary, NeumannBoundary
export solve!

using LinearAlgebra
using BenchmarkTools
using FiniteElements
using FEOperator
using FEXGrid
using ExtendableGrids

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
    diffusion # Real/Matrix or matrix-valued function
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

function boundarydata!(Solution::FEFunction, PD::PoissonProblemDescription; talkative::Bool = false)
    FE = Solution.FEType
    xdim = size(Solution.FEType.xgrid[Coordinates],1) 
    xBFaces = FE.xgrid[BFaces]
    nbfaces = length(xBFaces)
    xBFaceRegions = FE.xgrid[BFaceRegions]

    # Dirichlet boundary
    fixed_bdofs = []
    HomDirichletBoundaryRegions = findall(x->x == HomogeneousDirichletBoundary, PD.boundarytype4bregion)
    if length(HomDirichletBoundaryRegions) > 0

        # find Dirichlet dofs
        xFaceDofs = FE.FaceDofs
        for bface = 1 : nbfaces
            for r = 1 : length(HomDirichletBoundaryRegions)
                if xBFaceRegions[bface] == HomDirichletBoundaryRegions[r]
                    append!(fixed_bdofs,xFaceDofs[:,xBFaces[bface]])
                    break
                end    
            end    
        end

        if talkative == true
            println("   Hom-DBnd = $HomDirichletBoundaryRegions")
        end    
        
    end
    BADirichletBoundaryRegions = findall(x->x == BestapproxDirichletBoundary, PD.boundarytype4bregion)
    if length(BADirichletBoundaryRegions) > 0

        # find Dirichlet dofs
        xFaceDofs = FE.FaceDofs
        for bface = 1 : nbfaces
            for r = 1 : length(BADirichletBoundaryRegions)
                if xBFaceRegions[bface] == BADirichletBoundaryRegions[r]
                    append!(fixed_bdofs,xFaceDofs[:,xBFaces[bface]])
                    break
                end    
            end    
        end
        fixed_bdofs = unique(fixed_bdofs)

        if talkative == true
            println("    BA-DBnd = $BADirichletBoundaryRegions")
            println("  DBnd ndof = $(length(fixed_bdofs))")
                
        end    
        # rhs action for region-wise boundarydata best approximation
        function bnd_rhs_function(result,input,x,region)
            PD.boundarydata4bregion[region](result,x)
            result[1] = result[1]*input[1] 
        end    
        action = RegionWiseXFunctionAction(bnd_rhs_function,FE.xgrid[BFaceRegions],1,xdim)
        bonus_quadorder = maximum(PD.quadorder4bregion)
        b = RightHandSide(FE, action, AbstractAssemblyTypeBFACE; regions = BADirichletBoundaryRegions, bonus_quadorder = bonus_quadorder)[:]

        # compute mass matrix
        action = MultiplyScalarAction(1.0,1)
        A = MassMatrix(FE, action, AbstractAssemblyTypeBFACE; regions = Array{Int64,1}([BADirichletBoundaryRegions; HomDirichletBoundaryRegions]))
   

        # solve best approximation problem on boundary and write into Solution
        Solution.coefficients[fixed_bdofs] = A[fixed_bdofs,fixed_bdofs]\b[fixed_bdofs]
    end    


    return fixed_bdofs
end


# computes solution of Poisson problem
function solve!(Solution::FEFunction, PD::PoissonProblemDescription; dirichlet_penalty::Float64 = 1e60, talkative::Bool = false)

        FE = Solution.FEType

        if (talkative == true)
            println("\nSOLVING POISSON PROBLEM")
            println("=======================")
            println("         FE = $(FE.name), ndofs = $(FE.ndofs)")
        end

        # boundarydata
        fixed_bdofs = boundarydata!(Solution, PD; talkative = talkative)
    
        # compute stiffness matrix
        xdim = size(Solution.FEType.xgrid[Coordinates],1) 
        diffusion_action = MultiplyScalarAction(PD.diffusion,xdim)
        A = StiffnessMatrix(FE, diffusion_action; talkative = talkative)
        
        # compute right hand side
        function rhs_function(result,input,x,region)
            PD.volumedata4region[region](result,x)
            result[1] = result[1]*input[1] 
        end    
        action = RegionWiseXFunctionAction(rhs_function, FE.xgrid[CellRegions],1,xdim)
        b = RightHandSide(FE, action; talkative = talkative, bonus_quadorder = 2)[:]


        for j = 1 : length(fixed_bdofs)
            b[fixed_bdofs[j]] = dirichlet_penalty * Solution.coefficients[fixed_bdofs[j]]
            A[fixed_bdofs[j],fixed_bdofs[j]] = dirichlet_penalty
        end

        # solve
        Solution.coefficients[:] = A\b

end


end
