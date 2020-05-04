module FESolvePoisson

export PoissonProblemDescription,solve!

using LinearAlgebra
using BenchmarkTools
using FiniteElements
using FEOperator
using FEXGrid
using ExtendableGrids


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
    boundarytype4bregion:: Vector{Int64}
    quadorder4bregion:: Vector{Int64}
    PoissonProblemDescription() = new("undefined Poisson problem", false,1.0,0,Nothing,-1)
end

function show(PD::PoissonProblemDescription)

	println("PoissonProblem description");
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

function boundarydata!(Solution::FEFunction, PD::PoissonProblemDescription; talkative::Bool = false)
    FE = Solution.FEType
    xdim = size(Solution.FEType.xgrid[Coordinates],1) 

    # Dirichlet boundary
    fixed_bdofs = []
    DirichletBoundaryRegions = findall(x->x == 1, PD.boundarytype4bregion)
    if length(DirichletBoundaryRegions) > 0

        # find Dirichlet dofs
        xBFaces = FE.xgrid[BFaces]
        nbfaces = length(xBFaces)
        xFaceDofs = FE.FaceDofs
        for bface = 1 : nbfaces
            append!(fixed_bdofs,xFaceDofs[:,xBFaces[bface]])
        end
        fixed_bdofs = unique(fixed_bdofs)

        if talkative == true
            println("  DbRegions = $DirichletBoundaryRegions (Db ndofs = $(length(fixed_bdofs)))")
        end    
        # rhs action for region-wise boundarydata best approximation
        function bnd_rhs_function(result,input,x,region)
            PD.boundarydata4bregion[region](result,x)
            result[1] = result[1]*input[1] 
        end    
        action = RegionWiseXFunctionAction(bnd_rhs_function,FE.xgrid[BFaceRegions],1,xdim)
        bonus_quadorder = maximum(PD.quadorder4bregion)
        b = RightHandSide(FE, action, AbstractAssemblyTypeBFACE; regions = DirichletBoundaryRegions, bonus_quadorder = bonus_quadorder)[:]

        # compute mass matrix
        action = MultiplyScalarAction(1.0,1)
        A = MassMatrix(FE, action, AbstractAssemblyTypeBFACE)
   

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
