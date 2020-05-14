module FEAssembly

using FiniteElements
using ExtendableGrids
using FEXGrid
using QuadratureRules
using ExtendableSparse
using SparseArrays
using ForwardDiff # for FEBasisEvaluator

export AbstractFunctionOperator
export Identity, Gradient, SymmetricGradient, Laplacian, Hessian, Curl, Rotation, Divergence, Trace, Deviator
export NeededDerivatives4Operator

include("FEAssembly_FEBasisEvaluator.jl")
export FEBasisEvaluator, update!

include("FEAssembly_AbstractAction.jl")
export AbstractAction
export DoNotChangeAction
export MultiplyScalarAction
export MultiplyVectorAction
export MultiplyMatrixAction
export RegionWiseMultiplyScalarAction
export RegionWiseMultiplyVectorAction
export FunctionAction
export XFunctionAction
export RegionWiseXFunctionAction


include("FEAssembly_AbstractAssemblyPattern.jl")
export AbstractAssemblyPattern,ItemIntegrator,LinearForm,BilinearForm,SymmetricBilinearForm
export assemble!, evaluate!, evaluate
export L2ErrorIntegrator, L2bestapproximate!, H1bestapproximate!


function L2ErrorIntegrator(exact_function::Function, operator::Type{<:AbstractFunctionOperator}, xdim::Int, ncomponents::Int = 1; AT::Type{<:AbstractAssemblyType} = AbstractAssemblyTypeCELL, bonus_quadorder::Int = 0)
    function L2error_function()
        temp = zeros(Float64,ncomponents)
        function closure(result,input,x)
            exact_function(temp,x)
            result[1] = 0.0
            for j=1:length(temp)
                result[1] += (temp[j] - input[j])^2
            end    
        end
    end    
    L2error_action = XFunctionAction(L2error_function(),1,xdim; bonus_quadorder = bonus_quadorder)
    return ItemIntegrator(AT, operator, L2error_action)
end


function boundarydata!(Target::FEVectorBlock, exact_function::Function; regions = [0], verbosity::Int = 0, bonus_quadorder::Int = 0)
    FE = Target.FEType
    xdim = size(FE.xgrid[Coordinates],1)
    ncomponents = get_ncomponents(typeof(FE))
    xBFaces = FE.xgrid[BFaces]
    xFaceDofs = FE.FaceDofs
    xBFaceRegions = FE.xgrid[BFaceRegions]

    if regions == []
        return []
    elseif regions == [0]
        try
            regions = Array{Int,1}(Base.unique(xBFaceRegions[:]))
        catch
            regions = [xBFaceRegions[1]]
        end        
    else
        regions = Array{Int,1}(regions)    
    end
    
    # find Dirichlet dofs
    fixed_bdofs = []
    for bface = 1 : length(xBFaces)
        append!(fixed_bdofs,xFaceDofs[:,xBFaces[bface]])
    end
    fixed_bdofs = Base.unique(fixed_bdofs)

    # rhs action for region-wise boundarydata best approximation
    function bnd_rhs_function()
        temp = zeros(Float64,ncomponents)
        function closure(result, input, x)
            exact_function(temp,x)
            result[1] = 0.0
            for j = 1 : ncomponents
                result[1] += temp[j]*input[j] 
            end 
        end   
    end   
    
    action = XFunctionAction(bnd_rhs_function(),1,xdim; bonus_quadorder = bonus_quadorder)
    RHS_bnd = LinearForm(AbstractAssemblyTypeBFACE, FE, Identity, action; regions = regions)
    b_bnd = FEVector{Float64}("RhsBnd", FE)
    FEAssembly.assemble!(b_bnd[1], RHS_bnd; verbosity = verbosity - 1)

    A_bnd = FEMatrix{Float64}("MassMatrixBnd", FE)
    L2ProductBnd = SymmetricBilinearForm(AbstractAssemblyTypeBFACE, FE, Identity, DoNotChangeAction(ncomponents); regions = regions)    
    FEAssembly.assemble!(A_bnd[1],L2ProductBnd; verbosity = verbosity - 1)

    # solve best approximation problem on boundary and write into Target
    Target[fixed_bdofs] = A_bnd.entries[fixed_bdofs,fixed_bdofs]\b_bnd.entries[fixed_bdofs,1]

    return fixed_bdofs
end


function L2bestapproximate!(
    Target::FEVectorBlock,
    exact_function::Function;
    boundary_regions = [0],
    dirichlet_penalty::Float64 = 1e60,
    verbosity::Int = 0,
    bonus_quadorder::Int = 0)
    FE = Target.FEType
    ncomponents = get_ncomponents(typeof(FE))

    if verbosity > 0
        println("\nL2-BESTAPPROXIMATING")
        println("====================")
        println("     target = $(Target.name)")
        println("         FE = $(FE.name) (ndofs = $(FE.ndofs))")
    end

    # matrix
    L2Product = SymmetricBilinearForm(AbstractAssemblyTypeCELL, FE, Identity, DoNotChangeAction(ncomponents))    
    A = FEMatrix{Float64}("Matrix",FE)
    assemble!(A[1], L2Product; verbosity = verbosity - 1)

    # rhs
    function rhs_function()
        temp = zeros(Float64,ncomponents)
        function closure(result,input,x)
            exact_function(temp,x)
            result[1] = 0
            for j = 1 : ncomponents
                result[1] += temp[j]*input[j] 
            end
        end
    end    
    action = XFunctionAction(rhs_function(),1; bonus_quadorder = bonus_quadorder)
    b = FEVector{Float64}("rhs",FE)
    RHS = LinearForm(AbstractAssemblyTypeCELL, FE, Identity, action)
    assemble!(b[1], RHS; verbosity = verbosity - 1)

    fixed_bdofs = boundarydata!(Target, exact_function; regions = boundary_regions, verbosity = verbosity, bonus_quadorder = bonus_quadorder)

    # fix in global matrix
    for j = 1 : length(fixed_bdofs)
        b.entries[fixed_bdofs[j]] = dirichlet_penalty * Target.entries[fixed_bdofs[j]]
        A.entries[fixed_bdofs[j],fixed_bdofs[j]] = dirichlet_penalty
    end

    Target[:] = A.entries\b.entries
end

function H1bestapproximate!(
    Target::FEVectorBlock,
    exact_function_gradient::Function,
    exact_function::Function;
    boundary_regions = [0],
    verbosity::Int = 0,
    bonus_quadorder::Int = 0,
    dirichlet_penalty::Float64 = 1e60)
    
    FE = Target.FEType
    xdim = size(FE.xgrid[Coordinates],1)
    ncomponents = get_ncomponents(typeof(FE))

    if verbosity > 0
        println("\nH1-BESTAPPROXIMATING")
        println("====================")
        println("     target = $(Target.name)")
        println("         FE = $(FE.name) (ndofs = $(FE.ndofs))")
    end

    # matrix
    H1Product = SymmetricBilinearForm(AbstractAssemblyTypeCELL, FE, Gradient, DoNotChangeAction(ncomponents*xdim))    
    A = FEMatrix{Float64}("Matrix",FE)
    assemble!(A[1], H1Product; verbosity = verbosity - 1)

    # rhs
    function rhs_function()
        temp = zeros(Float64,ncomponents*xdim)
        function closure(result,input,x)
            exact_function_gradient(temp,x)
            result[1] = 0
            for j = 1 : ncomponents
                result[1] += temp[j]*input[j] 
            end
        end
    end    
    action = XFunctionAction(rhs_function(),1; bonus_quadorder = bonus_quadorder - 1)
    b = FEVector{Float64}("rhs",FE)
    RHS = LinearForm(AbstractAssemblyTypeCELL, FE, Identity, action)
    assemble!(b[1], RHS; verbosity = verbosity - 1)

    fixed_bdofs = boundarydata!(Target, exact_function; regions = boundary_regions, verbosity = verbosity, bonus_quadorder = bonus_quadorder)

    # fix in global matrix
    for j = 1 : length(fixed_bdofs)
        b.entries[fixed_bdofs[j]] = dirichlet_penalty * Target.entries[fixed_bdofs[j]]
        A.entries[fixed_bdofs[j],fixed_bdofs[j]] = dirichlet_penalty
    end

    Target[:] = A.entries\b.entries
end

end
