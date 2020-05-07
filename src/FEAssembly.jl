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
export DoNothingAction
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
export L2ErrorIntegrator

function L2ErrorIntegrator(exact_function::Function, operator::Type{<:AbstractFunctionOperator}, xdim::Int; AT::Type{<:AbstractAssemblyType} = AbstractAssemblyTypeCELL, bonus_quadorder::Int = 0)
    function L2error_function(result,input,x)
        exact_function(result,x)
        result[1] = (result[1] - input[1])^2
        for j=2:length(input)
            result[1] += (result[j] - input[j])^2
        end    
        for j=2:length(result)
            result[2] = 0.0
        end    
    end    
    L2error_action = XFunctionAction(L2error_function,Length4Operator(operator,xdim),xdim)
    return ItemIntegrator(AT, operator, L2error_action; bonus_quadorder = bonus_quadorder)
end
end
