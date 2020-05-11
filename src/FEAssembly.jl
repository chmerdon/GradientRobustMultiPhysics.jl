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
export L2ErrorIntegrator

function L2ErrorIntegrator(exact_function::Function, operator::Type{<:AbstractFunctionOperator}, xdim::Int, ncomponents::Int = 1; AT::Type{<:AbstractAssemblyType} = AbstractAssemblyTypeCELL, bonus_quadorder::Int = 0)
    function L2error_function()
        temp = zeros(Float64,ncomponents*Length4Operator(operator,xdim))
        function closure(result,input,x)
            exact_function(temp,x)
            result[1] = 0.0
            for j=1:length(temp)
                result[1] += (temp[j] - input[j])^2
            end    
        end
    end    
    L2error_action = XFunctionAction(L2error_function(),1,xdim)
    return ItemIntegrator(AT, operator, L2error_action; bonus_quadorder = bonus_quadorder)
end
end
