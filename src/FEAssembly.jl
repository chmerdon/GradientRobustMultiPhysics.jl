module FEAssembly

using FiniteElements
using ExtendableGrids
using FEXGrid
using QuadratureRules
using ExtendableSparse
using SparseArrays


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
export ItemWiseXFunctionAction
export RegionWiseXFunctionAction


include("FEAssembly_AbstractAssemblyPattern.jl")
export AbstractAssemblyPattern,ItemIntegrator,LinearForm,BilinearForm
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
    return ItemIntegrator{Float64,AT}(operator, L2error_action, [0])
end

end
