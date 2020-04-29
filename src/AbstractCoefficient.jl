abstract type AbstractCoefficient end

# more ideas for constants
# ItemWiseConstant (like CellDiameters)
# FEFunction (but possibly needs its own FEBasisEvaluators)

struct ScalarCoefficient{T <: Real} <: AbstractCoefficient
    value::Real
    resultdim::Int
end
struct VectorCoefficient{T <: Real} <: AbstractCoefficient
    value::Array{T,1}
    resultdim::Int
end
struct MatrixCoefficient{T <: Real} <: AbstractCoefficient
    value::Array{T,2}
    resultdim::Int
end
struct RegionWiseScalarCoefficient{T <: Real} <: AbstractCoefficient
    value::Array{Real,1} # one array for each region
    ItemRegions::AbstractElementRegions
    cregion::Int32
    resultdim::Int
end
struct RegionWiseVectorCoefficient{T <: Real} <: AbstractCoefficient
    value::Array{Array{T,1},1} # one array for each region
    ItemRegions::AbstractElementRegions
    cregion::Int32
    resultdim::Int
end
#struct FunctionCoefficient{T <: Real} <: AbstractCoefficient
#    f::Function # of the interface f!(result,x)
#    resultdim::Int
#    x::Array{T,1}
#end

export ScalarCoefficient
export VectorCoefficient
export MatrixCoefficient
export RegionWiseScalarCoefficient
export RegionWiseVectorCoefficient

function ScalarCoefficient(value::Real, resultdim::Int = 1)
    return ScalarCoefficient{eltype(value)}(value, resultdim)
end

function VectorCoefficient(value::Array{<:Real,1})
    return VectorCoefficient{eltype(value)}(value, length(value))
end

function MatrixCoefficient(value::Array{<:Real,2})
    return MatrixCoefficient{eltype(value)}(value, size(value,1))
end

function RegionWiseVectorCoefficient(value::Array{Array{<:Real,1},1}, ItemRegions::AbstractElementRegions, resultdim::Int)
    return RegionWiseVectorCoefficient{eltype(value[1])}(value, ItemRegions, 1, length(value[1]))
end

function RegionWiseScalarCoefficient(value::Array{<:Real,1}, ItemRegions::AbstractElementRegions, resultdim::Int = 1)
    return RegionWiseScalarCoefficient{eltype(value)}(value, ItemRegions, 1, resultdim)
end

function update!(C::AbstractCoefficient, item::Int32)
    # do nothing
end

function update!(C::Union{RegionWiseVectorCoefficient,RegionWiseScalarCoefficient}, item::Int32)
    C.cregion = C.ItemRegions[item]
end

function apply_coefficient!(result::Array{<:Real,1}, input::Array{<:Real,1}, C::ScalarCoefficient)
    for j = 1:length(result)
        result[j] = input[j] * C.value;    
    end    
end

function apply_coefficient!(result::Array{<:Real,1}, input::Array{<:Real,1}, C::RegionWiseScalarCoefficient)
    for j = 1:length(result)
        result[j] = input[j] * C.value[C.cregion];    
    end    
end

function apply_coefficient!(result::Array{<:Real,1}, input::Array{<:Real,1}, C::VectorCoefficient)
    for j = 1:length(result)
        result[j] = input[j] * C.value[j];
    end    
end

function apply_coefficient!(result::Array{<:Real,1}, input::Array{<:Real,1}, C::MatrixCoefficient)
    for j = 1:length(result)
        result[j] = 0
        for k = 1:length(input)
            result[j] += C.value[j,k]*input[k];
        end    
    end    
end

function apply_coefficient!(result::Array{<:Real,1}, C::RegionWiseVectorCoefficient)
    for j = 1:length(result)
        result[j] = input[j] * C.value[C.cregion][j];
    end    
end

#function eval!(result, xref, C::FunctionCoefficient, L2G::L2GTransformer, item::Int32 = Int32(0), region::Int32 = Int32(0))
#    FEXGrid.eval!(C.x, L2G, xref)
#    C.f(result,C.x);
#end