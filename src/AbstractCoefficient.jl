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
struct FunctionCoefficient{T <: Real} <: AbstractCoefficient
    f::Function # of the interface f!(result,input,x)
    resultdim::Int
    x::Array{Array{T,1},1}
end

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

function FunctionCoefficient(f::Function, resultdim::Int = 1, xdim::Int = 2)
    return FunctionCoefficient{Float64}(f, resultdim, [zeros(Float64,xdim)])
end

###
# updates are called on each change of item 
###

function update!(C::AbstractCoefficient, FEBE::FEBasisEvaluator, item::Int32)
    # do nothing
end

function update!(C::FunctionCoefficient, FEBE::FEBasisEvaluator, item::Int32)
    # compute global coordinates for function evaluation
    if FEBE.L2G.citem != item 
        FEXGrid.update!(FEBE.L2G, item)
    end
    for i = 1 : length(FEBE.xref)
        # we don't know at contruction time how many quadrature points are needed
        # so we expand the array here if needed
        if length(C.x) < length(FEBE.xref)
            push!(C.x,FEBE.xref[i])
        end    
        FEXGrid.eval!(C.x[i],FEBE.L2G,FEBE.xref[i])
    end    
end

function update!(C::Union{RegionWiseVectorCoefficient,RegionWiseScalarCoefficient}, item::Int32)
    C.cregion = C.ItemRegions[item]
end

###
# apply_coefficient is called for each dof and i-th quadrature point
###

function apply_coefficient!(result::Array{<:Real,1}, input::Array{<:Real,1}, C::ScalarCoefficient, i::Int = 0)
    for j = 1:length(result)
        result[j] = input[j] * C.value;    
    end    
end

function apply_coefficient!(result::Array{<:Real,1}, input::Array{<:Real,1}, C::RegionWiseScalarCoefficient, i::Int = 0)
    for j = 1:length(result)
        result[j] = input[j] * C.value[C.cregion];    
    end    
end

function apply_coefficient!(result::Array{<:Real,1}, input::Array{<:Real,1}, C::VectorCoefficient, i::Int = 0)
    for j = 1:length(result)
        result[j] = input[j] * C.value[j];
    end    
end

function apply_coefficient!(result::Array{<:Real,1}, input::Array{<:Real,1}, C::MatrixCoefficient, i::Int = 0)
    for j = 1:length(result)
        result[j] = 0
        for k = 1:length(input)
            result[j] += C.value[j,k]*input[k];
        end    
    end    
end

function apply_coefficient!(result::Array{<:Real,1}, input::Array{<:Real,1}, C::RegionWiseVectorCoefficient, i::Int = 0)
    for j = 1:length(result)
        result[j] = input[j] * C.value[C.cregion][j];
    end    
end

function apply_coefficient!(result::Array{<:Real,1}, input::Array{<:Real,1}, C::FunctionCoefficient, i::Int)
    C.f(result, input, C.x[i]);
end