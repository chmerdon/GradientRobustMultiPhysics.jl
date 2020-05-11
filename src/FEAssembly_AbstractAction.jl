abstract type AbstractAction end
struct DoNotChangeAction <: AbstractAction
    resultdim::Int
end

struct MultiplyScalarAction{T <: Real} <: AbstractAction
    value::Real
    resultdim::Int
end
struct MultiplyVectorAction{T <: Real} <: AbstractAction
    value::Array{T,1}
    resultdim::Int
end
struct MultiplyMatrixAction{T <: Real} <: AbstractAction
    value::Array{T,2}
    resultdim::Int
end
struct RegionWiseMultiplyScalarAction{T <: Real} <: AbstractAction
    value::Array{Real,1} # one array for each region
    ItemRegions::AbstractElementRegions
    cregion::Int
    resultdim::Int
end
struct RegionWiseMultiplyVectorAction{T <: Real} <: AbstractAction
    value::Array{Array{T,1},1} # one array for each region
    ItemRegions::AbstractElementRegions
    cregion::Int
    resultdim::Int
end
struct FunctionAction{T <: Real} <: AbstractAction
    f::Function # of the interface f!(result,input)
    resultdim::Int
end
struct XFunctionAction{T <: Real} <: AbstractAction
    f::Function # of the interface f!(result,input,x)
    resultdim::Int
    x::Array{Array{T,1},1}
end
mutable struct RegionWiseXFunctionAction{T <: Real} <: AbstractAction
    f::Function # of the interface f!(result,input,x,region)
    ItemRegions::Union{Array{Int32,1},AbstractElementRegions,VectorOfConstants{Int32}}
    cregion::Int
    resultdim::Int
    x::Array{Array{T,1},1}
end

function MultiplyScalarAction(value::Real, resultdim::Int = 1)
    return MultiplyScalarAction{eltype(value)}(value, resultdim)
end

function MultiplyVectorAction(value::Array{<:Real,1})
    return MultiplyVectorAction{eltype(value)}(value, length(value))
end

function MultiplyMatrixAction(value::Array{<:Real,2})
    return MultiplyMatrixAction{eltype(value)}(value, size(value,1))
end

function RegionWiseMultiplyVectorAction(value::Array{Array{<:Real,1},1}, ItemRegions::AbstractElementRegions, resultdim::Int)
    return RegionWiseMultiplyVectorAction{eltype(value[1])}(value, ItemRegions, 1, length(value[1]))
end

function RegionWiseMultiplyScalarAction(value::Array{<:Real,1}, ItemRegions::AbstractElementRegions, resultdim::Int = 1)
    return RegionWiseMultiplyScalarAction{eltype(value)}(value, ItemRegions, 1, resultdim)
end

function FunctionAction(f::Function, resultdim::Int = 1)
    return FunctionAction{Float64}(f, resultdim)
end

function XFunctionAction(f::Function, resultdim::Int = 1, xdim::Int = 2)
    return XFunctionAction{Float64}(f, resultdim, [])
end

function RegionWiseXFunctionAction(f::Function, ItemRegions::Union{Array{Int32,1},AbstractElementRegions,VectorOfConstants{Int32}}, resultdim::Int = 1, xdim::Int = 2)
    return RegionWiseXFunctionAction{Float64}(f, ItemRegions, 1, resultdim, [])
end

###
# update! is called on each change of item 
###

function update!(C::AbstractAction, FEBE::FEBasisEvaluator, item::Int)
    # do nothing
end

function update!(C::RegionWiseXFunctionAction, FEBE::FEBasisEvaluator, item::Int)
    C.cregion = C.ItemRegions[item]

    # compute global coordinates for function evaluation
    if FEBE.L2G.citem != item 
        FEXGrid.update!(FEBE.L2G, item)
    end
    # we don't know at contruction time how many quadrature points are needed
    # so we expand the array here if needed
    while length(C.x) < length(FEBE.xref)
        push!(C.x,deepcopy(FEBE.xref[1]))
    end  
    for i = 1 : length(FEBE.xref) 
        FEXGrid.eval!(C.x[i],FEBE.L2G,FEBE.xref[i])
    end    

end

function update!(C::XFunctionAction, FEBE::FEBasisEvaluator, item::Int)
    # compute global coordinates for function evaluation
    if FEBE.L2G.citem != item 
        FEXGrid.update!(FEBE.L2G, item)
    end
    # we don't know at contruction time how many quadrature points are needed
    # so we expand the array here if needed
    while length(C.x) < length(FEBE.xref)
        push!(C.x,deepcopy(FEBE.xref[1]))
    end  
    for i = 1 : length(FEBE.xref) 
        FEXGrid.eval!(C.x[i],FEBE.L2G,FEBE.xref[i])
    end    

end

function update!(C::Union{RegionWiseMultiplyVectorAction,RegionWiseMultiplyScalarAction}, item::Int)
    C.cregion = C.ItemRegions[item]
end

###
# apply_action! is called for each dof and i-th quadrature point
###


function apply_action!(result::Array{<:Real,1}, input::Array{<:Real,1}, C::DoNotChangeAction, i::Int = 0)
    for j = 1:length(result)
        result[j] = input[j];    
    end    
end

function apply_action!(result::Array{<:Real,1}, input::Array{<:Real,1}, C::MultiplyScalarAction, i::Int = 0)
    for j = 1:length(result)
        result[j] = input[j] * C.value;    
    end    
end

function apply_action!(result::Array{<:Real,1}, input::Array{<:Real,1}, C::RegionWiseMultiplyScalarAction, i::Int = 0)
    for j = 1:length(result)
        result[j] = input[j] * C.value[C.cregion];    
    end    
end

function apply_action!(result::Array{<:Real,1}, input::Array{<:Real,1}, C::MultiplyVectorAction, i::Int = 0)
    for j = 1:length(result)
        result[j] = input[j] * C.value[j];
    end    
end

function apply_action!(result::Array{<:Real,1}, input::Array{<:Real,1}, C::MultiplyMatrixAction, i::Int = 0)
    for j = 1:length(result)
        result[j] = 0
        for k = 1:length(input)
            result[j] += C.value[j,k]*input[k];
        end    
    end    
end

function apply_action!(result::Array{<:Real,1}, input::Array{<:Real,1}, C::RegionWiseMultiplyVectorAction, i::Int = 0)
    for j = 1:length(result)
        result[j] = input[j] * C.value[C.cregion][j];
    end    
end

function apply_action!(result::Array{<:Real,1}, input::Array{<:Real,1}, C::FunctionAction, i::Int)
    C.f(result, input);
end

function apply_action!(result::Array{<:Real,1}, input::Array{<:Real,1}, C::XFunctionAction, i::Int)
    C.f(result, input, C.x[i]);
end

function apply_action!(result::Array{<:Real,1}, input::Array{<:Real,1}, C::RegionWiseXFunctionAction, i::Int)
    C.f(result, input, C.x[i], C.cregion);
end