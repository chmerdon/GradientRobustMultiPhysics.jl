


abstract type AbstractAction end

"""
$(TYPEDEF)

action that does nothing to the input
"""
struct DoNotChangeAction <: AbstractAction
    resultdim::Int
    bonus_quadorder::Int
end

"""
$(TYPEDEF)

action that multiplies a scalar to the input
"""
struct MultiplyScalarAction{T <: Real} <: AbstractAction
    value::T
    resultdim::Int
    bonus_quadorder::Int
end

"""
$(TYPEDEF)

action that multiplies a different scalar to each component of the input
"""
struct MultiplyVectorAction{T <: Real} <: AbstractAction
    value::Array{T,1}
    resultdim::Int
    bonus_quadorder::Int
end

"""
$(TYPEDEF)

action that multiplies input vector with a matrix
(note that also gradients are transferred as vectors)
"""
struct MultiplyMatrixAction{T <: Real} <: AbstractAction
    value::Array{T,2}
    resultdim::Int
    bonus_quadorder::Int
end

"""
$(TYPEDEF)

action that multiplies a ergion-dependent scalar to the input
"""
struct RegionWiseMultiplyScalarAction{T <: Real} <: AbstractAction
    value::Array{T,1} # one array for each region
    cregion::Int
    resultdim::Int
    bonus_quadorder::Int
end

"""
$(TYPEDEF)

action that multiplies a different region-dependent scalar to each component of the input
"""
struct RegionWiseMultiplyVectorAction{T <: Real} <: AbstractAction
    value::Array{Array{T,1},1} # one array for each region
    cregion::Int
    resultdim::Int
    bonus_quadorder::Int
end

"""
$(TYPEDEF)

action that puts input into the given function and returns the result
"""
struct FunctionAction{T <: Real} <: AbstractAction
    f::Function # of the interface f!(result,input)
    resultdim::Int
    bonus_quadorder::Int
end

"""
$(TYPEDEF)

action that puts input into the given function, that can additionally depend on the global coordinate x, and returns the result
"""
struct XFunctionAction{T <: Real} <: AbstractAction
    f::Function # of the interface f!(result,input,x)
    resultdim::Int
    x::Array{Array{T,1},1}
    bonus_quadorder::Int
end

"""
$(TYPEDEF)

action that puts input into the given function, that can additionally depend on the item number, and returns the result
"""
mutable struct ItemWiseXFunctionAction{T <: Real} <: AbstractAction
    f::Function # of the interface f!(result,input,x,item)
    citem::Int
    resultdim::Int
    x::Array{Array{T,1},1}
    bonus_quadorder::Int
end

"""
$(TYPEDEF)

action that puts input into the given function, that can additionally depend on the region number, and returns the result
"""
mutable struct RegionWiseXFunctionAction{T <: Real} <: AbstractAction
    f::Function # of the interface f!(result,input,x,region)
    cregion::Int
    resultdim::Int
    x::Array{Array{T,1},1}
    bonus_quadorder::Int
end

"""
````
function DoNotChangeAction(resultdim::Int)
````

creates action that does nothing. Nevertheless resultdim has to be specified.

"""
function DoNotChangeAction(resultdim::Int)
    return DoNotChangeAction(resultdim,0)
end

"""
````
function MultiplyScalarAction(value::Real, resultdim::Int = 1)
````

creates action that multiplies the first resultdim entries of the input vector with the specified valuse.

"""
function MultiplyScalarAction(value::Real, resultdim::Int = 1)
    return MultiplyScalarAction{eltype(value)}(value, resultdim,0)
end

"""
````
function MultiplyVectorAction(values::Array{<:Real,1})
````

creates action that multiplies input vector and values element-wise (until length of values).

"""
function MultiplyVectorAction(values::Array{<:Real,1})
    return MultiplyVectorAction{eltype(values)}(values, length(value),0)
end

"""
````
function MultiplyMatrixAction(matrix::Array{<:Real,2})
````

creates action that return the matrix vector product of the input vector and the specified matrix.

"""
function MultiplyMatrixAction(matrix::Array{<:Real,2})
    return MultiplyMatrixAction{eltype(matrix)}(matrix, size(matrix,1),0)
end

"""
````
function RegionWiseMultiplyScalarAction(value4region::Array{<:Real,1}, resultdim::Int = 1)
````

creates action that multiplies the first resultdim entries of the input vector with the specified value in the current region.

"""
function RegionWiseMultiplyScalarAction(value4region::Array{<:Real,1}, resultdim::Int = 1)
    return RegionWiseMultiplyScalarAction{eltype(value4region)}(value4region, 1, resultdim,0)
end

"""
````
function RegionWiseMultiplyVectorAction(values4region::Array{Array{<:Real,1},1}, resultdim::Int)
````

creates action that multiplies input vector and the values4region of the current region element-wise.

"""
function RegionWiseMultiplyVectorAction(values4region::Array{Array{<:Real,1},1}, resultdim::Int)
    return RegionWiseMultiplyVectorAction{eltype(values4region[1])}(values4region, 1, length(values4region[1]),0)
end

"""
````
function FunctionAction(f::Function, resultdim::Int = 1, xdim::Int = 2; bonus_quadorder::Int = 0)
````

creates action that puts input into the specified function and return its result. The function should have the interface
    function!(result,input)
and result is expected to be of length resultdim. The quadrature order of assemblies that involve this action can be altered with bonus_quadorder.

"""
function FunctionAction(f::Function, resultdim::Int = 1, xdim::Int = 2; bonus_quadorder::Int = 0)
    return FunctionAction{Float64}(f, resultdim, bonus_quadorder)
end


"""
````
function XFunctionAction(f::Function, resultdim::Int = 1, xdim::Int = 2; bonus_quadorder::Int = 0)
````

creates action that puts input into the specified function and return its result. The function can depend on x and should have the interface
    function!(result,input,x)
and result is expected to be of length resultdim. The quadrature order of assemblies that involve this action can be altered with bonus_quadorder.

"""
function XFunctionAction(f::Function, resultdim::Int = 1, xdim::Int = 2; bonus_quadorder::Int = 0)
    return XFunctionAction{Float64}(f, resultdim, [zeros(Float64,xdim)], bonus_quadorder)
end


"""
````
function ItemWiseXFunctionAction(f::Function, resultdim::Int = 1, xdim::Int = 2; bonus_quadorder::Int = 0)
````

creates action that puts input into the specified function and return its result. The function can depend on x and the item number and should have the interface
    function!(result,input,x,item)
and result is expected to be of length resultdim. The quadrature order of assemblies that involve this action can be altered with bonus_quadorder.

"""
function ItemWiseXFunctionAction(f::Function, resultdim::Int = 1, xdim::Int = 2; bonus_quadorder::Int = 0)
    return ItemWiseXFunctionAction{Float64}(f, 0, resultdim, [zeros(Float64,xdim)], bonus_quadorder)
end

"""
````
function RegionWiseXFunctionAction(f::Function, resultdim::Int = 1, xdim::Int = 2; bonus_quadorder::Int = 0)
````

creates action that puts input into the specified function and return its result. The function can depend on x and the region number and should have the interface
    function!(result,input,x,region)
and result is expected to be of length resultdim. The quadrature order of assemblies that involve this action can be altered with bonus_quadorder.

"""
function RegionWiseXFunctionAction(f::Function, resultdim::Int = 1, xdim::Int = 2; bonus_quadorder::Int = 0)
    return RegionWiseXFunctionAction{Float64}(f, 1, resultdim, [zeros(Float64,xdim)], bonus_quadorder)
end

###
# update! is called on each change of item 
###

function update!(C::AbstractAction, FEBE::FEBasisEvaluator, item::Int, region)
    # do nothing
end

function update!(C::RegionWiseXFunctionAction, FEBE::FEBasisEvaluator, item::Int, region)
    C.cregion = region

    # compute global coordinates for function evaluation
    if FEBE.L2G.citem != item 
        update!(FEBE.L2G, item)
    end
    # we don't know at contruction time how many quadrature points are needed
    # so we expand the array here if needed
    while length(C.x) < length(FEBE.xref)
        push!(C.x,deepcopy(C.x[1]))
    end  
    for i = 1 : length(FEBE.xref) 
        eval!(C.x[i],FEBE.L2G,FEBE.xref[i])
    end    

end


function update!(C::ItemWiseXFunctionAction, FEBE::FEBasisEvaluator, item::Int, region)
    C.citem = item
    # compute global coordinates for function evaluation
    if FEBE.L2G.citem != item 
        update!(FEBE.L2G, item)
    end
    # we don't know at contruction time how many quadrature points are needed
    # so we expand the array here if needed
    while length(C.x) < length(FEBE.xref)
        push!(C.x,deepcopy(C.x[1]))
    end  
    for i = 1 : length(FEBE.xref) 
        eval!(C.x[i],FEBE.L2G,FEBE.xref[i])
    end    

end

function update!(C::XFunctionAction, FEBE::FEBasisEvaluator, item::Int, region)
    # compute global coordinates for function evaluation
    if FEBE.L2G.citem != item 
        update!(FEBE.L2G, item)
    end
    # we don't know at contruction time how many quadrature points are needed
    # so we expand the array here if needed
    while length(C.x) < length(FEBE.xref)
        push!(C.x,deepcopy(C.x[1]))
    end  
    for i = 1 : length(FEBE.xref) 
        eval!(C.x[i],FEBE.L2G,FEBE.xref[i])
    end    

end

function update!(C::Union{RegionWiseMultiplyVectorAction,RegionWiseMultiplyScalarAction}, item::Int, region)
    C.cregion = region
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
        result[j] = input[j] * C.value
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

function apply_action!(result::Array{<:Real,1}, input::Array{<:Real,1}, C::ItemWiseXFunctionAction, i::Int)
    C.f(result, input, C.x[i], C.citem);
end