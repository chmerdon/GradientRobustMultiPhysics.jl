


abstract type AbstractAction end

"""
$(TYPEDEF)

action that does nothing to the input, i.e.

result[j] = input[j]

(for j = 1 : argsizes[1])

"""
struct DoNotChangeAction <: AbstractAction
    argsizes::Array{Int,1}
    bonus_quadorder::Int
end

"""
$(TYPEDEF)

action that multiplies a scalar to the input, i.e.

result[j] = input[j] * value

(for j = 1 : argsizes[1])
"""
struct MultiplyScalarAction{T <: Real} <: AbstractAction
    value::T
    argsizes::Array{Int,1}
    bonus_quadorder::Int
end

"""
$(TYPEDEF)

action that multiplies a different scalar to each component of the input, i.e.

result[j] = input[j] * value[j]

(for j = 1 : argsizes[1])
"""
struct MultiplyVectorAction{T <: Real} <: AbstractAction
    value::AbstractArray{T,1}
    argsizes::Array{Int,1}
    bonus_quadorder::Int
end

"""
$(TYPEDEF)

action that multiplies input vector with a matrix, i.e.
(note that also gradients are transferred as vectors)

result = value * input

(resultdim = argsizes[1])

"""
struct MultiplyMatrixAction{T <: Real} <: AbstractAction
    value::AbstractArray{T,2}
    argsizes::Array{Int,1}
    bonus_quadorder::Int
end

"""
$(TYPEDEF)

action that multiplies a region-dependent scalar to the input, i.e.

result[j] = input[j] * value[region]

(for j = 1 : argsizes[1])
"""
struct RegionWiseMultiplyScalarAction{T <: Real} <: AbstractAction
    value::AbstractArray{T,1}
    cregion::Int
    argsizes::Array{Int,1}
    bonus_quadorder::Int
end

"""
$(TYPEDEF)

action that multiplies an item-dependent scalar to the input, i.e.

result[j] = input[j] * value[item]

(for j = 1 : argsizes[1])
"""
mutable struct ItemWiseMultiplyScalarAction{T <: Real} <: AbstractAction
    value::AbstractArray{T,1}
    citem::Int
    cvalue::T
    argsizes::Array{Int,1}
    bonus_quadorder::Int
end

"""
$(TYPEDEF)

action that multiplies a different region-dependent scalar to each component of the input, i.e.

result[j] = input[j] * value[region][j]

(for j = 1 : argsizes[1])
"""
struct RegionWiseMultiplyVectorAction{T <: Real} <: AbstractAction
    value::AbstractArray{AbstractArray{T,1},1} # one array for each region
    cregion::Int
    argsizes::Array{Int,1}
    bonus_quadorder::Int
end

"""
$(TYPEDEF)

action that puts input into the specified function f! and returns its result. The function f! should have the interface

    f!(result,input)

and result is expected to be of length argsizes[1], input of length argsizes[2].

The quadrature order of assemblies that involve this action can be altered with bonus_quadorder.
"""
struct FunctionAction{T <: Real} <: AbstractAction
    f!::Function # of the interface f!(result,input)
    argsizes::Array{Int,1}
    bonus_quadorder::Int
end

"""
$(TYPEDEF)

action that puts input into the specified function f! and returns its result. The function f! should have the interface

    f!(result,input,item)
        
and result is expected to be of length argsizes[1], input of length argsizes[2].

The quadrature order of assemblies that involve this action can be altered with bonus_quadorder.
"""
mutable struct ItemWiseFunctionAction{T <: Real} <: AbstractAction
    f!::Function # of the interface f!(result,input,item)
    citem::Int
    argsizes::Array{Int,1}
    bonus_quadorder::Int
end

"""
$(TYPEDEF)

action that puts input into the specified function f! and returns its result. The function f! should have the interface

    f!(result,input,x)
        
and result is expected to be of length argsizes[1], input of length argsizes[2].

The quadrature order of assemblies that involve this action can be altered with bonus_quadorder.
"""
struct XFunctionAction{T <: Real} <: AbstractAction
    f!::Function # of the interface f!(result,input,x)
    argsizes::Array{Int,1}
    x::Array{Array{T,1},1}
    bonus_quadorder::Int
end

"""
$(TYPEDEF)

action that puts input into the specified function f! and returns its result. The function f! should have the interface

    f!(result,input,x,item)
        
and result is expected to be of length argsizes[1], input of length argsizes[2].

The quadrature order of assemblies that involve this action can be altered with bonus_quadorder.
"""
mutable struct ItemWiseXFunctionAction{T <: Real} <: AbstractAction
    f!::Function # of the interface f!(result,input,x,item)
    citem::Int
    argsizes::Array{Int,1}
    x::Array{Array{T,1},1}
    bonus_quadorder::Int
end

"""
$(TYPEDEF)

action that puts input into the specified function f! and returns its result. The function f! should have the interface

    f!(result,input,x,region)
        
and result is expected to be of length argsizes[1], input of length argsizes[2].

The quadrature order of assemblies that involve this action can be altered with bonus_quadorder.
"""
mutable struct RegionWiseXFunctionAction{T <: Real} <: AbstractAction
    f!::Function # of the interface f!(result,input,x,region)
    cregion::Int
    argsizes::Array{Int,1}
    x::Array{Array{T,1},1}
    bonus_quadorder::Int
end

"""
````
function DoNotChangeAction(ncomponents::Int)
````

creates DoNotChangeAction with specified argsizes = (ncomponents, ncomponents).

"""
function DoNotChangeAction(ncomponents::Int)
    return DoNotChangeAction([ncomponents, ncomponents],0)
end


"""
````
function MultiplyScalarAction(value::Real, argsizes::Array{Int,1})
````

creates MultiplyScalarAction with specified argsizes = (length(result), length(input)).

"""
function MultiplyScalarAction(value::Real, argsizes::Array{Int,1})
    return MultiplyScalarAction{eltype(value)}(value, argsizes, 0)
end
function MultiplyScalarAction(value::Real, argsizes::Int)
    return MultiplyScalarAction{eltype(value)}(value, [argsizes, argsizes], 0)
end

"""
````
function MultiplyVectorAction(values::Array{<:Real,1}, argsizes::Array{Int,1})
````

creates MultiplyVectorAction with specified argsizes = (length(result), length(input)).

"""
function MultiplyVectorAction(values::AbstractArray{<:Real,1}, argsizes::Array{Int,1})
    return MultiplyVectorAction{eltype(values)}(values, argsizes, 0)
end

"""
````
function MultiplyMatrixAction(matrix::Array{<:Real,2}, argsizes::Array{Int,1})
````

creates MultiplyMatrixAction with specified argsizes = (length(result), length(input)).

"""
function MultiplyMatrixAction(matrix::Array{<:Real,2}, argsizes::Array{Int,1})
    return MultiplyMatrixAction{eltype(matrix)}(matrix, argsizes, 0)
end

"""
````
function RegionWiseMultiplyScalarAction(value4region::Array{<:Real,1}, argsizes::Array{Int,1})
````

creates RegionWiseMultiplyScalarAction with specified argsizes = (length(result), length(input)).

"""
function RegionWiseMultiplyScalarAction(value4region::AbstractArray{<:Real,1}, argsizes::Array{Int,1})
    return RegionWiseMultiplyScalarAction{eltype(value4region)}(value4region, 1, argsizes,0)
end

"""
````
function ItemWiseMultiplyScalarAction(value4item::Array{<:Real,1}, argsizes::Array{Int,1})
````

creates ItemWiseMultiplyScalarAction with specified argsizes = (length(result), length(input)).

"""
function ItemWiseMultiplyScalarAction(value4item::AbstractArray{<:Real,1}, argsizes::Array{Int,1})
    return ItemWiseMultiplyScalarAction{eltype(value4item)}(value4item, 1, 0, argsizes, 0)
end


"""
````
function RegionWiseMultiplyVectorAction(values4region::Array{Array{<:Real,1},1}, argsizes::Array{Int,1})
````

creates RegionWiseMultiplyVectorAction with specified argsizes = (length(result), length(input)).

"""
function RegionWiseMultiplyVectorAction(values4region::Array{Array{<:Real,1},1}, argsizes::Array{Int,1})
    return RegionWiseMultiplyVectorAction{eltype(values4region[1])}(values4region, 1, argsizes,0)
end

"""
````
function FunctionAction(f!::Function, argsizes::Array{Int,1}; bonus_quadorder::Int = 0)
````

creates FunctionAction.

"""
function FunctionAction(f!::Function, argsizes::Array{Int,1}; bonus_quadorder::Int = 0)
    return FunctionAction{Float64}(f!, argsizes, bonus_quadorder)
end


"""
````
function XFunctionAction(f!::Function, argsizes::Array{Int,1}, xdim::Int; bonus_quadorder::Int = 0)
````

creates XFunctionAction.

"""
function XFunctionAction(f!::Function, argsizes::Array{Int,1}, xdim::Int; bonus_quadorder::Int = 0)
    return XFunctionAction{Float64}(f!, argsizes, [zeros(Float64,xdim)], bonus_quadorder)
end


"""
````
function ItemWiseFunctionAction(f!::Function, argsizes::Array{Int,1}; bonus_quadorder::Int = 0)
````

creates ItemWiseFunctionAction.

"""
function ItemWiseFunctionAction(f!::Function, argsizes::Array{Int,1}; bonus_quadorder::Int = 0)
    return ItemWiseFunctionAction{Float64}(f!, 0, argsizes, bonus_quadorder)
end

"""
````
function ItemWiseXFunctionAction(f!::Function, argsizes::Array{Int,1}, xdim::Int; bonus_quadorder::Int = 0)
````

creates ItemWiseXFunctionAction.

"""
function ItemWiseXFunctionAction(f!::Function, argsizes::Array{Int,1}, xdim::Int; bonus_quadorder::Int = 0)
    return ItemWiseXFunctionAction{Float64}(f!, 0, argsizes, [zeros(Float64,xdim)], bonus_quadorder)
end

"""
````
function RegionWiseXFunctionAction(f!::Function, argsizes::Array{Int,1}, xdim::Int; bonus_quadorder::Int = 0)
````

creates RegionWiseXFunctionAction.

"""
function RegionWiseXFunctionAction(f!::Function, argsizes::Array{Int,1}, xdim::Int; bonus_quadorder::Int = 0)
    return RegionWiseXFunctionAction{Float64}(f!, 1, argsizes, [zeros(Float64,xdim)], bonus_quadorder)
end

###
# update! is called on each change of item 
###

function update!(C::AbstractAction, FEBE::FEBasisEvaluator, qitem::Int, vitem::Int, region)
    # do nothing
end

function update!(C::RegionWiseXFunctionAction, FEBE::FEBasisEvaluator, qitem::Int, vitem::Int, region)
    C.cregion = region

    # compute global coordinates for function evaluation
    if FEBE.L2G.citem != qitem 
        update!(FEBE.L2G, qitem)
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


function update!(C::ItemWiseXFunctionAction, FEBE::FEBasisEvaluator, qitem::Int, vitem::Int, region)
    C.citem = vitem
    # compute global coordinates for function evaluation
    if FEBE.L2G.citem != qitem 
        update!(FEBE.L2G, qitem)
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

function update!(C::XFunctionAction, FEBE::FEBasisEvaluator, qitem::Int, vitem::Int, region)
    # compute global coordinates for function evaluation
    if FEBE.L2G.citem != qitem 
        update!(FEBE.L2G, qitem)
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

function update!(C::ItemWiseFunctionAction, FEBE::FEBasisEvaluator, qitem::Int, vitem::Int, region)
    C.citem = copy(vitem)
end

function update!(C::ItemWiseMultiplyScalarAction, FEBE::FEBasisEvaluator, qitem::Int, vitem::Int, region)
    C.cvalue = C.value[C.citem]
end

function update!(C::Union{RegionWiseMultiplyVectorAction,RegionWiseMultiplyScalarAction}, FEBE::FEBasisEvaluator, qitem::Int, vitem::Int, region)
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

function apply_action!(result::Array{<:Real,1}, input::Array{<:Real,1}, C::ItemWiseMultiplyScalarAction, i::Int = 0)
    for j = 1:length(result)
        result[j] = input[j] * C.cvalue;    
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
    C.f!(result, input);
end

function apply_action!(result::Array{<:Real,1}, input::Array{<:Real,1}, C::XFunctionAction, i::Int)
    C.f!(result, input, C.x[i]);
end

function apply_action!(result::Array{<:Real,1}, input::Array{<:Real,1}, C::RegionWiseXFunctionAction, i::Int)
    C.f!(result, input, C.x[i], C.cregion);
end

function apply_action!(result::Array{<:Real,1}, input::Array{<:Real,1}, C::ItemWiseXFunctionAction, i::Int)
    C.f!(result, input, C.x[i], C.citem);
end

function apply_action!(result::Array{<:Real,1}, input::Array{<:Real,1}, C::ItemWiseFunctionAction, i::Int)
    C.f!(result, input, C.citem);
end



## apply! function for action in nonlinear assemblies

function apply_action!(result::Array{<:Real,1}, input_last::Array{<:Real,1}, input_ansatz::Array{<:Real,1}, C::FunctionAction, i::Int)
    C.f!(result, input_last, input_ansatz);
end