
abstract type AbstractAction end

mutable struct Action{T <: Real} <: AbstractAction
    kernel::UserData{<:AbstractActionKernel}
    name::String
    citem::Int
    cregion::Int
    bonus_quadorder::Int
    argsizes::Array{Int,1}
    xref::Array{Array{T,1},1}
end

# actions that do depend on x
# and require additional managament of global evaluation points
mutable struct TAction{T <: Real} <: AbstractAction
    kernel::UserData{<:AbstractActionKernel}
    name::String
    citem::Int
    cregion::Int
    ctime::T
    bonus_quadorder::Int
    argsizes::Array{Int,1}
    xref::Array{Array{T,1},1}
end

# actions that do depend on x
# and require additional managament of global evaluation points
mutable struct XAction{T <: Real} <: AbstractAction
    kernel::UserData{<:AbstractActionKernel}
    name::String
    citem::Int
    cregion::Int
    bonus_quadorder::Int
    argsizes::Array{Int,1}
    xref::Array{Array{T,1},1}
    x::Array{Array{T,1},1}
end

# actions that do depend on x
# and require additional managament of global evaluation points
mutable struct XTAction{T <: Real} <: AbstractAction
    kernel::UserData{<:AbstractActionKernel}
    name::String
    citem::Int
    cregion::Int
    ctime::T
    bonus_quadorder::Int
    argsizes::Array{Int,1}
    xref::Array{Array{T,1},1}
    x::Array{Array{T,1},1}
end

"""
````
function Action(
    T::Type{<:Real},
    kernel::UserData{<:AbstractActionKernel};
    name::String = "user action"
````

Creates an Action from a given specified action kernel that then can be used in an assembly pattern. T specifies the number format
that should match the number format of the used quadrature rules and grid coordinates in the mesh (usually Float64).
"""
function Action(T, kernel::UserData{<:AbstractActionKernel}; name = "user action")
    if is_xdependent(kernel)
        if is_timedependent(kernel)
            return XTAction{T}(kernel, name, 0, 0, 0, kernel.quadorder, kernel.dimensions, Array{T,1}(undef,0), Array{T,1}(undef,0))
        else
            return XAction{T}(kernel, name, 0, 0, kernel.quadorder, kernel.dimensions, Array{T,1}(undef,0), Array{T,1}(undef,0))
        end
    else
        if is_timedependent(kernel)
            return TAction{T}(kernel, name, 0, 0, 0, kernel.quadorder, kernel.dimensions, Array{T,1}(undef,0))
        else
            return Action{T}(kernel, name, 0, 0, kernel.quadorder, kernel.dimensions, Array{T,1}(undef,0))
        end
    end
end

"""
````
function MultiplyScalarAction(value, ncomponents::Int)
````

Directly creates an Action that multiplies a scalar value to the input (vector of length ncomponents).
"""
function MultiplyScalarAction(value, ncomponents::Int)
    function multiply_scalar_action_kernel(result, input)
        for j = 1 : ncomponents
            result[j] = input[j] * value
        end
        return nothing
    end
    kernel = ActionKernel(multiply_scalar_action_kernel,[ncomponents, ncomponents]; dependencies = "", quadorder = 0)
    return Action(Float64, kernel; name = "multiply scalar action")
end

"""
````
function MultiplyScalarAction(value, ncomponents::Int)
````

Directly creates an Action that just copies the input to the result.
"""
function DoNotChangeAction(ncomponents::Int)
    function do_nothing_kernel(result, input)
        for j = 1 : ncomponents
            result[j] = input[j]
        end
        return nothing
    end
    kernel = ActionKernel(do_nothing_kernel,[ncomponents, ncomponents]; dependencies = "", quadorder = 0)
    return Action(Float64, kernel; name = "do nothing action")
end

# set_time! is called in the beginning of every operator assembly
function set_time!(C::AbstractAction, time)
    return nothing
end

function set_time!(C::Union{XTAction{T},TAction{T}}, time::T) where {T <: Real}
    C.ctime = time
    return nothing
end

function update!(C::Union{Action{T}, TAction{T}}, FEBE::FEBasisEvaluator, qitem::Int, vitem::Int, region) where {T <: Real}
    C.cregion = region
    C.citem = vitem
    C.xref = FEBE.xref
    return nothing
end

function update!(C::Union{XAction{T}, XTAction{T}}, FEBE::FEBasisEvaluator, qitem::Int, vitem::Int, region) where {T <: Real}
    C.cregion = region
    C.citem = vitem
    C.xref = FEBE.xref

    # compute global coordinates for function evaluation
    if FEBE.L2G.citem != qitem 
        update!(FEBE.L2G, qitem)
    end
    # we don't know at contruction time how many quadrature points are needed
    # so we expand the array here if needed
    while length(C.x) < length(FEBE.xref)
        push!(C.x,zeros(T,size(FEBE.FE.xgrid[Coordinates],1)))
    end  
    for i = 1 : length(FEBE.xref) 
        eval!(C.x[i],FEBE.L2G,FEBE.xref[i])
    end    
    return nothing
end

function apply_action!(result::Array{<:Real,1}, input::Array{<:Real,1}, C::Action{T}, i::Int) where {T <: Real}
    eval!(result, C.kernel, input, nothing, nothing, C.cregion, C.citem, C.xref[i]);
    return nothing
end

function apply_action!(result::Array{<:Real,1}, input::Array{<:Real,1}, C::XAction{T}, i::Int) where {T <: Real}
    eval!(result, C.kernel, input, C.x[i], nothing, C.cregion, C.citem, C.xref[i]);
    return nothing
end

function apply_action!(result::Array{<:Real,1}, input::Array{<:Real,1}, C::TAction{T}, i::Int) where {T <: Real}
    eval!(result, C.kernel, input, nothing, C.ctime, C.cregion, C.citem, C.xref[i]);
    return nothing
end

function apply_action!(result::Array{<:Real,1}, input::Array{<:Real,1}, C::XTAction{T}, i::Int) where {T <: Real}
    eval!(result, C.kernel, input, C.x[i], C.ctime, C.cregion, C.citem, C.xref[i]);
    return nothing
end

## apply! function for action in nonlinear assemblies

function apply_action!(result::Array{<:Real,1}, input_last::Array{<:Real,1}, input_ansatz::Array{<:Real,1}, C::Action{T}, i::Int) where {T <: Real}
    eval!(result, C.kernel, input_last, input_ansatz, nothing, nothing, C.cregion, C.citem, C.xref[i]);
    return nothing
end

function apply_action!(result::Array{<:Real,1}, input_last::Array{<:Real,1}, input_ansatz::Array{<:Real,1}, C::XAction{T}, i::Int) where {T <: Real}
    eval!(result, C.kernel, input_last, input_ansatz, C.x[i], nothing, C.cregion, C.citem, C.xref[i]);
    return nothing
end

function apply_action!(result::Array{<:Real,1}, input_last::Array{<:Real,1}, input_ansatz::Array{<:Real,1}, C::TAction{T}, i::Int) where {T <: Real}
    eval!(result, C.kernel, input_last, input_ansatz, nothing, C.ctime, C.cregion, C.citem, C.xref[i]);
    return nothing
end

function apply_action!(result::Array{<:Real,1}, input_last::Array{<:Real,1}, input_ansatz::Array{<:Real,1}, C::XTAction{T}, i::Int) where {T <: Real}
    eval!(result, C.kernel, input_last, input_ansatz, C.x[i], C.ctime, C.cregion, C.citem, C.xref[i]);
    return nothing
end