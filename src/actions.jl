
abstract type AbstractAction end


# dummy action that does nothing (but can be only used with certain assembly patterns)
struct NoAction <: AbstractAction
    name::String
    bonus_quadorder::Int
end

"""
````
function NoAction()
````

Creates a NoAction that causes the assembly pattern to ignore the action assembly.
"""
function NoAction(; name = "no action", quadorder = 0)
    return NoAction(name,quadorder)
end

# dummy array that is infinitely long and only has nothing entries
struct InfNothingArray
    val::Nothing
end
Base.getindex(::InfNothingArray,i) = nothing

mutable struct Action{T <: Real,KernelType <: UserData{<:AbstractActionKernel}, nsizes} <: AbstractAction
    kernel::KernelType
    name::String
    citem::Array{Int,1}
    cregion::Array{Int,1}
    bonus_quadorder::Int
    argsizes::SVector{nsizes,Int}
    xref::Union{InfNothingArray,Array{Array{T,1},1}}
end

# actions that do depend on t
mutable struct TAction{T <: Real,KernelType <: UserData{<:AbstractActionKernel},nsizes} <: AbstractAction
    kernel::KernelType
    name::String
    citem::Array{Int,1}
    cregion::Array{Int,1}
    ctime::Array{T,1}
    bonus_quadorder::Int
    argsizes::SVector{nsizes,Int}
    xref::Union{InfNothingArray,Array{Array{T,1},1}}
end

# actions that do depend on x
# and require additional managament of global evaluation points
mutable struct XAction{T <: Real,KernelType <: UserData{<:AbstractActionKernel},nsizes} <: AbstractAction
    kernel::KernelType
    name::String
    citem::Array{Int,1}
    cregion::Array{Int,1}
    bonus_quadorder::Int
    argsizes::SVector{nsizes,Int}
    xref::Union{InfNothingArray,Array{Array{T,1},1}}
    x::Array{Array{T,1},1}
end

# actions that do depend on x and t
# and require additional managament of global evaluation points
mutable struct XTAction{T <: Real,KernelType <: UserData{<:AbstractActionKernel},nsizes} <: AbstractAction
    kernel::KernelType
    name::String
    citem::Array{Int,1}
    cregion::Array{Int,1}
    ctime::Array{T,1}
    bonus_quadorder::Int
    argsizes::SVector{nsizes,Int}
    xref::Union{InfNothingArray,Array{Array{T,1},1}}
    x::Array{Array{T,1},1}
end

"""
````
function Action(
    T::Type{<:Real},
    kernel::UserData{<:AbstractActionKernel};
    name::String = "user action")
````

Creates an Action from a given specified action kernel that then can be used in an assembly pattern. T specifies the number format
that should match the number format of the used quadrature rules and grid coordinates in the mesh (usually Float64).
"""
function Action(T, kernel::UserData{<:AbstractActionKernel}; name = "user action")
    if is_xdependent(kernel)
        if is_timedependent(kernel)
            return XTAction{T,typeof(kernel),length(kernel.dimensions)}(kernel, name, [0], [0], zeros(T,1), kernel.quadorder, kernel.dimensions, InfNothingArray(nothing), Array{T,1}(undef,0))
        else
            return XAction{T,typeof(kernel),length(kernel.dimensions)}(kernel, name, [0], [0], kernel.quadorder, kernel.dimensions, InfNothingArray(nothing), Array{T,1}(undef,0))
        end
    else
        if is_timedependent(kernel)
            return TAction{T,typeof(kernel),length(kernel.dimensions)}(kernel, name, [0], [0], zeros(T,1), kernel.quadorder, kernel.dimensions, InfNothingArray(nothing))
        else
            return Action{T,typeof(kernel),length(kernel.dimensions)}(kernel, name, [0], [0], kernel.quadorder, kernel.dimensions, InfNothingArray(nothing))
        end
    end
end

"""
````
function Action(
    T::Type{<:Real},
    kernel_function::Function,
    dimensions::Array{Int,1};
    name = "user action",
    dependencies = "",
    quadorder = 0)
````

Creates an Action directly from a kernel function (plus additional information to complement the action kernel) that then can be used in an assembly pattern. T specifies the number format
that should match the number format of the used quadrature rules and grid coordinates in the mesh (usually Float64).
"""
function Action(T::Type{<:Real}, kernel_function::Function, dimensions; name = "user action", dependencies = "", quadorder = 0)
    kernel =  ActionKernel(kernel_function, dimensions; name = name * " (kernel)", dependencies = dependencies, quadorder = quadorder)
    return Action(T, kernel; name = name)
end

"""
````
function MultiplyScalarAction(value, ncomponents::Int)
````

Directly creates an Action that multiplies a scalar value to the input (vector of length ncomponents).
"""
function MultiplyScalarAction(value, ncomponents::Int, T::Type{<:Real} = Float64)
    function multiply_scalar_action_kernel(result, input)
        for j = 1 : ncomponents
            result[j] = input[j] * value
        end
        return nothing
    end
    kernel = ActionKernel(multiply_scalar_action_kernel,[ncomponents, ncomponents]; dependencies = "", quadorder = 0)
    return Action(T, kernel; name = "multiply scalar action")
end

# set_time! is called in the beginning of every operator assembly
function set_time!(C::AbstractAction, time)
    return nothing
end

function set_time!(C::Union{XTAction{T},TAction{T}}, time) where {T <: Real}
    if is_timedependent(C.kernel)
        C.ctime[1] = time
    end
    return nothing
end

function update!(C::AbstractAction, FEBE::FEBasisEvaluator, qitem, vitem, region)
    return nothing
end

function update!(C::Union{Action{T}, TAction{T}}, FEBE::FEBasisEvaluator, qitem::Int, vitem::Int, region) where {T <: Real}
    if is_regiondependent(C.kernel)
        C.cregion[1] = region
    end
    if is_itemdependent(C.kernel)
        C.citem[1] = vitem
    end
    if is_ldependent(C.kernel)
        C.xref = FEBE.xref
    end
    return nothing
end

function update!(C::Union{XAction{T}, XTAction{T}}, FEBE::FEBasisEvaluator, qitem::Int, vitem::Int, region) where {T <: Real}
    if is_regiondependent(C.kernel)
        C.cregion[1] = region
    end
    if is_itemdependent(C.kernel)
        C.citem[1] = vitem
    end
    if is_ldependent(C.kernel)
        C.xref = FEBE.xref
    end
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

function apply_action!(result::Array{T,1}, input::Array{T,1}, C::AbstractAction, i::Int) where {T <: Real}
    return nothing
end

function apply_action!(result::Array{T,1}, input::Array{T,1}, C::Action{T}, i::Int) where {T <: Real}
    eval!(result, C.kernel, input, nothing, nothing, C.cregion[1], C.citem[1], C.xref[i]);
    return nothing
end

function apply_action!(result::Array{T,1}, input::Array{T,1}, C::XAction{T}, i::Int) where {T <: Real}
    eval!(result, C.kernel, input, C.x[i], nothing, C.cregion[1], C.citem[1], C.xref[i]);
    return nothing
end

function apply_action!(result::Array{T,1}, input::Array{T,1}, C::TAction{T}, i::Int) where {T <: Real}
    eval!(result, C.kernel, input, nothing, C.ctime[1], C.cregion[1], C.citem[1], C.xref[i]);
    return nothing
end

function apply_action!(result::Array{T,1}, input::Array{T,1}, C::XTAction{T}, i::Int) where {T <: Real}
    eval!(result, C.kernel, input, C.x[i], C.ctime[1], C.cregion[1], C.citem[1], C.xref[i]);
    return nothing
end

## apply! function for action in nonlinear assemblies

function apply_action!(result::Array{<:Real,1}, input_last::Array{<:Real,1}, input_ansatz::Array{<:Real,1}, C::Action{T}, i::Int) where {T <: Real}
    eval!(result, C.kernel, input_last, input_ansatz, nothing, nothing, C.cregion[1], C.citem[1], C.xref[i]);
    return nothing
end

function apply_action!(result::Array{<:Real,1}, input_last::Array{<:Real,1}, input_ansatz::Array{<:Real,1}, C::XAction{T}, i::Int) where {T <: Real}
    eval!(result, C.kernel, input_last, input_ansatz, C.x[i], nothing, C.cregion[1], C.citem[1], C.xref[i]);
    return nothing
end

function apply_action!(result::Array{<:Real,1}, input_last::Array{<:Real,1}, input_ansatz::Array{<:Real,1}, C::TAction{T}, i::Int) where {T <: Real}
    eval!(result, C.kernel, input_last, input_ansatz, nothing, C.ctime[1], C.cregion[1], C.citem[1], C.xref[i]);
    return nothing
end

function apply_action!(result::Array{<:Real,1}, input_last::Array{<:Real,1}, input_ansatz::Array{<:Real,1}, C::XTAction{T}, i::Int) where {T <: Real}
    eval!(result, C.kernel, input_last, input_ansatz, C.x[i], C.ctime[1], C.cregion[1], C.citem[1], C.xref[i]);
    return nothing
end