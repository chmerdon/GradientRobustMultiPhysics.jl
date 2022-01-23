
abstract type AbstractAction end

is_xdependent(A::AbstractAction) = false
is_itemdependent(A::AbstractAction) = false
is_xrefdependent(A::AbstractAction) = false
set_time!(A::AbstractAction, time) = ()
eval_action!(A::AbstractAction, input) = ()

struct ItemInformation{Ti}
    item::Int
    parent::Int
    region::Int
end

mutable struct DefaultUserAction{T,Ti,dx,dt,di,dl,ndim,KernelType} <: AbstractAction
    name::String
    kernel::KernelType
    argsizes::SVector{ndim,Int}
    x::Vector{T}
    xref::Vector{T}
    item::Array{Ti,1}   # contains item number, parent item, region number
    time::T
    bonus_quadorder::Int
    val::Array{T,1}
end

set_time!(A::DefaultUserAction{T,Ti,dx,true,di,dl,ndim}, time) where {T,Ti,dx,di,dl,ndim} = (A.time = time)

is_xdependent(A::DefaultUserAction{T,Ti,dx,dt,di,dl,ndim}) where {T,Ti,dx,dt,di,dl,ndim} = dx
is_timedependent(A::DefaultUserAction{T,Ti,dx,dt,di,dl,ndim}) where {T,Ti,dx,dt,di,dl,ndim} = dt
is_itemdependent(A::DefaultUserAction{T,Ti,dx,dt,di,dl,ndim}) where {T,Ti,dx,dt,di,dl,ndim} = di
is_xrefdependent(A::DefaultUserAction{T,Ti,dx,dt,di,dl,ndim}) where {T,Ti,dx,dt,di,dl,ndim} = dl


"""
$(TYPEDSIGNATURES)

generates an Action that can be used in the construction of PDEoperators and essentially consists of a kernel function
specified by the user plus additional information on argument dimensions and additional dependencies:

- kernel   : Function with interface (result, input, ...)
- argsizes : expected lengths of [result, interface]

Optional arguments:

- dependencies    : substring of "XTIL" that specifies if the kernel also depends on space coordinates (X), time (T), item (I), local coordinates (L)
- bonus_quadorder : is added to the quadrature order computed based on the used FESpaces during assembly
- name            : name of this Action used in print messages
- Tv              : expected NumberType for result/input
- Ti              : expected NumberType for grid enumeration infos (e.g. item/region numbers when "I" dependecy is used)
"""
function Action(kernel::Function, argsizes; Tv = Float64, Ti = Int32, dependencies = "", bonus_quadorder = 0, name = "user action")
    dx = occursin("X", dependencies)
    dt = occursin("T", dependencies)
    di = occursin("I", dependencies)
    dl = occursin("L", dependencies)
    return DefaultUserAction{Tv,Ti,dx,dt,di,dl,length(argsizes),typeof(kernel)}(
        name, kernel, argsizes, zeros(Tv, 3), zeros(Tv, 3), zeros(Ti,3), 0, bonus_quadorder, zeros(Tv, argsizes[1]))
end

function eval_action!(A::DefaultUserAction{T,Ti,false,false,false,false}, input_current) where {T,Ti}
    A.kernel(A.val, input_current)
    return nothing
end
function eval_action!(A::DefaultUserAction{T,Ti,true,false,false,false}, input_current) where {T,Ti}
    A.kernel(A.val, input_current, A.x)
    return nothing
end
function eval_action!(A::DefaultUserAction{T,Ti,false,true,false,false}, input_current) where {T,Ti}
    A.kernel(A.val, input_current, A.time)
    return nothing
end
function eval_action!(A::DefaultUserAction{T,Ti,true,true,false,false}, input_current) where {T,Ti}
    A.kernel(A.val, input_current, A.x, A.time)
    return nothing
end
function eval_action!(A::DefaultUserAction{T,Ti,false,false,true,false}, input_current) where {T,Ti}
    A.kernel(A.val, input_current, A.item)
    return nothing
end
function eval_action!(A::DefaultUserAction{T,Ti,true,false,true,false}, input_current) where {T,Ti}
    A.kernel(A.val, input_current, A.x, A.item)
    return nothing
end
function eval_action!(A::DefaultUserAction{T,Ti,true,true,true,false}, input_current) where {T,Ti}
    A.kernel(A.val, input_current, A.x, A.time, A.item)
    return nothing
end


# dummy action that does nothing (but can be only used with certain assembly patterns)
mutable struct NoAction <: AbstractAction
    name::String
    bonus_quadorder::Int
end

"""
````
function NoAction()
````

Creates a NoAction that causes the assembly pattern to ignore the action-related assembly.
"""
function NoAction(; name = "no action", quadorder = 0)
    return NoAction(name,quadorder)
end

"""
````
function fdot_action(data::UserData) -> Action
````

Creates an action that just evaluates the DataFunction f.
"""
function fdot_action(data::UserData; Tv = Float64, Ti = Int32)
    f_as_action(result, input, kwargs...) = data.user_function(result, kwargs...)
    return Action(f_as_action, [data.dimensions[1], 0]; Tv = Tv, Ti = Ti, dependencies = data.dependencies, bonus_quadorder = data.quadorder)
end

"""
````
function fdotv_action(data::UserData) -> Action
````

Creates an action that evaluates the DataFunction f and performs a vector product with the input.
"""
function fdotv_action(data::UserData; Tv = Float64, Ti = Int32)
    feval = zeros(Tv, data.dimensions[1])
    function vdotg_kernel(result, input, kwargs...)
        data.user_function(feval, kwargs...)
        result[1] = dot(feval, input)
    end
    return Action(vdotg_kernel, [1 data.dimensions[2]]; dependencies = data.dependencies, name = "v⋅g")
end