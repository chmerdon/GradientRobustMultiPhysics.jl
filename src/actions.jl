
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
        name, kernel, argsizes, zeros(Tv, 3), zeros(Tv, 3), zeros(Ti,5), 0, bonus_quadorder, zeros(Tv, argsizes[1]))
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
function NoAction(; name = "no action", bonus_quadorder = 0)
    return NoAction(name, bonus_quadorder)
end

"""
````
function fdot_action(data::UserData) -> Action
````

Creates an action that just evaluates the DataFunction f.
"""
function fdot_action(data::AbstractUserDataType; Tv = Float64, Ti = Int32)
    f_as_action(result, input, kwargs...) = data.kernel(result, kwargs...)
    dependencies = is_xdependent(data) ? "X" : ""
    dependencies *= is_timedependent(data) ? "T" : ""
    dependencies *= is_itemdependent(data) ? "I" : ""
    dependencies *= is_xrefdependent(data) ? "L" : ""
    action = Action(f_as_action, [data.argsizes[1], 0]; Tv = Tv, Ti = Ti, dependencies = dependencies, bonus_quadorder = data.bonus_quadorder)
    set_time!(action, data.time)
    return action
end

"""
````
function fdotv_action(data::AbstractUserDataType) -> Action
````

Creates an action that evaluates the DataFunction f and performs a vector product with the input.
"""
function fdotv_action(data::AbstractUserDataType; Tv = Float64, Ti = Int32)
    feval = zeros(Tv, data.argsizes[1])
    function vdotg_kernel(result, input, kwargs...)
        data.kernel(feval, kwargs...)
        result[1] = dot(feval, input)
    end
    action = Action(vdotg_kernel, [1 data.argsizes[1]]; Tv = Tv, Ti = Ti, dependencies = dependencies(data), name = "$(data.name)⋅v")
    set_time!(action, data.time)
    return action
end


"""
````
function feval_action(data::AbstractUserDataType) -> Action
````

Creates an action that evaluates the DataFunction f in the input.
"""
function feval_action(data::AbstractUserDataType; Tv = Float64, Ti = Int32)
    feval = zeros(Tv, data.argsizes[1])
    function feval_kernel(result, input, kwargs...)
        data.kernel(feval, input, kwargs...)
        result .= feval
    end
    action = Action(feval_kernel, data.argsizes; Tv = Tv, Ti = Ti, dependencies = dependencies(data), name = "$(data.name)⋅v")
    set_time!(action, data.time)
    return action
end


"""
````
function fdotv_action(data::AbstractUserDataType) -> Action
````

Creates an action that evaluates the DataFunction f times the normal vector.
"""
function fdotn_action(data::AbstractUserDataType, xgrid; Tv = Float64, Ti = Int32, bfaces = false)
    xFaceNormals = xgrid[FaceNormals]
    xBFaceFaces = xgrid[BFaceFaces]
    function vdotg_kernel(result, input, kwargs...)
        eval_data!(data)
        if bfaces
            result[1] = dot(data.val, view(xFaceNormals,:,xBFaceFaces[data.item[1]]))
        else
            result[1] = dot(data.val, view(xFaceNormals,:,data.item[1]))
        end
        return nothing
    end
    action = Action(vdotg_kernel, [1 0]; Tv = Tv, Ti = Ti, dependencies = dependencies(data; enforce = "I"), bonus_quadorder = data.bonus_quadorder, name = "$(data.name)⋅n")
    data.x = action.x
    data.xref = action.xref
    data.item = action.item
    return action
end


"""
````
function fdotv_action(data::AbstractUserDataType) -> Action
````

Creates an action that evaluates the DataFunction f times the tangential vector on 2D faces.
"""
function fdott2d_action(data::AbstractUserDataType, xgrid; Tv = Float64, Ti = Int32, bfaces = false)
    xFaceNormals = xgrid[FaceNormals]
    xBFaceFaces = xgrid[BFaceFaces]
    function vdott_kernel(result, input, kwargs...)
        eval_data!(data)
        if bfaces
            result[1] = -data.val[1] * xFaceNormals[2,xBFaceFaces[item[1]]]
            result[1] += data.val[2] * xFaceNormals[1,xBFaceFaces[item[1]]]
        else
            result[1] = -data.val[1] * xFaceNormals[2,item[1]]
            result[1] += data.val[2] * xFaceNormals[1,item[1]]
        end
        return nothing
    end
    action = Action(vdott_kernel, [1 0]; Tv = Tv, Ti = Ti, dependencies = dependencies(data; enforce = "I"), bonus_quadorder = data.bonus_quadorder, name = "$(data.name)⋅n")
    data.x = action.x
    data.xref = action.xref
    data.item = action.item
    return action
end


"""
````
function fdotv_action(data::AbstractUserDataType) -> Action
````

Creates an action that evaluates the DataFunction f times the tangential vector on 3D edges.
"""
function fdott3d_action(data::AbstractUserDataType, xgrid; Tv = Float64, Ti = Int32, bedges = false)
    xEdgeTangents = FE.xgrid[EdgeTangents]
    xBEdgeEdges = FE.xgrid[BEdgeEdges]
    function vdott_kernel(result, input, kwargs...)
        eval_data!(data)
        if bedges
            result[1] = temp[1] * xEdgeTangents[1,xBEdgeEdges[item[1]]]
            result[1] += temp[2] * xEdgeTangents[2,xBEdgeEdges[item[1]]]
            result[1] += temp[3] * xEdgeTangents[3,xBEdgeEdges[item[1]]]
        else
            result[1] = temp[1] * xEdgeTangents[1,item[1]]
            result[1] += temp[2] * xEdgeTangents[2,item[1]]
            result[1] += temp[3] * xEdgeTangents[3,item[1]]
        end
        return nothing
    end
    action = Action(vdott_kernel, [1 0]; Tv = Tv, Ti = Ti, dependencies = dependencies(data; enforce = "I"), bonus_quadorder = data.bonus_quadorder, name = "$(data.name)⋅n")
    data.x = action.x
    data.xref = action.xref
    data.item = action.item
    return action
end