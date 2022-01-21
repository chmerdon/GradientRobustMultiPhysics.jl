
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


function Action(kernel::Function, argsizes; Tv = Float64, Ti = Int32, dependencies = "", bonus_quadorder = 0, name = "user action") where {T}
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

Creates a NoAction that causes the assembly pattern to ignore the action assembly.
"""
function NoAction(; name = "no action", quadorder = 0)
    return NoAction(name,quadorder)
end


function fdot_action(T, data::UserData)
    ncomponents = data.dimensions[1]
    if typeof(data) <: UserData{AbstractExtendedDataFunction}
        function rhs_function_ext() # result = F(v) = f*operator(v) = f*input
            temp = zeros(T,ncomponents)
            function closure(result,input,x, t, region, item, xref)
                eval_data!(temp, data, x, t, region, item, xref)
                result[1] = 0
                for j = 1 : ncomponents
                    result[1] += temp[j]*input[j] 
                end
                return nothing
            end
        end    
        action = Action(rhs_function_ext(),[1, ncomponents]; dependencies = "XTRIL", bonus_quadorder = data.quadorder)
    else
        if data.dependencies == "XT"
            function rhs_function_xt() # result = F(v) = f*operator(v) = f*input
                temp = zeros(T,ncomponents)
                function closure(result,input,x, t)
                    eval_data!(temp, data, x, t)
                    result[1] = 0
                    for j = 1 : ncomponents
                        result[1] += temp[j]*input[j] 
                    end
                    return nothing
                end
            end    
            action = Action(rhs_function_xt(),[1, ncomponents]; dependencies = "XT", bonus_quadorder = data.quadorder)
        elseif data.dependencies == "X"
            function rhs_function_x() # result = F(v) = f*operator(v) = f*input
                temp = zeros(T,ncomponents)
                function closure(result,input,x)
                    eval_data!(temp, data, x, nothing)
                    result[1] = 0
                    for j = 1 : ncomponents
                        result[1] += temp[j]*input[j] 
                    end
                    return nothing
                end
            end    
            action = Action(rhs_function_x(),[1, ncomponents]; dependencies = "X", bonus_quadorder = data.quadorder)
        elseif data.dependencies == "T"
            function rhs_function_t() # result = F(v) = f*operator(v) = f*input
                temp = zeros(T,ncomponents)
                function closure(result,input,t)
                    eval_data!(temp, data, nothing, t)
                    result[1] = 0
                    for j = 1 : ncomponents
                        result[1] += temp[j]*input[j] 
                    end
                    return nothing
                end
            end    
            action = Action(rhs_function_t(),[1, ncomponents]; dependencies = "T", bonus_quadorder = data.quadorder)
        else
            function rhs_function_c() # result = F(v) = f*operator(v) = f*input
                temp = zeros(T,ncomponents)
                function closure(result,input)
                    eval_data!(temp, data, nothing, nothing)
                    result[1] = 0
                    for j = 1 : ncomponents
                        result[1] += temp[j]*input[j] 
                    end
                    return nothing
                end
            end    
            action = Action(rhs_function_c(),[1, ncomponents]; dependencies = "", bonus_quadorder = data.quadorder)
        end
    end
    return action
end
