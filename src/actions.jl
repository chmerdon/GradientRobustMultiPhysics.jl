
abstract type AbstractAction end


# dummy action that does nothing (but can be only used with certain assembly patterns)
struct NoAction <: AbstractAction
    name::String
    bonus_quadorder::Base.RefValue{Int}
end

"""
````
function NoAction()
````

Creates a NoAction that causes the assembly pattern to ignore the action assembly.
"""
function NoAction(; name = "no action", quadorder = 0)
    return NoAction(name,Ref(quadorder))
end

## dummy array that is infinitely long and only has nothing entries
#struct InfNothingArray
#    val::Nothing
#end
#Base.getindex(::InfNothingArray,i) = nothing

struct Action{T <: Real,KernelType <: UserData{<:AbstractActionKernel}, nsizes} <: AbstractAction
    kernel::KernelType
    name::String
    citem::Base.RefValue{Int}
    cregion::Base.RefValue{Int}
    bonus_quadorder::Int
    argsizes::SVector{nsizes,Int}
end

# actions that do depend on t
struct TAction{T <: Real,KernelType <: UserData{<:AbstractActionKernel},nsizes} <: AbstractAction
    kernel::KernelType
    name::String
    citem::Base.RefValue{Int}
    cregion::Base.RefValue{Int}
    ctime::Base.RefValue{T}
    bonus_quadorder::Int
    argsizes::SVector{nsizes,Int}
end

# actions that do depend on x
# and require additional managament of global evaluation points
struct XAction{T <: Real,KernelType <: UserData{<:AbstractActionKernel},nsizes} <: AbstractAction
    kernel::KernelType
    name::String
    citem::Base.RefValue{Int}
    cregion::Base.RefValue{Int}
    bonus_quadorder::Int
    argsizes::SVector{nsizes,Int}
    x::Array{Array{T,1},1}
end

# actions that do depend on x and t
# and require additional managament of global evaluation points
struct XTAction{T <: Real,KernelType <: UserData{<:AbstractActionKernel},nsizes} <: AbstractAction
    kernel::KernelType
    name::String
    citem::Base.RefValue{Int}
    cregion::Base.RefValue{Int}
    ctime::Base.RefValue{T}
    bonus_quadorder::Int
    argsizes::SVector{nsizes,Int}
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
that should match the number format of the used quadrature rules and grid coordinates in the mesh (usually T).
"""
function Action{T}(kernel::UserData{<:AbstractActionKernel}; name = "user action") where {T}
    citem::Int = 0
    cregion::Int = 0
    ctime::T = 0
    if is_xdependent(kernel)
        if is_timedependent(kernel)
            return XTAction{T,typeof(kernel),length(kernel.dimensions)}(kernel, name, Ref(citem), Ref(cregion), Ref(ctime), kernel.quadorder[1], kernel.dimensions, Array{Array{T,1},1}(undef,0))
        else
            return XAction{T,typeof(kernel),length(kernel.dimensions)}(kernel, name, Ref(citem), Ref(cregion), kernel.quadorder[1], kernel.dimensions, Array{Array{T,1},1}(undef,0))
        end
    else
        if is_timedependent(kernel)
            return TAction{T,typeof(kernel),length(kernel.dimensions)}(kernel, name, Ref(citem), Ref(cregion), Ref(ctime), kernel.quadorder[1], kernel.dimensions)
        else
            return Action{T,typeof(kernel),length(kernel.dimensions)}(kernel, name, Ref(citem), Ref(cregion), kernel.quadorder[1], kernel.dimensions)
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
that should match the number format of the used quadrature rules and grid coordinates in the mesh (usually T).
"""
function Action{T}(kernel_function::Function, dimensions; name = "user action", dependencies = "", quadorder = 0) where {T}
    kernel = ActionKernel(kernel_function, dimensions; name = name * " (kernel)", dependencies = dependencies, quadorder = quadorder)
    return Action{T}(kernel; name = name)
end


# set_time! is called in the beginning of every operator assembly
function set_time!(C::AbstractAction, time)
    return nothing
end

function set_time!(C::Union{XTAction,TAction}, time)
    if is_timedependent(C.kernel)
        C.ctime[] = time
    end
    return nothing
end


function update_action!(C::NoAction, FEBE::FEBasisEvaluator, qitem, vitem, region)
    return nothing
end

function update_action!(C::Action, FEBE::FEBasisEvaluator, qitem, vitem, region)
    if is_regiondependent(C.kernel)
        C.cregion[] = region
    end
    if is_itemdependent(C.kernel)
        C.citem[] = vitem
    end
    return nothing
end

function update_action!(C::TAction, FEBE::FEBasisEvaluator, qitem::Int, vitem::Int, region::Int) 
    if is_regiondependent(C.kernel)
        C.cregion[] = region
    end
    if is_itemdependent(C.kernel)
        C.citem[] = vitem
    end
    return nothing
end

function update_action!(C::Union{XAction{T}, XTAction{T}}, FEBE::FEBasisEvaluator, qitem, vitem, region) where {T}
    if is_regiondependent(C.kernel)
        C.cregion[] = region
    end
    if is_itemdependent(C.kernel)
        C.citem[] = vitem
    end
    # compute global coordinates for function evaluation
    if FEBE.L2G.citem[] != qitem 
        update_trafo!(FEBE.L2G, qitem)
    end
    # we don't know at contruction time how many quadrature points are needed
    # so we expand the array here if needed
    while length(C.x) < length(FEBE.xref)
        push!(C.x,zeros(T,size(FEBE.FE.xgrid[Coordinates],1)))
    end  
    for i = 1 : length(FEBE.xref)
        eval_trafo!(C.x[i],FEBE.L2G,FEBE.xref[i])
    end    
    return nothing
end

function apply_action!(result, input, C::AbstractAction, i, xref) where {T}
    return nothing
end

function apply_action!(result, input, C::Action{T}, i, xref) where {T}
    eval_data!(result, C.kernel, input, nothing, nothing, C.cregion[], C.citem[], xref);
    return nothing
end

const test_x = [0,0]

function apply_action!(result, input, C::XAction{T}, i, xref) where {T}
    eval_data!(result, C.kernel, input, C.x[i], nothing, C.cregion[], C.citem[], xref)
    return nothing
end

function apply_action!(result, input, C::TAction{T}, i, xref) where {T}
    eval_data!(result, C.kernel, input, nothing, C.ctime[], C.cregion[], C.citem[], xref);
    return nothing
end

function apply_action!(result, input, C::XTAction{T}, i, xref) where {T}
    eval_data!(result, C.kernel, input, C.x[i], C.ctime[], C.cregion[], C.citem[], xref);
    return nothing
end

## apply! function for action in nonlinear assemblies

function apply_action!(result, input_last, input_ansatz, C::Action{T}, i, xref) where {T}
    eval_data!(result, C.kernel, input_last, input_ansatz, nothing, nothing, C.cregion[], C.citem[], xref);
    return nothing
end

function apply_action!(result, input_last, input_ansatz, C::XAction{T}, i, xref) where {T}
    eval_data!(result, C.kernel, input_last, input_ansatz, C.x[i], nothing, C.cregion[], C.citem[], xref);
    return nothing
end

function apply_action!(result, input_last, input_ansatz, C::TAction{T}, i, xref) where {T}
    eval_data!(result, C.kernel, input_last, input_ansatz, nothing, C.ctime[], C.cregion[], C.citem[], xref);
    return nothing
end

function apply_action!(result, input_last, input_ansatz, C::XTAction{T}, i, xref) where {T}
    eval_data!(result, C.kernel, input_last, input_ansatz, C.x[i], C.ctime[], C.cregion[], C.citem[], xref);
    return nothing
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
        action_kernel = ActionKernel(rhs_function_ext(),[1, ncomponents]; dependencies = "XTRIL", quadorder = data.quadorder)
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
            action_kernel = ActionKernel(rhs_function_xt(),[1, ncomponents]; dependencies = "XT", quadorder = data.quadorder)
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
            action_kernel = ActionKernel(rhs_function_x(),[1, ncomponents]; dependencies = "X", quadorder = data.quadorder)
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
            action_kernel = ActionKernel(rhs_function_t(),[1, ncomponents]; dependencies = "T", quadorder = data.quadorder)
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
            action_kernel = ActionKernel(rhs_function_c(),[1, ncomponents]; dependencies = "", quadorder = data.quadorder)
        end
    end
    return Action{T}(action_kernel)
end
